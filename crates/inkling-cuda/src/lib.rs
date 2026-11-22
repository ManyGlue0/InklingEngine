//! CUDA layer for InklingEngine: device context, the embedded kernel module, raw device buffers and launches.
//!
//! The engine runs everything on one stream and later captures the token step into a CUDA graph, so buffers are
//! addressed by raw device pointers and cudarc's per-buffer event tracking is disabled.

mod nvml;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub use cudarc;

/// Every kernel in `kernels/*.cu`, compiled by build.rs into one fatbin.
static KERNELS_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inkling_kernels.fatbin"));

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_mem: usize,
    pub sm_count: i32,
    pub l2_bytes: i32,
}

/// Up to 8 device pointers passed by value (kernel-side `struct Ptrs8 { const float* p[8]; }`).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Ptrs8(pub [u64; 8]);
unsafe impl DeviceRepr for Ptrs8 {}

/// One segment of a multi-matrix Q8 matvec (`MVSeg` in decode.cu).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MvSeg {
    pub qs: u64,
    pub d: u64,
    pub x: u64,
    pub y: u64,
    pub n: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Mv4 {
    pub s: [MvSeg; 4],
    pub nseg: i32,
    pub k: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MvSegB {
    pub w: u64,
    pub x: u64,
    pub y: u64,
    pub n: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MvB4 {
    pub s: [MvSegB; 4],
    pub nseg: i32,
    pub k: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ExpW {
    pub qs: u64,
    pub sc: u64,
    pub scale: u64,
    pub fmt: i32,
    /// prefill grouped GEMMs: only experts in [lo, hi) (erange = lo | hi << 16; 0 = all)
    pub erange: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Ctrl4 {
    pub c: [u64; 4],
    pub m: [u64; 4],
    pub wd: [u64; 4],
    pub wg: [u64; 4],
    pub wf: [u64; 4],
    pub g: [u64; 4],
    pub out: [u64; 4],
    pub n: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MixArgs {
    pub st: [u64; 8],
    pub q: u64,
    pub out: u64,
    pub outn: u64,
    pub nw: [u64; 4],
    pub n: i32,
    pub nq: i32,
    pub d: i32,
    pub eps: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MvSegQ {
    pub qs: u64,
    pub d: u64,
    pub xq: u64,
    pub xd: u64,
    pub y: u64,
    pub n: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MvQ4 {
    pub s: [MvSegQ; 4],
    pub nseg: i32,
    pub k: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MixArgsQ {
    pub m: MixArgs,
    pub xq: u64,
    pub xd: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct NormQ4 {
    pub x: [u64; 4],
    pub w: [u64; 4],
    pub y: [u64; 4],
    pub xq: [u64; 4],
    pub xd: [u64; 4],
}
/// One segment of the prefill int8 GEMM (mirrors `GemmSeg` in prefill.cu).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GemmSeg {
    pub w: u64,
    pub wd: u64,
    pub x: u64,
    pub xd: u64,
    pub y: u64,
    pub n: i32,
    pub ldy: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GemmArgs {
    pub s: [GemmSeg; 4],
    pub nseg: i32,
    pub t: i32,
    pub k: i32,
}
unsafe impl DeviceRepr for GemmArgs {}
unsafe impl DeviceRepr for MvQ4 {}
unsafe impl DeviceRepr for MixArgsQ {}
unsafe impl DeviceRepr for NormQ4 {}
unsafe impl DeviceRepr for Mv4 {}
unsafe impl DeviceRepr for MvB4 {}
unsafe impl DeviceRepr for ExpW {}
unsafe impl DeviceRepr for Ctrl4 {}
unsafe impl DeviceRepr for MixArgs {}

/// One kernel argument.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    F32(f32),
    Ptrs(Ptrs8),
    Mv(Mv4),
    MvB(MvB4),
    Exp(ExpW),
    Ctrl(Ctrl4),
    Mix(MixArgs),
    MvQ(MvQ4),
    MixQ(MixArgsQ),
    NormQ(NormQ4),
    Gemm(GemmArgs),
}

impl From<&DevBuf> for Arg {
    fn from(b: &DevBuf) -> Self {
        Arg::Ptr(b.ptr)
    }
}

/// A device allocation addressed by raw pointer.
pub struct DevBuf {
    _slice: CudaSlice<u8>,
    pub ptr: u64,
    pub bytes: usize,
}

impl DevBuf {
    /// Pointer `off` bytes into the buffer.
    pub fn at(&self, off: usize) -> u64 {
        debug_assert!(off <= self.bytes);
        self.ptr + off as u64
    }
}

/// Page-locked host memory mapped into the device address space: kernels write it directly, the host reads it
/// after synchronizing on an event recorded behind those kernels.
pub struct HostMapped {
    host: *mut u8,
    pub dev: u64,
    pub bytes: usize,
}

unsafe impl Send for HostMapped {}
unsafe impl Sync for HostMapped {}

impl HostMapped {
    pub fn read_i32(&self, idx: usize) -> i32 {
        assert!((idx + 1) * 4 <= self.bytes);
        unsafe { std::ptr::read_volatile((self.host as *const i32).add(idx)) }
    }

    /// Host view of the buffer (the device may be reading it: only write while no kernel uses it).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host, self.bytes) }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host, self.bytes) }
    }

    pub fn host_ptr(&self) -> *mut u8 {
        self.host
    }
}

impl Drop for HostMapped {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::result::free_host(self.host as *mut std::ffi::c_void);
        }
    }
}

/// One GPU: context, main stream, the loaded kernel module and its functions.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub info: DeviceInfo,
    funcs: std::sync::Mutex<HashMap<String, CudaFunction>>,
}

impl Gpu {
    /// Free VRAM that can really be used: the smaller of CUDA's figure and NVML's device-wide one (on Windows CUDA
    /// counts other processes' memory as available because the driver can page it out to system RAM).
    pub fn free_vram(&self) -> Result<usize> {
        let (cu_free, _) = cudarc::driver::result::mem_get_info()?;
        let mut buf = [0 as std::ffi::c_char; 32];
        let r = unsafe { cudarc::driver::sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as i32, self.ctx.cu_device()) };
        let pci = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
        let nv_free = if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS { nvml::free_bytes(&pci) } else { None };
        Ok(nv_free.map_or(cu_free, |n| cu_free.min(n as usize)))
    }

    /// Total VRAM of the device.
    pub fn total_vram(&self) -> Result<usize> {
        Ok(cudarc::driver::result::mem_get_info()?.1)
    }

    /// Enqueue a host function on the stream (capturable into a graph as a host node). `f` must stay valid for
    /// as long as any graph containing the node can run, must not call CUDA, and must not unwind.
    pub fn launch_host(&self, f: unsafe extern "C" fn(*mut std::ffi::c_void), data: *mut std::ffi::c_void) -> Result<()> {
        unsafe { cudarc::driver::sys::cuLaunchHostFunc(self.stream.cu_stream(), Some(f), data) }.result()?;
        Ok(())
    }

    /// Asynchronous host -> device copy from raw host memory (pinned for a true async DMA).
    ///
    /// # Safety
    /// `src` must point to `bytes` readable bytes that stay valid until the copy has run.
    pub unsafe fn copy_htod_raw(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        unsafe { cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, src as *const std::ffi::c_void, bytes, self.stream.cu_stream()) }.result()?;
        Ok(())
    }

    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).context("creating CUDA context (is an NVIDIA driver installed?)")?;
        // single stream + graph capture: cudarc's cross-stream event bookkeeping is unnecessary overhead
        unsafe { ctx.disable_event_tracking() };
        let attr = |a| ctx.attribute(a).map_err(anyhow::Error::from);
        let info = DeviceInfo {
            ordinal,
            name: ctx.name()?,
            compute_capability: (
                attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            total_mem: ctx.total_mem()?,
            sm_count: attr(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            l2_bytes: attr(Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
        };
        let (major, minor) = info.compute_capability;
        anyhow::ensure!(
            (major, minor) >= (7, 5),
            "{} has compute capability {major}.{minor}; InklingEngine needs 7.5 or newer (Turing / RTX 20xx and later)",
            info.name
        );
        let module = ctx
            .load_module(Ptx::from_binary(KERNELS_FATBIN.to_vec()))
            .context("loading kernel fatbin: this build may not include your GPU architecture (see INKLING_CUDA_ARCHS)")?;
        // a real stream: the legacy default stream cannot be captured into a CUDA graph
        let stream = ctx.new_stream()?;
        Ok(Self { ctx, stream, module, info, funcs: Default::default() })
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        let mut f = self.funcs.lock().unwrap();
        if let Some(k) = f.get(name) {
            return Ok(k.clone());
        }
        let k = self.module.load_function(name).with_context(|| format!("kernel `{name}` not found"))?;
        f.insert(name.to_owned(), k.clone());
        Ok(k)
    }

    /// Resident blocks per SM for kernel `name` at `block` threads (sizes persistent grids that spin on barriers).
    pub fn max_blocks_per_sm(&self, name: &str, block: u32) -> Result<u32> {
        Ok(self.func(name)?.occupancy_max_active_blocks_per_multiprocessor(block, 0, None)?)
    }

    /// Allocate `bytes` of zeroed device memory.
    pub fn alloc(&self, bytes: usize) -> Result<DevBuf> {
        let slice = self.stream.alloc_zeros::<u8>(bytes.max(1))?;
        let ptr = raw_ptr(&slice, &self.stream);
        Ok(DevBuf { ptr, bytes, _slice: slice })
    }

    /// Allocate zeroed host-mapped pinned memory.
    pub fn alloc_host_mapped(&self, bytes: usize) -> Result<HostMapped> {
        self.ctx.bind_to_thread()?;
        const PORTABLE: u32 = 0x01;
        const DEVICEMAP: u32 = 0x02;
        let host = unsafe { cudarc::driver::result::malloc_host(bytes.max(4), PORTABLE | DEVICEMAP) }? as *mut u8;
        unsafe { std::ptr::write_bytes(host, 0, bytes.max(4)) };
        let mut dev: cudarc::driver::sys::CUdeviceptr = 0;
        unsafe { cudarc::driver::sys::cuMemHostGetDevicePointer_v2(&mut dev, host as *mut std::ffi::c_void, 0) }.result()?;
        Ok(HostMapped { host, dev, bytes })
    }

    /// Allocate host-mapped pinned memory without zeroing it (large buffers the caller fills entirely).
    pub fn alloc_host_mapped_uninit(&self, bytes: usize) -> Result<HostMapped> {
        self.ctx.bind_to_thread()?;
        let host = unsafe { cudarc::driver::result::malloc_host(bytes.max(4), 0x01 | 0x02) }? as *mut u8;
        let mut dev: cudarc::driver::sys::CUdeviceptr = 0;
        unsafe { cudarc::driver::sys::cuMemHostGetDevicePointer_v2(&mut dev, host as *mut std::ffi::c_void, 0) }.result()?;
        Ok(HostMapped { host, dev, bytes })
    }

    /// Upload host bytes into a new device buffer.
    pub fn upload(&self, data: &[u8]) -> Result<DevBuf> {
        let slice = self.stream.clone_htod(data)?;
        let ptr = raw_ptr(&slice, &self.stream);
        Ok(DevBuf { ptr, bytes: data.len(), _slice: slice })
    }

    pub fn upload_f32(&self, data: &[f32]) -> Result<DevBuf> {
        self.upload(as_bytes(data))
    }

    /// Copy host bytes into an existing device location.
    pub fn write(&self, dst: u64, data: &[u8]) -> Result<()> {
        unsafe { cudarc::driver::result::memcpy_htod_async(dst, data, self.stream.cu_stream()) }?;
        Ok(())
    }

    /// Copy `n` f32 values from device to host (synchronizes the stream).
    pub fn read_f32(&self, src: u64, n: usize) -> Result<Vec<f32>> {
        let mut out = vec![0f32; n];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4),
                src,
                self.stream.cu_stream(),
            )
        }?;
        self.stream.synchronize()?;
        Ok(out)
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    /// Launch kernel `name` with a 1-D/2-D grid.
    pub fn launch(&self, name: &str, grid: (u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        self.launch3(name, (grid.0, grid.1, 1), block, args)
    }

    /// Launch with a 3-D grid.
    pub fn launch3(&self, name: &str, grid: (u32, u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        let f = self.func(name)?;
        let mut l = self.stream.launch_builder(&f);
        for a in args {
            match a {
                Arg::Ptr(p) => l.arg(p),
                Arg::I32(v) => l.arg(v),
                Arg::F32(v) => l.arg(v),
                Arg::Ptrs(v) => l.arg(v),
                Arg::Mv(v) => l.arg(v),
                Arg::MvB(v) => l.arg(v),
                Arg::Exp(v) => l.arg(v),
                Arg::Ctrl(v) => l.arg(v),
                Arg::Mix(v) => l.arg(v),
                Arg::MvQ(v) => l.arg(v),
                Arg::MixQ(v) => l.arg(v),
                Arg::NormQ(v) => l.arg(v),
                Arg::Gemm(v) => l.arg(v),
            };
        }
        let cfg = LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 };
        unsafe { l.launch(cfg) }.with_context(|| format!("launching {name}"))?;
        Ok(())
    }
}

fn raw_ptr(slice: &CudaSlice<u8>, stream: &CudaStream) -> u64 {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
