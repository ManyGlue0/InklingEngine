//! CUDA layer for InklingEngine: device context, the embedded kernel module, raw device buffers and launches.
//!
//! The engine runs everything on one stream and later captures the token step into a CUDA graph, so buffers are
//! addressed by raw device pointers and cudarc's per-buffer event tracking is disabled.

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

impl HostMapped {
    pub fn read_i32(&self, idx: usize) -> i32 {
        assert!((idx + 1) * 4 <= self.bytes);
        unsafe { std::ptr::read_volatile((self.host as *const i32).add(idx)) }
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
            };
        }
        let cfg = LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
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
