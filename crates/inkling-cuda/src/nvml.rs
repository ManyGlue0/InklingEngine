//! Device-wide free VRAM through NVML (ships with the NVIDIA driver, loaded at runtime).
//!
//! On Windows (WDDM) `cuMemGetInfo` reports this process's video-memory budget, which counts memory the driver can
//! take from other processes by paging it into system RAM. So a second model load "fits" and the other process
//! (or both) end up thrashing. NVML reports what is actually in use on the device, like nvidia-smi.

use std::ffi::{CString, c_char, c_int, c_void};

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

type Init = unsafe extern "C" fn() -> c_int;
type ByPci = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int;
type MemInfo = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> c_int;
type Shutdown = unsafe extern "C" fn() -> c_int;

const LIBS: &[&str] = if cfg!(windows) {
    &["nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"]
} else {
    &["libnvidia-ml.so.1", "libnvidia-ml.so"]
};

/// Free bytes on the device with this PCI bus id ("0000:01:00.0"), or None when NVML is unavailable.
pub fn free_bytes(pci_bus_id: &str) -> Option<u64> {
    let lib = LIBS.iter().find_map(|name| unsafe { libloading::Library::new(name) }.ok())?;
    unsafe {
        let init: libloading::Symbol<Init> = lib.get(b"nvmlInit_v2\0").ok()?;
        let by_pci: libloading::Symbol<ByPci> = lib.get(b"nvmlDeviceGetHandleByPciBusId_v2\0").ok()?;
        let mem_info: libloading::Symbol<MemInfo> = lib.get(b"nvmlDeviceGetMemoryInfo\0").ok()?;
        let shutdown: libloading::Symbol<Shutdown> = lib.get(b"nvmlShutdown\0").ok()?;
        if init() != 0 {
            return None;
        }
        let id = CString::new(pci_bus_id).ok()?;
        let mut dev: *mut c_void = std::ptr::null_mut();
        let mut mem = NvmlMemory::default();
        let ok = by_pci(id.as_ptr(), &mut dev) == 0 && mem_info(dev, &mut mem) == 0;
        shutdown();
        ok.then_some(mem.free)
    }
}
