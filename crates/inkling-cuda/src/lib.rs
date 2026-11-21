//! CUDA layer for InklingEngine: device context and the embedded kernel module.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::driver::sys::CUdevice_attribute as Attr;
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

/// One GPU: context, main stream and the loaded kernel module.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub info: DeviceInfo,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).context("creating CUDA context (is an NVIDIA driver installed?)")?;
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
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, module, info })
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        self.module.load_function(name).with_context(|| format!("kernel `{name}` not found"))
    }
}
