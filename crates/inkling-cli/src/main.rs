use std::time::Instant;

use anyhow::{Result, bail, ensure};
use inkling_cuda::Gpu;
use inkling_cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("smoke") => smoke(),
        _ => bail!("usage: inkling smoke"),
    }
}

/// Device info, a correctness check on a trivial kernel, and the achievable DRAM read bandwidth (decode roofline).
fn smoke() -> Result<()> {
    let gpu = Gpu::new(0)?;
    let i = &gpu.info;
    println!(
        "GPU 0: {} | sm_{}{} | {} SMs | {:.1} GiB | L2 {} MiB",
        i.name,
        i.compute_capability.0,
        i.compute_capability.1,
        i.sm_count,
        i.total_mem as f64 / (1u64 << 30) as f64,
        i.l2_bytes >> 20
    );

    // axpy correctness
    let n = 1usize << 20;
    let x: Vec<f32> = (0..n).map(|k| k as f32 * 0.5).collect();
    let y: Vec<f32> = (0..n).map(|k| 1.0 - k as f32).collect();
    let dx = gpu.stream.clone_htod(&x)?;
    let mut dy = gpu.stream.clone_htod(&y)?;
    let f = gpu.func("ink_axpy_f32")?;
    let (a, n64) = (2.0f32, n as u64);
    let mut l = gpu.stream.launch_builder(&f);
    l.arg(&a).arg(&dx).arg(&mut dy).arg(&n64);
    unsafe { l.launch(LaunchConfig::for_num_elems(n as u32)) }?;
    let out = gpu.stream.clone_dtoh(&dy)?;
    let bad = (0..n).filter(|&k| (out[k] - (a * x[k] + y[k])).abs() > 1e-3).count();
    ensure!(bad == 0, "axpy mismatch in {bad} elements");
    println!("axpy: OK");

    // read bandwidth: 2 GiB buffer, best of several passes
    let bytes = 2usize << 30;
    let words = bytes / 16;
    let src = gpu.stream.alloc_zeros::<u8>(bytes)?;
    let sink = gpu.stream.alloc_zeros::<u32>(4096)?;
    let rb = gpu.func("ink_read_bw")?;
    let cfg = LaunchConfig { grid_dim: (i.sm_count as u32 * 8, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
    let words64 = words as u64;
    let mut best = f64::MAX;
    for _ in 0..6 {
        gpu.stream.synchronize()?;
        let t = Instant::now();
        let mut l = gpu.stream.launch_builder(&rb);
        l.arg(&src).arg(&words64).arg(&sink);
        unsafe { l.launch(cfg) }?;
        gpu.stream.synchronize()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("DRAM read bandwidth: {:.0} GB/s (decode roofline)", bytes as f64 / best / 1e9);
    Ok(())
}
