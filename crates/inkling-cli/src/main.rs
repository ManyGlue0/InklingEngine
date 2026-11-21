use std::time::Instant;

use anyhow::{Result, bail, ensure};
use inkling_cuda::Gpu;
use inkling_cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("smoke") => smoke(),
        Some("inspect") if args.len() > 1 => inspect(&args[1]),
        Some("tokenize") if args.len() > 2 => tokenize(&args[1], &args[2], args.iter().any(|a| a == "--special")),
        _ => bail!("usage: inkling smoke | inkling inspect <model.gguf> | inkling tokenize <model.gguf> <file> [--special]"),
    }
}

/// Tokenize a file; prints comma-separated ids (same format as `llama-tokenize --ids`).
fn tokenize(model: &str, file: &str, special: bool) -> Result<()> {
    let g = inkling_core::Gguf::open(model)?;
    let t0 = Instant::now();
    let tok = inkling_core::Tokenizer::from_gguf(&g)?;
    let load = t0.elapsed();
    let text = std::fs::read_to_string(file)?;
    let t1 = Instant::now();
    let ids = tok.encode(&text, special)?;
    let enc = t1.elapsed();
    let s: Vec<String> = ids.iter().map(u32::to_string).collect();
    println!("[{}]", s.join(", "));
    eprintln!(
        "{} tokens | tokenizer load {:.0} ms | encode {:.1} ms | roundtrip {}",
        ids.len(),
        load.as_secs_f64() * 1e3,
        enc.as_secs_f64() * 1e3,
        if tok.decode(&ids, true) == text { "OK" } else { "MISMATCH" }
    );
    Ok(())
}

/// Parse the GGUF, print the config, the layer schedule and a weight-footprint breakdown.
fn inspect(path: &str) -> Result<()> {
    use std::collections::BTreeMap;
    let t = Instant::now();
    let g = inkling_core::Gguf::open(path)?;
    let cfg = inkling_core::Config::from_gguf(&g)?;
    println!("parsed GGUF v{} in {:.0} ms: {} tensors, {} metadata keys", g.version, t.elapsed().as_secs_f64() * 1e3, g.tensors.len(), g.kv.len());
    println!("{cfg:#?}");
    let sched = cfg.schedule();
    let full: Vec<usize> = sched.iter().filter(|p| !cfg.is_swa[p.phys]).map(|p| p.slot).collect();
    println!("{} passes per token; full-attention slots: {full:?}", sched.len());
    let line: Vec<String> = sched.iter().map(|p| format!("{}{}", p.phys, if p.pass > 0 { "'".repeat(p.pass) } else { String::new() })).collect();
    println!("schedule (phys, ' = repeated pass): {}", line.join(" "));

    // bytes read per token: core layers count n_passes times, experts only top-k of n_expert
    let mut by_type: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    let mut per_token = 0f64;
    for ti in &g.tensors {
        let visits = match ti.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<usize>().ok()) {
            Some(l) if (cfg.core_start..cfg.core_end).contains(&l) => cfg.n_passes as f64,
            _ => 1.0,
        };
        let frac = if ti.name.contains("_exps.weight") { cfg.n_expert_used as f64 / cfg.n_expert as f64 } else { 1.0 };
        let read = if ti.name == "token_embd.weight" { 0.0 } else { ti.n_bytes as f64 * visits * frac };
        per_token += read;
        let e = by_type.entry(format!("{:?}", ti.ty)).or_default();
        e.0 += ti.n_bytes;
        e.1 += read;
    }
    for (ty, (bytes, read)) in &by_type {
        println!("  {ty:>6}: {:>8.2} GiB stored, {:>7.1} MB read/token", *bytes as f64 / (1u64 << 30) as f64, read / 1e6);
    }
    println!("weights read per token ≈ {:.2} GB → roofline at 888 GB/s ≈ {:.0} tok/s", per_token / 1e9, 888e9 / per_token);
    Ok(())
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
