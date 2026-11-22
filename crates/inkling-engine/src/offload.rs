//! CPU offload planner: which MoE layers keep their expert weights in system RAM (run by `cpu_moe` during decode,
//! streamed to the GPU in expert groups during batched prompt processing) so the rest fits in VRAM.
//!
//! The expert tensors are ~90% of the weights, and every token reads only 8 of 128 experts per layer visit, so a
//! layer on the CPU costs about one expert-set read from RAM per visit. The prelude and coda layers run once per
//! token, the looped core layers three times, so prelude/coda layers are offloaded first ("hotness" order), larger
//! layers first within each group (fewer layers for the same VRAM, less per-visit overhead).
//!
//! INKLING_OFFLOAD (CLI --offload): auto (default; only what does not fit), off, or a number of layers.
//! INKLING_VRAM_LIMIT=<GiB>: plan as if the GPU had that much memory (simulate a smaller card).
//! INKLING_RAM_LIMIT=<GiB>: cap the measured free system RAM (test the RAM check).

use anyhow::{Result, bail, ensure};
use inkling_core::{Config, Gguf};

const GIB: f64 = (1u64 << 30) as f64;

/// System RAM to leave free beyond the offloaded experts (OS, other applications, load-time buffers).
const RAM_MARGIN: usize = 2 << 30;

/// Experts per prefill streaming group (two group buffers live in VRAM while any layer is offloaded).
pub const STREAM_GROUP: usize = 16;

#[derive(Clone, Debug)]
pub struct Plan {
    /// per physical layer: experts in system RAM
    pub on_cpu: Vec<bool>,
    /// bytes of expert weights held in system RAM
    pub host_bytes: usize,
    /// VRAM the plan needs (weights left on the GPU + KV + fixed + streaming buffers)
    pub vram_need: usize,
    /// VRAM free for the engine (after INKLING_VRAM_LIMIT)
    pub vram_free: usize,
    /// bytes of one prefill streaming buffer (0 without offload)
    pub group_bytes: usize,
}

impl Plan {
    pub fn n_cpu(&self) -> usize {
        self.on_cpu.iter().filter(|&&c| c).count()
    }
}

/// Expert-weight bytes (gate_up + down) of physical layer `i`.
pub fn expert_bytes(g: &Gguf, i: usize) -> usize {
    ["ffn_gate_up_exps.weight", "ffn_down_exps.weight"].iter().filter_map(|s| g.tensor(&format!("blk.{i}.{s}"))).map(|t| t.n_bytes).sum()
}

/// Order in which layers move to the CPU: prelude/coda (one visit per token) before the looped core (three
/// visits), larger layers first within each group.
pub fn hotness_order(g: &Gguf, cfg: &Config) -> Vec<usize> {
    let core = |i: usize| (cfg.core_start..cfg.core_end).contains(&i);
    let mut order: Vec<usize> = (0..cfg.n_phys).collect();
    // stable sort: core last, then descending size, then coda (higher index) first
    order.sort_by_key(|&i| (core(i), std::cmp::Reverse(expert_bytes(g, i)), std::cmp::Reverse(i)));
    order
}

/// Available physical memory (Windows: ullAvailPhys, which counts the standby cache as available; Linux:
/// MemAvailable). None when unknown.
pub fn free_ram() -> Option<usize> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
        }
        let mut m: MemoryStatusEx = unsafe { std::mem::zeroed() };
        m.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { Some(m.avail_phys as usize) } else { None }
    }
    #[cfg(not(windows))]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

/// Free VRAM as seen by the planner: the device's free memory, reduced by INKLING_VRAM_LIMIT (GiB) to what a card
/// of that size with the same other usage would have.
pub fn effective_free_vram(free: usize, total: usize) -> Result<usize> {
    match std::env::var("INKLING_VRAM_LIMIT") {
        Ok(v) if !v.is_empty() => {
            let gib: f64 = v.trim().parse().map_err(|_| anyhow::anyhow!("INKLING_VRAM_LIMIT: expected GiB, got `{v}`"))?;
            let limit = (gib * GIB) as usize;
            Ok(free.saturating_sub(total.saturating_sub(limit)))
        }
        _ => Ok(free),
    }
}

/// Choose the offloaded layers. `base_need` is the VRAM needed with every weight on the GPU (weights + KV +
/// fixed buffers), `margin` the headroom to keep free.
pub fn plan(g: &Gguf, cfg: &Config, base_need: usize, vram_free: usize, margin: usize, n_ctx: usize) -> Result<Plan> {
    let mode = std::env::var("INKLING_OFFLOAD").unwrap_or_default().trim().to_ascii_lowercase();
    let order = hotness_order(g, cfg);
    let group = |on: &[bool]| -> usize {
        // one streaming group of the largest offloaded layer
        (0..cfg.n_phys).filter(|&i| on[i]).map(|i| expert_bytes(g, i) / cfg.n_expert * STREAM_GROUP).max().unwrap_or(0)
    };
    let evaluate = |n: usize| -> Plan {
        let mut on = vec![false; cfg.n_phys];
        for &i in &order[..n] {
            on[i] = true;
        }
        let host_bytes: usize = (0..cfg.n_phys).filter(|&i| on[i]).map(|i| expert_bytes(g, i)).sum();
        let group_bytes = group(&on);
        Plan { vram_need: base_need - host_bytes + 2 * group_bytes, host_bytes, vram_free, group_bytes, on_cpu: on }
    };
    let fits = |p: &Plan| vram_free >= p.vram_need + margin;
    let skip_check = std::env::var_os("INKLING_SKIP_VRAM_CHECK").is_some();
    let p = match mode.as_str() {
        "" | "auto" => {
            if skip_check {
                evaluate(0)
            } else {
                match (0..=cfg.n_phys).map(evaluate).find(fits) {
                    Some(p) => p,
                    None => {
                        let all = evaluate(cfg.n_phys);
                        bail!(
                            "not enough free GPU memory even with every MoE layer on the CPU: {:.1} GiB free, about {:.1} GiB \
                             needed at context {n_ctx}. Use a smaller --ctx or a smaller KV cache (--kv q4), or close other \
                             GPU-heavy apps.",
                            vram_free as f64 / GIB,
                            (all.vram_need + margin) as f64 / GIB
                        )
                    }
                }
            }
        }
        "off" | "none" | "0" => evaluate(0),
        "all" => evaluate(cfg.n_phys),
        n => {
            let n: usize = n.parse().map_err(|_| anyhow::anyhow!("--offload: expected auto, off, all or a number of layers, got `{n}`"))?;
            ensure!(n <= cfg.n_phys, "--offload {n}: the model has {} MoE layers", cfg.n_phys);
            evaluate(n)
        }
    };
    if std::env::var_os("INKLING_VRAM_DEBUG").is_some() {
        eprintln!(
            "offload plan: vram free {:.2} GiB, base need {:.2} GiB, plan need {:.2} GiB (+{:.2} margin), {} layers on CPU ({:.2} GiB), group {:.0} MiB, order {:?}",
            vram_free as f64 / GIB,
            base_need as f64 / GIB,
            p.vram_need as f64 / GIB,
            margin as f64 / GIB,
            p.n_cpu(),
            p.host_bytes as f64 / GIB,
            p.group_bytes as f64 / (1 << 20) as f64,
            order.iter().map(|&i| (i, expert_bytes(g, i) >> 20)).collect::<Vec<_>>()
        );
    }
    if p.n_cpu() > 0 {
        let need = p.host_bytes + RAM_MARGIN;
        // INKLING_RAM_LIMIT=<GiB> caps the measured free RAM (testing)
        let cap = std::env::var("INKLING_RAM_LIMIT").ok().and_then(|v| v.trim().parse::<f64>().ok()).map(|g| (g * GIB) as usize);
        match free_ram().map(|f| cap.map_or(f, |c| f.min(c))) {
            Some(free) => ensure!(
                free >= need,
                "not enough free system RAM for CPU offload: {:.1} GiB free, {:.1} GiB needed to keep {} MoE layers in RAM \
                 (plus {:.0} GiB headroom). Close other applications, or use a smaller --ctx / --kv q4 so fewer layers \
                 need offloading.",
                free as f64 / GIB,
                p.host_bytes as f64 / GIB,
                p.n_cpu(),
                RAM_MARGIN as f64 / GIB
            ),
            None => eprintln!("warning: could not read free system RAM; offloading {:.1} GiB unchecked", p.host_bytes as f64 / GIB),
        }
    }
    Ok(p)
}
