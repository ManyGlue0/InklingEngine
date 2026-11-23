//! InklingEngine forward pass. The decode step (docs/MODEL.md) is recorded once into a CUDA graph and replayed
//! for every token; per-token inputs (token id, position) live in a device buffer written before each replay.
//!
//! Activations feeding the quantized matvecs are int8 (q8_1 blocks of 32, as llama.cpp's mmvq), produced by the
//! kernel that computes them (depth-attention mix, norms, attention output, expert gate_up).

mod codec_tables;
pub mod codecs;
pub mod cpu_moe;
pub mod offload;
mod prefill;
pub mod repack;
pub mod vision;
pub mod weights;

use anyhow::{Context, Result, ensure};
use inkling_core::{Config, Gguf};
use inkling_cuda::cudarc::driver::{CudaGraph, sys};
use inkling_cuda::{Arg, Ctrl4, DevBuf, ExpW, GemmArgs, GemmSeg, Gpu, HostMapped, MixArgs, MixArgsQ, MvB4, MvQ4, MvSegB, MvSegQ, NormQ4};

pub use vision::{Image, ImageEmbed, Vision};
pub use weights::{ExpertFmt, LayerW, Weights};

/// Prompt token ids at or above this value are image-embedding rows (the low bits only identify the image for the
/// prompt cache); their input comes from `generate_mm`'s embeddings instead of the token table.
pub const EMBED_TOKEN_BASE: u32 = 0x8000_0000;

const F32: usize = 4;

/// VRAM the engine needs beyond weights and KV cache (activations, prefill and MoE buffers, CUDA module); measured
/// on an RTX 3090 with INKLING_VRAM_DEBUG.
const VRAM_FIXED: usize = 1408 << 20;
/// Headroom left free for the driver and other applications.
const VRAM_MARGIN: usize = 256 << 20;

/// Fail with a clear message when `need` bytes do not fit in `free` VRAM (INKLING_SKIP_VRAM_CHECK=1 skips it).
pub(crate) fn check_vram(free: usize, need: usize, what: &str) -> Result<()> {
    if std::env::var_os("INKLING_SKIP_VRAM_CHECK").is_some() {
        return Ok(());
    }
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    ensure!(
        free >= need + VRAM_MARGIN,
        "not enough free GPU memory: {:.1} GiB free, about {:.1} GiB needed for {what}. Is another InklingEngine \
         instance (or another GPU-heavy app) running? Close it, use a smaller --ctx, or let part of \
         the model run on the CPU (--offload auto). (INKLING_SKIP_VRAM_CHECK=1 skips this check.)",
        gib(free),
        gib(need + VRAM_MARGIN)
    );
    Ok(())
}

/// KV cache format of the full-attention slots (the sliding-window rings are always f16). Quantized formats use
/// groups of 32 channels with an f16 scale; see `kv_store` in decode.cu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvType {
    F16 = 0,
    Q6 = 1,
    Q4 = 2,
    Q8 = 3,
}

impl KvType {
    /// bytes per position and kv head (128 channels)
    pub fn head_bytes(self) -> usize {
        match self {
            KvType::F16 => 256,
            KvType::Q6 => 104,
            KvType::Q4 => 72,
            KvType::Q8 => 136,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q6 => "q6",
            KvType::Q4 => "q4",
            KvType::Q8 => "q8",
        }
    }
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "16" => KvType::F16,
            "q8" | "q8_0" | "8" => KvType::Q8,
            "q6" | "6" => KvType::Q6,
            "q4" | "q4_0" | "4" => KvType::Q4,
            _ => anyhow::bail!("unknown KV cache type `{s}` (expected q4, q6, q8 or f16)"),
        })
    }
    /// K and V types from INKLING_KV (both), INKLING_KV_K / INKLING_KV_V (each); default q6
    pub fn from_env() -> Result<(Self, Self)> {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty()).map(|v| Self::parse(&v)).transpose();
        let both = get("INKLING_KV")?.unwrap_or(KvType::Q6);
        Ok((get("INKLING_KV_K")?.unwrap_or(both), get("INKLING_KV_V")?.unwrap_or(both)))
    }
}

/// KV cache bytes per context position over all full-attention slots (what grows with --ctx).
pub fn kv_bytes_per_token(cfg: &Config, k: KvType, v: KvType) -> usize {
    let n_full = cfg.schedule().iter().filter(|p| !cfg.is_swa[p.phys]).count();
    n_full * cfg.n_head_kv * (k.head_bytes() + v.head_bytes())
}

/// Per-slot recurrent state: KV cache + short-conv histories.
struct Slot {
    k: DevBuf,
    v: DevBuf,
    kv_cap: usize,
    /// packed cache format for the kernels: K type in bits 0..3, V type in bits 4..7 (KvType values)
    kv_fmt: i32,
    conv: DevBuf, // [k: 3x1024][v: 3x1024][attn: 3xD][ffn: 3xD] floats
}

/// A q8_1 activation buffer: int8 values + one f32 scale per 32.
struct Q8Buf {
    q: DevBuf,
    d: DevBuf,
}

impl Q8Buf {
    fn new(gpu: &Gpu, n: usize) -> Result<Self> {
        Ok(Self { q: gpu.alloc(n)?, d: gpu.alloc(n / 32 * F32)? })
    }
    /// (values, scales) pointers of the j-th vector of length `len`
    fn at(&self, j: usize, len: usize) -> (u64, u64) {
        (self.q.at(j * len), self.d.at(j * len / 32 * F32))
    }
}

/// Token sampling parameters (mirrors `SampleParams` in sample.cu). temperature <= 0 or top_k == 1: greedy.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: i32,
    pub seed: u64,
    /// llama.cpp-style repetition penalty (1 = off) over the last `last_n` context tokens
    pub repeat_penalty: f32,
    /// OpenAI-style penalties over the same window
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// penalty window (clamped to 1024; 0 = off)
    pub last_n: i32,
}

impl Sampling {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 0, seed: 0, repeat_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, last_n: 0 }
    }
}

/// Penalty ring size (mirrors smp::RING in sample.cu).
const PEN_RING: usize = 1024;

/// What the device state currently holds, for prompt caching across `generate` calls.
#[derive(Default)]
struct PromptCache {
    /// tokens whose KV / conv state is on the device, in order (state is exactly "after these tokens")
    tokens: Vec<u32>,
    /// conv-state snapshot taken right after prefilling `snap_tokens` (the previous prompt minus its last token)
    snap_tokens: Option<Vec<u32>>,
}

/// CPU offload state: the expert executor, the host-mapped exchange block, one host-node context per layer, and
/// the prefill streaming buffers.
struct CpuOffload {
    moe: Box<cpu_moe::CpuMoe>,
    io: HostMapped,
    cbs: Vec<Box<cpu_moe::HostCb>>,
    /// two expert-group buffers (gate_up qs | gate_up sc | down qs | down sc), alternating
    bufs: [DevBuf; 2],
    copy_stream: std::sync::Arc<inkling_cuda::cudarc::driver::CudaStream>,
    plan: offload::Plan,
}

unsafe impl Send for CpuOffload {}

/// Upper bound of the decode attention splits (sizes the partials buffer).
const ATTN_SPLITS_MAX: usize = 64;

/// Scratch activations.
struct Act {
    x: DevBuf,
    a: DevBuf,      // mix outputs [3][D]
    c: DevBuf,      // control outputs [3][D]
    nrm: DevBuf,    // normalized attention inputs [3][D]
    nq: Q8Buf,      // ... and their q8_1 copies [3][D]
    msg: DevBuf,    // loop messages [4][D]
    g: DevBuf,      // control bottleneck [4][R]
    qrg: DevBuf,    // q (3072) | r (384) | gate (3072)
    sink: DevBuf,   // 24 | dynk 32 | dynv 32
    kraw: DevBuf,
    vraw: DevBuf,
    att: DevBuf,
    attq: Q8Buf,    // q8_1 attention output (wo input)
    proj: DevBuf,
    attn_out: DevBuf,
    f: DevBuf,
    f2: DevBuf,
    fnorm: DevBuf,
    fq: Q8Buf,      // q8_1 FFN / head input
    hq: Q8Buf,      // int8 expert hidden [k][F], scales per 16 values [k][F/16]
    sel: DevBuf,
    wts: DevBuf,
    rlog: DevBuf,
    counter: DevBuf,
    bar: DevBuf,    // grid barrier [arrivals, generation] of the persistent MoE kernel
    attn_part: DevBuf, // split-K attention partials [8 kv heads][ATTN_SPLITS_MAX][3 x (2 + 128)]
    attn_cnt: DevBuf,  // split-K attention arrival counters [8]
    part: DevBuf,   // expert hidden h as float [k][F] (persistent MoE kernel)
    ffn_out: DevBuf,
    xp: DevBuf,
    xprev: DevBuf,
    blocks: DevBuf, // block storage [16][D], indexed by push order within a token
    logits: DevBuf,
    step: DevBuf,   // i32 [token, pos, n_out, pad]
    prompt: DevBuf, // i32 [n_ctx] prompt tokens (device-side prefill loop)
    hist: HostMapped, // i32 [n_ctx] generated tokens, written by the GPU into host memory
    sparams: DevBuf,  // Sampling
    pen_ring: DevBuf, // i32 [PEN_RING] context tokens by position, for the sampler's penalties
    mask_on: DevBuf,  // i32: 1 = apply `mask` before sampling (constrained decoding), 0 = off
    mask: DevBuf,     // u32 [ceil(n_vocab / 32)] allowed tokens, one bit each (ink_apply_mask)
    snap: DevBuf,     // conv states of every slot (prompt-cache snapshot)
}

/// Which work follows the forward pass in a captured graph.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tail {
    /// nothing: logits stay on the device
    None,
    /// next prompt token becomes the input, pos += 1
    Prefill,
    /// greedy: argmax becomes the input and is appended to hist, pos += 1
    Greedy,
}

pub struct Engine {
    pub gpu: Gpu,
    pub cfg: Config,
    pub w: Weights,
    slots: Vec<Slot>,
    act: Act,
    pub n_ctx: usize,
    graph: Option<CudaGraph>,
    graph_pre: Option<CudaGraph>,
    graph_dec: Option<CudaGraph>,
    /// disable graph replay (debug captures need per-layer host callbacks)
    pub use_graph: bool,
    /// blocks of the persistent MoE kernel (all resident), 0 = use the two-kernel path
    moe_blocks: u32,
    /// position splits of the decode attention (grid y); 0 = the old one-block-per-head kernel
    attn_splits: u32,
    /// INKLING_MOE_TL: per-slot, per-block phase timestamps of the persistent MoE kernel (u64 [slot][block][8])
    pub moe_tl: Option<DevBuf>,
    /// batched prompt-processing buffers
    pre: Option<prefill::PrefillBufs>,
    cache: PromptCache,
    vision: Option<Vision>,
    /// image embeddings of the current request: (absolute position of the first row, rows)
    embeds: Vec<(usize, ImageEmbed)>,
    /// MoE layers running on the CPU (declared last: dropped after the graphs that reference it)
    cpu: Option<CpuOffload>,
}

/// Optional per-pass hook, called after each layer pass with (slot, engine); use `Engine::debug_buf` to read state.
pub type Capture<'a> = &'a mut dyn FnMut(usize, &Engine);

impl Engine {
    pub fn load(model: &str, n_ctx: usize) -> Result<Self> {
        let g = Gguf::open(model)?;
        let cfg = Config::from_gguf(&g)?;
        // several kernels are specialised for Inkling 2 Mini's dimensions
        ensure!(
            cfg.n_embd == 2304 && cfg.n_ff_exp == 1152 && cfg.n_expert_used == 8 && cfg.n_expert <= 128 && cfg.head_dim == 128
                && cfg.n_head == 24 && cfg.n_head_kv == 8 && cfg.n_rel == 16,
            "unsupported model dimensions"
        );
        ensure!(n_ctx >= 1 && n_ctx <= cfg.context_length, "--ctx {n_ctx} is outside the model's range (1 to {})", cfg.context_length);
        let gpu = Gpu::new(0)?;
        let d = cfg.n_embd;
        let kvw = cfg.n_head_kv * cfg.head_dim;
        let mut slot_phys = vec![0; cfg.n_slots()];
        for p in cfg.schedule() {
            slot_phys[p.slot] = p.phys;
        }
        let (kv_k, kv_v) = KvType::from_env()?;
        // refuse to start when the model does not fit in free VRAM: on Windows the driver would otherwise page the
        // overflow (and other applications' GPU memory) into system RAM until the machine stalls
        let kv_bytes: usize = (0..cfg.n_slots())
            .map(|s| {
                let swa = cfg.is_swa[slot_phys[s]];
                let cap = if swa { (cfg.n_swa + prefill::PREFILL_CHUNK).min(n_ctx) } else { n_ctx };
                if swa { 2 * cap * kvw * 2 } else { cap * cfg.n_head_kv * (kv_k.head_bytes() + kv_v.head_bytes()) }
            })
            .sum();
        let weight_bytes: usize =
            g.tensors.iter().filter(|t| !(weights::embed_on_host() && t.name == "token_embd.weight")).map(|t| t.n_bytes).sum();
        let base_need = weight_bytes + kv_bytes + VRAM_FIXED;
        let free_at_start = gpu.free_vram()?;
        // INKLING_VRAM_LIMIT simulates a smaller card; the planner moves MoE layers to the CPU until the rest fits
        // (and checks that system RAM can hold them)
        let vram_free = offload::effective_free_vram(free_at_start, gpu.total_vram()?)?;
        let plan = offload::plan(&g, &cfg, base_need, vram_free, VRAM_MARGIN, n_ctx)?;
        let need = plan.vram_need;
        check_vram(vram_free, need, &format!("this model at context {n_ctx}"))?;
        let mut w = Weights::load(&gpu, &g, &cfg, &plan.on_cpu)?;
        let cpu = if plan.n_cpu() > 0 {
            let threads = std::env::var("INKLING_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(cpu_moe::default_threads);
            let moe = Box::new(cpu_moe::CpuMoe::new(std::mem::take(&mut w.cpu), d, cfg.n_ff_exp, cfg.situ_beta, cfg.situ_linear_beta, threads));
            let io = gpu.alloc_host_mapped(std::mem::size_of::<cpu_moe::MoeIo>()).context("CPU offload exchange buffer")?;
            let cbs = (0..cfg.n_phys).map(|layer| Box::new(cpu_moe::HostCb { moe: &*moe, io: io.host_ptr() as *mut cpu_moe::MoeIo, layer })).collect();
            let bufs = [gpu.alloc(plan.group_bytes)?, gpu.alloc(plan.group_bytes)?];
            let copy_stream = gpu.ctx.new_stream()?;
            Some(CpuOffload { moe, io, cbs, bufs, copy_stream, plan: plan.clone() })
        } else {
            None
        };
        let mut slots = Vec::with_capacity(cfg.n_slots());
        for s in 0..cfg.n_slots() {
            // sliding-window ring: a prefill chunk writes its K/V before attending, so window + chunk - 1 must fit
            let swa = cfg.is_swa[slot_phys[s]];
            let cap = if swa { (cfg.n_swa + prefill::PREFILL_CHUNK).min(n_ctx) } else { n_ctx };
            // full-attention slots grow with the context and use the chosen format (default q6, KV study: ~q8
            // quality); sliding-window rings stay f16 (fixed size)
            let (fk, fv) = if swa { (KvType::F16, KvType::F16) } else { (kv_k, kv_v) };
            let kv_fmt = fk as i32 | (fv as i32) << 4;
            slots.push(Slot {
                k: gpu.alloc(cap * cfg.n_head_kv * fk.head_bytes())?,
                v: gpu.alloc(cap * cfg.n_head_kv * fv.head_bytes())?,
                kv_cap: cap,
                kv_fmt,
                conv: gpu.alloc(3 * (2 * kvw + 2 * d) * F32)?,
            });
        }
        let f = |n: usize| gpu.alloc(n * F32);
        let qd = cfg.n_head * cfg.head_dim;
        let k = cfg.n_expert_used;
        let act = Act {
            x: f(d)?,
            a: f(3 * d)?,
            c: f(3 * d)?,
            nrm: f(3 * d)?,
            nq: Q8Buf::new(&gpu, 3 * d)?,
            msg: f(4 * d)?,
            g: f(4 * cfg.gate_rank)?,
            qrg: f(2 * qd + cfg.n_head * cfg.n_rel)?,
            sink: f(cfg.n_head + 2 * cfg.n_head_kv * cfg.n_conv)?,
            kraw: f(kvw)?,
            vraw: f(kvw)?,
            att: f(qd)?,
            attq: Q8Buf::new(&gpu, qd)?,
            proj: f(d)?,
            attn_out: f(d)?,
            f: f(d)?,
            f2: f(d)?,
            fnorm: f(d)?,
            fq: Q8Buf::new(&gpu, d)?,
            hq: Q8Buf { q: gpu.alloc(k * cfg.n_ff_exp)?, d: gpu.alloc(k * cfg.n_ff_exp / 16 * F32)? }, // 16-value groups
            sel: f(k)?,
            wts: f(k)?,
            rlog: f(cfg.n_expert)?,
            counter: f(1)?,
            bar: f(2)?,
            attn_part: f(cfg.n_head_kv * ATTN_SPLITS_MAX * 3 * (2 + cfg.head_dim))?,
            attn_cnt: f(cfg.n_head_kv)?,
            part: f(k * d)?,
            ffn_out: f(d)?,
            xp: f(d)?,
            xprev: f(d)?,
            blocks: f(16 * d)?,
            logits: f(cfg.n_vocab)?,
            step: f(4)?,
            prompt: f(n_ctx)?,
            hist: gpu.alloc_host_mapped(n_ctx * 4).context("token history buffer")?,
            sparams: gpu.alloc(std::mem::size_of::<Sampling>())?,
            pen_ring: gpu.alloc(PEN_RING * 4)?,
            mask_on: gpu.alloc(4)?,
            mask: gpu.alloc(cfg.n_vocab.div_ceil(32) * 4)?,
            snap: gpu.alloc(cfg.n_slots() * 3 * (2 * kvw + 2 * d) * F32)?,
        };
        gpu.write(act.mask_on.ptr, inkling_cuda::as_bytes(&[0i32]))?;
        gpu.sync()?;
        let use_graph = std::env::var_os("INKLING_NO_GRAPH").is_none();
        let moe_blocks = if std::env::var("INKLING_MOE").as_deref() == Ok("7") {
            0
        } else {
            // as many blocks as fit, rounded down so the work divides evenly: each block (8 warps) takes whole
            // shares of the 8*F/2 gate_up tasks and D down tasks when 8 * blocks divides gcd(4F, D)
            let fit = gpu.max_blocks_per_sm("ink_moe8", 256)? * gpu.info.sm_count as u32;
            let even = |b: u32| (4 * cfg.n_ff_exp as u32) % (8 * b) == 0 && (cfg.n_embd as u32) % (8 * b) == 0;
            let override_ = std::env::var("INKLING_MOE_BLOCKS").ok().and_then(|v| v.parse().ok());
            override_.unwrap_or_else(|| (fit / 2..=fit).rev().find(|&b| even(b)).unwrap_or(fit))
        };
        ensure!(std::env::var("INKLING_MOE").as_deref() == Ok("7") || moe_blocks > 0, "persistent MoE kernel does not fit on this GPU");
        // decode attention: one full wave of blocks (8 kv heads x splits), so no partial last wave; measured best at
        // every depth (fewer splits also mean a cheaper merge)
        let attn_slots = gpu.max_blocks_per_sm("ink_attn_mma", 128)? * gpu.info.sm_count as u32;
        let attn_splits = std::env::var("INKLING_SPLITS").ok().and_then(|v| v.parse().ok()).unwrap_or(attn_slots / cfg.n_head_kv as u32).clamp(4, ATTN_SPLITS_MAX as u32);
        let pre = Some(prefill::PrefillBufs::new(&gpu, &cfg)?);
        let moe_tl = if std::env::var_os("INKLING_MOE_TL").is_some() { Some(gpu.alloc(cfg.n_slots() * moe_blocks as usize * 64)?) } else { None };
        if std::env::var_os("INKLING_VRAM_DEBUG").is_some() {
            let free_now = gpu.free_vram()?;
            eprintln!("vram: estimated {need} B (fixed part {VRAM_FIXED}), used by load {} B", free_at_start.saturating_sub(free_now));
            if let Some((peak, now)) = offload::working_set() {
                eprintln!("ram: process working set {:.2} GiB (peak during load {:.2} GiB)", now as f64 / (1u64 << 30) as f64, peak as f64 / (1u64 << 30) as f64);
            }
        }
        let pageable: usize = w.host_bufs.iter().filter(|h| !h.pinned()).count();
        if pageable > 0 {
            eprintln!("note: the driver refused to pin part of the offloaded experts; using ordinary memory for {pageable} buffers (prompt processing is slower)");
        }
        if let Some(c) = &cpu {
            let layers: Vec<String> = (0..cfg.n_phys).filter(|&i| c.plan.on_cpu[i]).map(|i| i.to_string()).collect();
            eprintln!(
                "CPU offload: {} of {} MoE layers run on the CPU ({:.1} GiB of experts in system RAM: layers {}; {} threads, {})",
                c.plan.n_cpu(),
                cfg.n_phys,
                c.plan.host_bytes as f64 / (1u64 << 30) as f64,
                layers.join(","),
                c.moe.threads(),
                c.moe.simd()
            );
        }
        Ok(Self { gpu, cfg, w, slots, act, n_ctx, graph: None, graph_pre: None, graph_dec: None, use_graph, moe_blocks, moe_tl, attn_splits, pre, cache: PromptCache::default(), vision: None, embeds: Vec::new(), cpu })
    }

    /// One-line description of the CPU offload (None when everything runs on the GPU).
    pub fn offload_summary(&self) -> Option<String> {
        let c = self.cpu.as_ref()?;
        let layers: Vec<String> = (0..self.cfg.n_phys).filter(|&i| c.plan.on_cpu[i]).map(|i| i.to_string()).collect();
        Some(format!(
            "CPU offload: {} of {} MoE layers in system RAM ({:.1} GiB; layers {}), {} threads, {}",
            c.plan.n_cpu(),
            self.cfg.n_phys,
            c.plan.host_bytes as f64 / (1u64 << 30) as f64,
            layers.join(","),
            c.moe.threads(),
            c.moe.simd()
        ))
    }

    /// CPU expert visits run so far and their mean time (us).
    pub fn cpu_moe_stats(&self) -> Option<(u64, f64)> {
        let c = self.cpu.as_ref()?;
        let n = c.moe.calls.load(std::sync::atomic::Ordering::Relaxed);
        Some((n, c.moe.nanos.load(std::sync::atomic::Ordering::Relaxed) as f64 / n.max(1) as f64 / 1e3))
    }

    fn check_cpu(&self) -> Result<()> {
        if let Some(c) = &self.cpu {
            ensure!(!c.moe.failed.swap(false, std::sync::atomic::Ordering::AcqRel), "the CPU expert executor failed (see the panic message above)");
        }
        Ok(())
    }

    /// Zero all KV caches and conv states (start a new sequence).
    pub fn reset(&mut self) -> Result<()> {
        self.cache = PromptCache::default();
        for s in &self.slots {
            for b in [&s.k, &s.v, &s.conv] {
                unsafe { inkling_cuda::cudarc::driver::result::memset_d8_async(b.ptr, 0, b.bytes, self.gpu.stream.cu_stream()) }?;
            }
        }
        Ok(())
    }

    /// Persistent MoE kernel timeline of `slot`: per block [start, staged, A done, barrier, h staged, end] (ns).
    pub fn moe_timeline(&self, slot: usize) -> Result<Vec<[u64; 6]>> {
        let b = self.moe_tl.as_ref().context("INKLING_MOE_TL not set")?;
        let nb = self.moe_blocks as usize;
        let raw = self.gpu.read_f32(b.at(slot * nb * 64), nb * 16)?;
        let words: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
        Ok((0..nb).map(|i| std::array::from_fn(|k| words[i * 16 + 2 * k] as u64 | (words[i * 16 + 2 * k + 1] as u64) << 32)).collect())
    }

    /// Self-test of the prefill int8 GEMM against the decode matvec: random q8_1 activations for `t` tokens through
    /// layer `layer`'s q|r|gate, k and v projections. Returns (max abs error, max abs value), and the GEMM time (ms)
    /// of `reps` repetitions.
    pub fn test_gemm(&self, layer: usize, t: usize, reps: usize) -> Result<(f32, f32, f64)> {
        let cfg = &self.cfg;
        let (d, kvw, qd) = (cfg.n_embd, cfg.n_head_kv * cfg.head_dim, cfg.n_head * cfg.head_dim);
        let lw = self.w.layers[layer];
        let nq = 2 * qd + cfg.n_head * cfg.n_rel;
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let xq: Vec<i8> = (0..t * d).map(|_| (rnd() % 255) as i64 as i8).map(|v| v.wrapping_sub(127)).collect();
        let xd: Vec<f32> = (0..t * d / 32).map(|_| (rnd() % 1000) as f32 * 1e-5 + 1e-4).collect();
        let bx = self.gpu.upload(inkling_cuda::as_bytes(&xq))?;
        let bxd = self.gpu.upload_f32(&xd)?;
        let outs = [nq, kvw, kvw];
        let yg: Vec<DevBuf> = outs.iter().map(|&n| self.gpu.alloc(t * n * F32)).collect::<Result<_>>()?;
        let ym: Vec<DevBuf> = outs.iter().map(|&n| self.gpu.alloc(t * n * F32)).collect::<Result<_>>()?;
        ensure!(lw.nqrg == 1, "gemmtest needs a Q8_0 model");
        let ws = [lw.wqrg[0], lw.wk, lw.wv];
        let mut a = GemmArgs { nseg: 3, t: t as i32, k: d as i32, ..Default::default() };
        for j in 0..3 {
            a.s[j] = GemmSeg { w: ws[j].qs, wd: ws[j].d, x: bx.ptr, xd: bxd.ptr, y: yg[j].ptr, n: outs[j] as i32, ldy: outs[j] as i32 };
        }
        let tiles: usize = outs.iter().map(|n| n.div_ceil(64)).sum();
        let launch = || self.gpu.launch("ink_gemm_q8", (tiles as u32, t.div_ceil(64) as u32), 128, &[Arg::Gemm(a)]);
        launch()?;
        for tok in 0..t {
            let segs: Vec<_> = (0..3).map(|j| (ws[j], (bx.at(tok * d), bxd.at(tok * d / 32 * F32)), ym[j].at(tok * outs[j] * F32), outs[j])).collect();
            self.mv(&segs, d)?;
        }
        self.gpu.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            launch()?;
        }
        self.gpu.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
        let (mut err, mut mag) = (0.0f32, 0.0f32);
        for j in 0..3 {
            let g = self.gpu.read_f32(yg[j].ptr, t * outs[j])?;
            let m = self.gpu.read_f32(ym[j].ptr, t * outs[j])?;
            for (x, y) in g.iter().zip(&m) {
                err = err.max((x - y).abs());
                mag = mag.max(y.abs());
            }
        }
        Ok((err, mag, ms))
    }

    /// CPU expert executor microbenchmark of offloaded layer `li` (see CpuMoe::bench).
    pub fn bench_cpu_moe(&self, li: usize, iters: usize) -> Result<(f64, f64, f64)> {
        let c = self.cpu.as_ref().context("no layer is offloaded")?;
        ensure!(c.plan.on_cpu[li], "layer {li} is not offloaded");
        Ok(c.moe.bench(li, iters))
    }

    /// Self-test of the CPU expert executor against the GPU kernel (ink_moe8 reading the same pinned weights over
    /// PCIe) on random q8_1 inputs and random routing, for every offloaded layer. Returns per layer
    /// (layer, relative RMS error, max |y|).
    pub fn test_cpu_moe(&self, trials: usize) -> Result<Vec<(usize, f32, f32)>> {
        let c = self.cpu.as_ref().context("no layer is offloaded (use --offload N)")?;
        ensure!(self.w.host_bufs.iter().all(|h| h.pinned()), "cpumoetest needs pinned (GPU-mapped) expert memory");
        let (d, f, k) = (self.cfg.n_embd, self.cfg.n_ff_exp, self.cfg.n_expert_used);
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::new();
        for li in (0..self.cfg.n_phys).filter(|&i| c.plan.on_cpu[i]) {
            let lw = self.w.layers[li];
            let (mut num, mut den, mut mag) = (0f64, 0f64, 0f32);
            for _ in 0..trials {
                let mut io: Box<cpu_moe::MoeIo> = Box::new(unsafe { std::mem::zeroed() });
                // activations with a realistic spread: gaussian-ish values, a few large channels
                let x: Vec<f32> = (0..d).map(|i| ((rnd() + rnd() + rnd() - 1.5) * if i % 97 == 0 { 8.0 } else { 1.0 }) as f32).collect();
                for b in 0..d / 32 {
                    let amax = x[32 * b..32 * b + 32].iter().fold(0f32, |m, v| m.max(v.abs()));
                    let s = amax / 127.0;
                    io.xd[b] = s;
                    for i in 0..32 {
                        io.xq[32 * b + i] = if s == 0.0 { 0 } else { (x[32 * b + i] / s).round() as i8 };
                    }
                }
                let mut sel: Vec<i32> = Vec::new();
                while sel.len() < k {
                    let e = (rnd() * self.cfg.n_expert as f64) as i32;
                    if !sel.contains(&e) {
                        sel.push(e);
                    }
                }
                io.sel.copy_from_slice(&sel);
                for t in 0..k {
                    io.wts[t] = (0.05 + rnd() * 0.3) as f32;
                }
                // GPU: ink_moe8 with zero conv weights / state / residual -> x = y / sqrt(5)
                let xq = self.gpu.upload(inkling_cuda::as_bytes(&io.xq))?;
                let xd = self.gpu.upload_f32(&io.xd)?;
                let selb = self.gpu.upload(inkling_cuda::as_bytes(&io.sel))?;
                let wts = self.gpu.upload_f32(&io.wts)?;
                let h = self.gpu.alloc(k * f * F32)?;
                let cw = self.gpu.alloc(4 * d * F32)?;
                let st = self.gpu.alloc(3 * d * F32)?;
                let xo = self.gpu.alloc(d * F32)?;
                let bar = self.gpu.alloc(2 * F32)?;
                let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt.code(), erange: 0 };
                if lw.gate_up.fmt.tuned() && lw.down.fmt.tuned() {
                    self.gpu.launch("ink_moe8", (self.moe_blocks, 1), 256, &[
                        Arg::Exp(ew(lw.gate_up)), Arg::Exp(ew(lw.down)), Arg::Ptr(xq.ptr), Arg::Ptr(xd.ptr), Arg::Ptr(selb.ptr), Arg::Ptr(wts.ptr),
                        Arg::Ptr(h.ptr), Arg::Ptr(cw.ptr), Arg::Ptr(st.ptr), Arg::Ptr(xo.ptr), Arg::Ptr(0), Arg::Ptr(bar.ptr),
                        Arg::F32(self.cfg.situ_beta), Arg::F32(self.cfg.situ_linear_beta), Arg::Ptr(0),
                    ])?;
                } else {
                    // the decode path for layers with a natively stored tensor: two kernels via hq / hd
                    let hq = self.gpu.alloc(k * f)?;
                    let hd = self.gpu.alloc(k * f / 16 * F32)?;
                    let gu = if lw.gate_up.fmt.tuned() { "ink_moe_gate_up7" } else { "ink_moe_gate_up_gen" };
                    let dn = if lw.down.fmt.tuned() { "ink_moe_down7" } else { "ink_moe_down_gen" };
                    self.gpu.launch(gu, ((f / 16) as u32, k as u32), 256, &[
                        Arg::Exp(ew(lw.gate_up)), Arg::Ptr(xq.ptr), Arg::Ptr(xd.ptr), Arg::Ptr(selb.ptr), Arg::Ptr(hq.ptr), Arg::Ptr(hd.ptr),
                        Arg::F32(self.cfg.situ_beta), Arg::F32(self.cfg.situ_linear_beta),
                    ])?;
                    self.gpu.launch(dn, ((d / 8) as u32, 1), 256, &[
                        Arg::Exp(ew(lw.down)), Arg::Ptr(hq.ptr), Arg::Ptr(hd.ptr), Arg::Ptr(selb.ptr), Arg::Ptr(wts.ptr),
                        Arg::Ptr(cw.ptr), Arg::Ptr(st.ptr), Arg::Ptr(xo.ptr), Arg::Ptr(0),
                    ])?;
                }
                let yg: Vec<f32> = self.gpu.read_f32(xo.ptr, d)?.iter().map(|v| v * 5f32.sqrt()).collect();
                c.moe.run(li, &mut io);
                for (a, b) in io.y.iter().zip(&yg) {
                    num += ((a - b) as f64).powi(2);
                    den += (*b as f64).powi(2);
                    mag = mag.max(b.abs());
                }
            }
            out.push((li, (num / den.max(1e-30)).sqrt() as f32, mag));
        }
        Ok(out)
    }

    /// KV format study (INKLING_KQ / INKLING_VQ = bits, INKLING_KV_ALL = also sliding-window layers): bits the
    /// cache write fake-quantizes K and V to (0 = f16).
    pub(crate) fn kv_bits(&self, is_swa: bool) -> (i32, i32) {
        if is_swa && std::env::var_os("INKLING_KV_ALL").is_none() {
            return (0, 0);
        }
        let b = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        (b("INKLING_KQ"), b("INKLING_VQ"))
    }

    /// Read a scratch buffer by name (debugging / validation).
    pub fn debug_buf(&self, name: &str) -> Result<Vec<f32>> {
        let a = &self.act;
        let c = &self.cfg;
        let qd = c.n_head * c.head_dim;
        let (ptr, n) = match name {
            "x" => (a.x.ptr, c.n_embd),
            "q" => (a.qrg.ptr, qd),
            "r" => (a.qrg.at(qd * F32), c.n_head * c.n_rel),
            "nrm" => (a.nrm.ptr, 3 * c.n_embd),
            "att" => (a.att.ptr, qd),
            "attn_out" => (a.attn_out.ptr, c.n_embd),
            "fnorm" => (a.fnorm.ptr, c.n_embd),
            "rlog" => (a.rlog.ptr, c.n_expert),
            "wts" => (a.wts.ptr, c.n_expert_used),
            "ffn_out" => (a.ffn_out.ptr, c.n_embd),
            "msg" => (a.msg.ptr, 4 * c.n_embd),
            _ => anyhow::bail!("unknown debug buffer {name}"),
        };
        self.gpu.read_f32(ptr, n)
    }

    // ---- launch helpers ------------------------------------------------------------------------------------

    /// Depth attention over `states` with `nq` query rows. With `outn`, also writes rmsnorm(out_q) * nw[q] and,
    /// with `xq`, its q8_1 copy (values, scales) for each query row.
    fn mix(&self, states: &[u64], q: u64, nq: usize, out: u64, outn: u64, nw: [u64; 4], xq: (u64, u64)) -> Result<()> {
        ensure!(states.len() <= 6 && nq <= 4, "mix: too many states/queries");
        let mut m = MixArgs { q, out, outn, nw, n: states.len() as i32, nq: nq as i32, d: self.cfg.n_embd as i32, eps: self.cfg.rms_eps, ..Default::default() };
        m.st[..states.len()].copy_from_slice(states);
        let k = match nq { 1 => "ink_mix6_1", 3 => "ink_mix6_3", 4 => "ink_mix6_4", _ => anyhow::bail!("mix: nq = {nq} unsupported") };
        self.gpu.launch(k, (1, 1), 768, &[Arg::MixQ(MixArgsQ { m, xq: xq.0, xd: xq.1 })])
    }

    /// RMS-normalize up to 4 vectors (x_j * w_j) into y_j and their q8_1 copies.
    fn norm_q(&self, items: &[(u64, u64, u64, (u64, u64))]) -> Result<()> {
        let mut a = NormQ4::default();
        for (j, &(x, w, y, (q, d))) in items.iter().enumerate() {
            a.x[j] = x;
            a.w[j] = w;
            a.y[j] = y;
            a.xq[j] = q;
            a.xd[j] = d;
        }
        self.gpu.launch("ink_rmsnorm_q2", (items.len() as u32, 1), 768, &[Arg::NormQ(a), Arg::I32(self.cfg.n_embd as i32), Arg::F32(self.cfg.rms_eps)])
    }

    /// The q | rel | gate projection as matvec segments writing the contiguous qrg row at `y`.
    fn qrg_segs(&self, lw: &LayerW, x: (u64, u64), y: u64) -> Vec<(weights::Q8M, (u64, u64), u64, usize)> {
        let (qd, nr) = (self.cfg.n_head * self.cfg.head_dim, self.cfg.n_head * self.cfg.n_rel);
        if lw.nqrg == 1 {
            vec![(lw.wqrg[0], x, y, 2 * qd + nr)]
        } else {
            vec![(lw.wqrg[0], x, y, qd), (lw.wqrg[1], x, y + (qd * F32) as u64, nr), (lw.wqrg[2], x, y + ((qd + nr) * F32) as u64, qd)]
        }
    }

    /// Matvecs on q8_1 activations: segments (weights, (xq, xd), y, rows), all with K inputs. All-Q8_0 calls use the
    /// tuned kernel, others the generic one.
    fn mv(&self, segs: &[(weights::Q8M, (u64, u64), u64, usize)], k: usize) -> Result<()> {
        if segs.len() > 4 || segs.iter().any(|s| s.0.fmt != 0) || std::env::var_os("INKLING_GEN_DENSE").is_some() {
            ensure!(segs.len() <= 8, "mv: too many segments");
            let mut a = inkling_cuda::MvGen { nseg: segs.len() as i32, k: k as i32, ..Default::default() };
            let mut rows = 0;
            for (i, &(w, (xq, xd), y, n)) in segs.iter().enumerate() {
                a.s[i] = inkling_cuda::MvSegG { w: w.qs, wd: w.d, xq, xd, y, n: n as i32, fmt: w.fmt };
                rows += n;
            }
            return self.gpu.launch("ink_mv_gen", (rows.div_ceil(8) as u32, 1), 256, &[Arg::MvG(a)]);
        }
        let mut a = MvQ4 { nseg: segs.len() as i32, k: k as i32, ..Default::default() };
        let mut rows = 0;
        for (i, &(w, (xq, xd), y, n)) in segs.iter().enumerate() {
            a.s[i] = MvSegQ { qs: w.qs, d: w.d, xq, xd, y, n: n as i32 };
            rows += n;
        }
        self.gpu.launch("ink_mv_q8q", (rows.div_ceil(8) as u32, 1), 256, &[Arg::MvQ(a)])
    }

    fn mv_bf16(&self, segs: &[(u64, u64, u64, usize)], k: usize) -> Result<()> {
        let mut a = MvB4 { nseg: segs.len() as i32, k: k as i32, ..Default::default() };
        let mut rows = 0;
        for (i, &(w, x, y, n)) in segs.iter().enumerate() {
            a.s[i] = MvSegB { w, x, y, n: n as i32 };
            rows += n;
        }
        self.gpu.launch("ink_mv_bf16", (rows.div_ceil(8) as u32, 1), 256, &[Arg::MvB(a)])
    }

    /// Post conv over the sum of `nparts` inputs (D apart), written to x (accumulate) or replacing x.
    fn conv_res(&self, xin: u64, nparts: usize, w: u64, state: u64, x: u64, dbg: u64, accumulate: bool) -> Result<()> {
        let d = self.cfg.n_embd;
        self.gpu.launch("ink_conv_res", (d.div_ceil(256) as u32, 1), 256, &[
            Arg::Ptr(xin), Arg::I32(nparts as i32), Arg::I32(d as i32), Arg::Ptr(w), Arg::Ptr(state), Arg::Ptr(x), Arg::Ptr(dbg),
            Arg::I32(accumulate as i32), Arg::I32(d as i32),
        ])
    }

    fn copy(&self, src: u64, dst: u64) -> Result<()> {
        let d = self.cfg.n_embd;
        self.gpu.launch("ink_copy", (d.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(src), Arg::Ptr(dst), Arg::I32(d as i32)])
    }

    /// Loop controls: out_j = control(c_j, m_j) using control index idx_j of the layer.
    fn controls(&self, lp: &weights::LoopW, ctrls: &[(usize, u64, u64, u64)]) -> Result<()> {
        let mut a = Ctrl4 { n: ctrls.len() as i32, ..Default::default() };
        let r = self.cfg.gate_rank;
        for (j, &(idx, c, m, out)) in ctrls.iter().enumerate() {
            a.c[j] = c;
            a.m[j] = m;
            a.wd[j] = lp.down[idx];
            a.wg[j] = lp.gain[idx];
            a.wf[j] = lp.forget[idx];
            a.g[j] = self.act.g.at(j * r * F32);
            a.out[j] = out;
        }
        let (d, n) = (self.cfg.n_embd as i32, ctrls.len() as u32);
        // Ctrl4 and Ctrl4H share one layout (pointer arrays); the kernel picks the weight type
        if lp.f16 {
            self.gpu.launch("ink_ctrl_down2", (r as u32, n), 192, &[Arg::Ctrl(a), Arg::I32(d), Arg::I32(r as i32)])?;
        } else {
            self.gpu.launch("ink_ctrl_down", ((r / 8) as u32, n), 256, &[Arg::Ctrl(a), Arg::I32(d), Arg::I32(r as i32)])?;
        }
        let ku = if lp.f16 { "ink_ctrl_up_h" } else { "ink_ctrl_up" };
        self.gpu.launch(ku, ((self.cfg.n_embd / 8) as u32, n), 256, &[Arg::Ctrl(a), Arg::I32(d), Arg::I32(r as i32)])
    }

    // ---- the token step ------------------------------------------------------------------------------------

    /// Enqueue every kernel of one decode step (no host synchronization inside: graph-capturable).
    fn enqueue_step(&self, mut capture: Option<Capture>) -> Result<()> {
        let cfg = &self.cfg;
        let d = cfg.n_embd;
        let a = &self.act;
        let kvw = cfg.n_head_kv * cfg.head_dim;
        let qd = cfg.n_head * cfg.head_dim;
        let (q_ptr, r_ptr, gate_ptr) = (a.qrg.ptr, a.qrg.at(qd * F32), a.qrg.at((qd + cfg.n_head * cfg.n_rel) * F32));
        let nkc = cfg.n_head_kv * cfg.n_conv;
        let (sink_ptr, dynk_ptr, dynv_ptr) = (a.sink.ptr, a.sink.at(cfg.n_head * F32), a.sink.at((cfg.n_head + nkc) * F32));
        let row = |j: usize| j * d * F32;
        let nqv = |j: usize| a.nq.at(j, d);

        if self.w.token_embd_fmt == 8 {
            self.gpu.launch("ink_embed", (1, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::Ptr(a.step.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(a.x.ptr), Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
            ])?;
        } else {
            self.gpu.launch("ink_embed_gen", (1, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::I32(self.w.token_embd_fmt), Arg::Ptr(a.step.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(a.x.ptr),
                Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
            ])?;
        }

        let mut blocks: Vec<u64> = Vec::new();
        let mut prelude_blocks: Vec<u64> = Vec::new();
        let mut bank: Vec<u64> = Vec::new();
        let mut n_push = 0usize;

        for p in cfg.schedule() {
            let lw = self.w.layers[p.phys];
            if p.phys == cfg.core_start {
                if p.pass == 0 {
                    self.copy(a.x.ptr, a.xp.ptr)?;
                    prelude_blocks = blocks.clone();
                } else {
                    bank = blocks[prelude_blocks.len() + 1..].to_vec();
                    self.copy(a.x.ptr, a.xprev.ptr)?;
                    self.copy(a.xp.ptr, a.x.ptr)?;
                    blocks = prelude_blocks.clone();
                }
            }
            let slot = &self.slots[p.slot];
            let (conv_k, conv_v) = (slot.conv.ptr, slot.conv.at(3 * kvw * F32));
            let (conv_attn, conv_ffn) = (slot.conv.at(6 * kvw * F32), slot.conv.at((6 * kvw + 3 * d) * F32));
            let looped = p.pass > 0;
            let lp = lw.lp;

            if looped {
                let mut st = bank.clone();
                st.push(a.xprev.ptr);
                self.mix(&st, lp.unwrap().msg_query, 4, a.msg.ptr, 0, [0; 4], (0, 0))?;
            }

            let boundary = p.phys % cfg.attn_res_block == 0;
            let inp_copy = if boundary {
                let dst = a.blocks.at(n_push * d * F32);
                n_push += 1;
                self.copy(a.x.ptr, dst)?;
                Some(dst)
            } else {
                None
            };

            // attention inputs -> normalized float (nrm[j]) + q8_1 (nq[j])
            let n = [a.nrm.ptr, a.nrm.at(row(1)), a.nrm.at(row(2))];
            let norms = if lw.is_swa { [lw.attn_norm; 3] } else { [lw.q_norm, lw.k_norm, lw.v_norm] };
            let nqry = if lw.is_swa { 1 } else { 3 };
            let shared_input = lw.is_swa && !looped; // q, k and v read the same normalized vector
            if !looped {
                if lw.attn_res_query != 0 {
                    let mut st = blocks.clone();
                    st.push(a.x.ptr);
                    let nw = if lw.is_swa { [norms[0], 0, 0, 0] } else { [norms[0], norms[1], norms[2], 0] };
                    self.mix(&st, lw.attn_res_query, nqry, a.a.ptr, a.nrm.ptr, nw, nqv(0))?;
                } else {
                    let items: Vec<_> = (0..nqry).map(|j| (a.x.ptr, norms[j], n[j], nqv(j))).collect();
                    self.norm_q(&items)?;
                }
            } else {
                let mut st = blocks.clone();
                st.push(a.x.ptr);
                self.mix(&st, lw.attn_res_query, nqry, a.a.ptr, 0, [0; 4], (0, 0))?;
                let inp = |j: usize| if lw.is_swa { a.a.ptr } else { a.a.at(row(j)) };
                let ctl: Vec<_> = (0..3).map(|j| (j, inp(j), a.msg.at(row(j)), a.c.at(row(j)))).collect();
                self.controls(&lp.unwrap(), &ctl)?;
                let items: Vec<_> = (0..3).map(|j| (a.c.at(row(j)), norms[j], n[j], nqv(j))).collect();
                self.norm_q(&items)?;
            }
            let (iq, ik, iv) = if shared_input { (0, 0, 0) } else { (0, 1, 2) };

            // projections
            let mut segs = self.qrg_segs(&lw, nqv(iq), a.qrg.ptr);
            segs.push((lw.wk, nqv(ik), a.kraw.ptr, kvw));
            segs.push((lw.wv, nqv(iv), a.vraw.ptr, kvw));
            self.mv(&segs, d)?;
            let (dk, dv) = if lw.is_swa {
                self.mv_bf16(&[(lw.wsink, n[iq], sink_ptr, cfg.n_head)], d)?;
                (0, 0)
            } else {
                self.mv_bf16(&[(lw.wsink, n[iq], sink_ptr, cfg.n_head), (lw.kconv_proj, n[ik], dynk_ptr, nkc), (lw.vconv_proj, n[iv], dynv_ptr, nkc)], d)?;
                (dynk_ptr, dynv_ptr)
            };
            // K/V short convs + clip/norm + cache write, q/r clip + tau
            self.gpu.launch("ink_attn_prep2", (64, 1), 128, &[
                Arg::Ptr(q_ptr), Arg::Ptr(a.kraw.ptr), Arg::Ptr(a.vraw.ptr), Arg::Ptr(r_ptr), Arg::Ptr(lw.q_clip), Arg::Ptr(lw.rel_clip),
                Arg::Ptr(lw.k_conv), Arg::Ptr(lw.v_conv), Arg::Ptr(conv_k), Arg::Ptr(conv_v), Arg::Ptr(dk), Arg::Ptr(dv),
                Arg::Ptr(a.step.ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(!lw.is_swa as i32),
                Arg::F32(cfg.qk_clip), Arg::F32(cfg.rel_clip), Arg::F32(cfg.log_scale_floor), Arg::F32(cfg.log_scale_alpha), Arg::F32(cfg.rms_eps),
                Arg::I32(self.kv_bits(lw.is_swa).0), Arg::I32(self.kv_bits(lw.is_swa).1), Arg::I32(slot.kv_fmt),
            ])?;
            let window = if lw.is_swa { cfg.n_swa as i32 } else { 0 };
            let (oq, od) = a.attq.at(0, qd);
            // tensor-core decode attention (INKLING_ATTN=split: the previous CUDA-core kernel, for A/B checks)
            let attn_kernel = if std::env::var("INKLING_ATTN").as_deref() == Ok("split") { "ink_attn_split" } else { "ink_attn_mma" };
            self.gpu.launch(attn_kernel, (cfg.n_head_kv as u32, self.attn_splits), 128, &[
                Arg::Ptr(q_ptr), Arg::Ptr(r_ptr), Arg::Ptr(sink_ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::Ptr(lw.rel_bias),
                Arg::Ptr(gate_ptr), Arg::Ptr(a.att.ptr), Arg::Ptr(oq), Arg::Ptr(od),
                Arg::Ptr(a.step.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(window), Arg::I32(cfg.rel_extent as i32),
                Arg::Ptr(a.attn_part.ptr), Arg::Ptr(a.attn_cnt.ptr), Arg::I32((2 * qd + cfg.n_head * cfg.n_rel) as i32), Arg::I32(0), Arg::I32(slot.kv_fmt),
            ])?;
            self.mv(&[(lw.wo, (oq, od), a.proj.ptr, d)], qd)?;

            // post conv fused with the residual: block boundary starts a new stream (x = attn), else x += attn
            if let Some(dst) = inp_copy {
                blocks.push(dst);
            }
            self.conv_res(a.proj.ptr, 1, lw.attn_post_conv, conv_attn, a.x.ptr, a.attn_out.ptr, inp_copy.is_none())?;

            // FFN input -> fnorm (+ q8_1)
            let mut st = blocks.clone();
            st.push(a.x.ptr);
            let fq = a.fq.at(0, d);
            if looped {
                self.mix(&st, lw.ffn_res_query, 1, a.f.ptr, 0, [0; 4], (0, 0))?;
                self.controls(&lp.unwrap(), &[(3, a.f.ptr, a.msg.at(row(3)), a.f2.ptr)])?;
                self.norm_q(&[(a.f2.ptr, lw.ffn_norm, a.fnorm.ptr, fq)])?;
            } else {
                self.mix(&st, lw.ffn_res_query, 1, a.f.ptr, a.fnorm.ptr, [lw.ffn_norm, 0, 0, 0], fq)?;
            }

            // MoE: router + top-k, experts on int8 activations, weighted partials summed by the post conv
            let k = cfg.n_expert_used;
            self.gpu.launch("ink_router4", (cfg.n_expert as u32, 1), 192, &[
                Arg::Ptr(a.fnorm.ptr), Arg::Ptr(lw.router), Arg::Ptr(lw.probs_bias), Arg::Ptr(a.rlog.ptr), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr),
                Arg::Ptr(a.counter.ptr), Arg::I32(d as i32), Arg::I32(cfg.n_expert as i32), Arg::I32(k as i32), Arg::F32(cfg.expert_weights_scale),
            ])?;
            let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt.code(), erange: 0 };
            if let Some(c) = self.cpu.as_ref().filter(|c| c.plan.on_cpu[p.phys]) {
                // experts on the CPU: export the FFN input and routing to mapped memory, run them in a host node, then
                // the post conv + residual reads the weighted expert sum back over PCIe
                self.gpu.launch("ink_moe_export", (1, 1), 256, &[
                    Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr), Arg::Ptr(c.io.dev), Arg::I32(d as i32),
                ])?;
                let cb: *const cpu_moe::HostCb = &*c.cbs[p.phys];
                self.gpu.launch_host(cpu_moe::host_cb, cb as *mut std::ffi::c_void)?;
                let y = c.io.dev + std::mem::offset_of!(cpu_moe::MoeIo, y) as u64;
                self.conv_res(y, 1, lw.ffn_post_conv, conv_ffn, a.x.ptr, a.ffn_out.ptr, true)?;
            } else if self.moe_blocks > 0 && lw.gate_up.fmt.tuned() && lw.down.fmt.tuned() {
                // gate_up + down + expert sum + post conv + residual, one persistent kernel
                self.gpu.launch("ink_moe8", (self.moe_blocks, 1), 256, &[
                    Arg::Exp(ew(lw.gate_up)), Arg::Exp(ew(lw.down)), Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr),
                    Arg::Ptr(a.part.ptr), Arg::Ptr(lw.ffn_post_conv), Arg::Ptr(conv_ffn), Arg::Ptr(a.x.ptr), Arg::Ptr(a.ffn_out.ptr),
                    Arg::Ptr(a.bar.ptr), Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
                    Arg::Ptr(self.moe_tl.as_ref().map_or(0, |b| b.at(p.slot * self.moe_blocks as usize * 64))),
                ])?;
            } else {
                // two kernels; each tensor runs on the tuned v7 kernel or the generic one for its format
                let (hq, hd) = (a.hq.q.ptr, a.hq.d.ptr);
                let gu_kernel = if lw.gate_up.fmt.tuned() { "ink_moe_gate_up7" } else { "ink_moe_gate_up_gen" };
                let dn_kernel = if lw.down.fmt.tuned() { "ink_moe_down7" } else { "ink_moe_down_gen" };
                self.gpu.launch(gu_kernel, ((cfg.n_ff_exp / 16) as u32, k as u32), 256, &[
                    Arg::Exp(ew(lw.gate_up)), Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(a.sel.ptr), Arg::Ptr(hq), Arg::Ptr(hd),
                    Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
                ])?;
                // down + weighted sum over the k experts + post conv + residual
                self.gpu.launch(dn_kernel, ((d / 8) as u32, 1), 256, &[
                    Arg::Exp(ew(lw.down)), Arg::Ptr(hq), Arg::Ptr(hd), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr),
                    Arg::Ptr(lw.ffn_post_conv), Arg::Ptr(conv_ffn), Arg::Ptr(a.x.ptr), Arg::Ptr(a.ffn_out.ptr),
                ])?;
            }

            if let Some(cb) = capture.as_mut() {
                cb(p.slot, self);
            }
        }

        // output head
        let mut st = blocks.clone();
        st.push(a.x.ptr);
        let fq = a.fq.at(0, d);
        self.mix(&st, self.w.output_res_query, 1, a.f.ptr, a.fnorm.ptr, [self.w.output_norm, 0, 0, 0], fq)?;
        self.mv(&[(self.w.output, fq, a.logits.ptr, cfg.n_vocab)], d)?;
        let nv = cfg.n_vocab_unpadded;
        self.gpu.launch("ink_softcap", (nv.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(a.logits.ptr), Arg::I32(nv as i32), Arg::F32(cfg.final_softcap)])
    }

    fn enqueue_tail(&self, tail: Tail) -> Result<()> {
        match tail {
            Tail::None => Ok(()),
            Tail::Prefill => self.gpu.launch("ink_advance_prompt", (1, 1), 32, &[Arg::Ptr(self.act.step.ptr), Arg::Ptr(self.act.prompt.ptr)]),
            Tail::Greedy => {
                let nv = self.cfg.n_vocab_unpadded as i32;
                self.gpu.launch("ink_apply_mask", (((nv + 1023) / 1024) as u32, 1), 1024, &[
                    Arg::Ptr(self.act.logits.ptr), Arg::I32(nv), Arg::Ptr(self.act.mask_on.ptr), Arg::Ptr(self.act.mask.ptr),
                ])?;
                self.gpu.launch("ink_sample", (1, 1), 1024, &[
                Arg::Ptr(self.act.logits.ptr), Arg::I32(self.cfg.n_vocab_unpadded as i32), Arg::Ptr(self.act.sparams.ptr),
                Arg::Ptr(self.act.step.ptr), Arg::Ptr(self.act.hist.dev), Arg::Ptr(self.act.pen_ring.ptr),
            ])
            }
        }
    }

    fn graph_slot(&mut self, tail: Tail) -> &mut Option<CudaGraph> {
        match tail {
            Tail::None => &mut self.graph,
            Tail::Prefill => &mut self.graph_pre,
            Tail::Greedy => &mut self.graph_dec,
        }
    }

    /// Capture (once) and launch the graph for `tail`.
    fn launch_graph(&mut self, tail: Tail) -> Result<()> {
        if self.graph_slot(tail).is_none() {
            self.gpu.sync()?;
            self.gpu.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let r = self.enqueue_step(None).and_then(|_| self.enqueue_tail(tail));
            let g = self.gpu.stream.end_capture(sys::CUgraphInstantiate_flags(0))?;
            r?;
            *self.graph_slot(tail) = g;
        }
        self.graph_slot(tail).as_ref().expect("graph").launch()?;
        Ok(())
    }

    /// Copy every slot's conv state to (`save`) or from the prompt-cache snapshot.
    fn snapshot(&self, save: bool) -> Result<()> {
        let per = self.slots[0].conv.bytes;
        for (i, s) in self.slots.iter().enumerate() {
            let (src, dst) = if save { (s.conv.ptr, self.act.snap.at(i * per)) } else { (self.act.snap.at(i * per), s.conv.ptr) };
            unsafe { inkling_cuda::cudarc::driver::result::memcpy_dtod_async(dst, src, per, self.gpu.stream.cu_stream()) }?;
        }
        Ok(())
    }

    /// Positions that may be processed after a snapshot before the sliding-window rings overwrite entries the
    /// snapshot point still needs.
    fn snap_margin(&self) -> usize {
        self.slots.iter().filter(|s| s.kv_cap < self.n_ctx && s.kv_cap > self.cfg.n_swa).map(|s| s.kv_cap - self.cfg.n_swa).min().unwrap_or(usize::MAX)
    }

    /// Bring the device state to "after prompt[..n-1]", reusing what the previous call left behind when the new
    /// prompt extends it (prompt caching). Returns how many prompt tokens were reused.
    fn prepare_prompt(&mut self, prompt: &[u32]) -> Result<usize> {
        let n = prompt.len();
        let head = &prompt[..n - 1];
        let c = &self.cache;
        let start = if !c.tokens.is_empty() && c.tokens.len() <= head.len() && head.starts_with(&c.tokens) {
            c.tokens.len() // the state is exactly after a prefix of the new prompt
        } else if let Some(st) = c.snap_tokens.as_ref().filter(|st| {
            !st.is_empty() && st.len() <= head.len() && head.starts_with(st) && c.tokens.len() - st.len() <= self.snap_margin()
        }) {
            let k = st.len();
            self.snapshot(false)?; // conv states back to the snapshot; KV beyond position k is simply overwritten
            k
        } else {
            self.reset()?;
            0
        };
        if std::env::var_os("INKLING_NO_PREFILL").is_some() {
            ensure!(start == 0, "INKLING_NO_PREFILL with a cached prompt");
            return Ok(0);
        }
        self.prefill(&head[start..], start)?;
        self.snapshot(true)?;
        self.cache = PromptCache { tokens: head.to_vec(), snap_tokens: Some(head.to_vec()) };
        Ok(start)
    }

    /// Generation with the autoregressive loop on the device: the prompt is prefilled in batches (reusing the
    /// state left by the previous call when the new prompt extends it), every decode step samples its token on
    /// the device as the next input, and the host queues steps ahead, reading tokens from mapped memory behind
    /// per-step events. `on_token` returns false to stop. Returns the sampled tokens.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, sampling: &Sampling, mut on_token: impl FnMut(u32) -> bool) -> Result<Vec<u32>> {
        self.generate_ex(prompt, max_new, sampling, &mut on_token).map(|(out, _)| out)
    }

    /// As `generate`, also returning how many prompt tokens came from the cache.
    pub fn generate_ex(&mut self, prompt: &[u32], max_new: usize, sampling: &Sampling, on_token: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, usize)> {
        self.generate_mm(prompt, &[], max_new, sampling, on_token)
    }

    /// As `generate_ex` with image embeddings: `embeds` are (position, embedding) pairs whose rows replace the
    /// prompt positions pos..pos + n (those prompt ids must be >= EMBED_TOKEN_BASE and should identify the image,
    /// e.g. EMBED_TOKEN_BASE | hash bits, so the prompt cache does not confuse different images).
    pub fn generate_mm(&mut self, prompt: &[u32], embeds: &[(usize, ImageEmbed)], max_new: usize, sampling: &Sampling, on_token: &mut dyn FnMut(u32) -> bool) -> Result<(Vec<u32>, usize)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(*prompt.last().unwrap() < EMBED_TOKEN_BASE, "the prompt must not end inside an image");
        for (pos, e) in embeds {
            ensure!(e.data.len() == e.n * self.cfg.n_embd, "image embedding width");
            ensure!(pos + e.n <= prompt.len() && prompt[*pos..pos + e.n].iter().all(|&t| t >= EMBED_TOKEN_BASE), "image embedding positions do not match the prompt");
        }
        ensure!(prompt.iter().filter(|&&t| t >= EMBED_TOKEN_BASE).count() == embeds.iter().map(|(_, e)| e.n).sum::<usize>(), "prompt has image positions without embeddings");
        self.embeds = embeds.to_vec();
        let r = self.generate_inner(prompt, max_new, sampling, on_token, None);
        self.embeds.clear();
        r
    }

    /// As `generate_mm`, with a token constraint: before every step `mask` gives the allowed tokens (one bit per
    /// token over the unpadded vocabulary, 32 per word) or None for no constraint on that step. Steps run one at a
    /// time (no lookahead), so `on_token` sees each token before the next mask is asked for.
    pub fn generate_mm_constrained(&mut self, prompt: &[u32], embeds: &[(usize, ImageEmbed)], max_new: usize, sampling: &Sampling,
                                   on_token: &mut dyn FnMut(u32) -> bool, mask: &mut dyn FnMut() -> Option<Vec<u32>>) -> Result<(Vec<u32>, usize)> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(*prompt.last().unwrap() < EMBED_TOKEN_BASE, "the prompt must not end inside an image");
        ensure!(prompt.iter().filter(|&&t| t >= EMBED_TOKEN_BASE).count() == embeds.iter().map(|(_, e)| e.n).sum::<usize>(), "prompt has image positions without embeddings");
        self.embeds = embeds.to_vec();
        let r = self.generate_inner(prompt, max_new, sampling, on_token, Some(mask));
        self.embeds.clear();
        r
    }

    fn generate_inner(&mut self, prompt: &[u32], max_new: usize, sampling: &Sampling, on_token: &mut dyn FnMut(u32) -> bool,
                      mut mask: Option<&mut dyn FnMut() -> Option<Vec<u32>>>) -> Result<(Vec<u32>, usize)> {
        ensure!(prompt.len() + max_new <= self.n_ctx, "prompt + generation exceed the context ({})", self.n_ctx);
        ensure!(self.use_graph, "generate needs CUDA graphs");
        ensure!(max_new > 0, "max_new must be positive");
        self.gpu.write(self.act.sparams.ptr, inkling_cuda::as_bytes(std::slice::from_ref(sampling)))?;
        let n = prompt.len();
        let p: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        if sampling.last_n > 0 {
            let mut ring = vec![-1i32; PEN_RING];
            for i in n.saturating_sub(PEN_RING)..n {
                ring[i % PEN_RING] = p[i];
            }
            self.gpu.write(self.act.pen_ring.ptr, inkling_cuda::as_bytes(&ring))?;
        }
        let reused = if std::env::var_os("INKLING_NO_PREFILL").is_some() {
            ensure!(self.embeds.is_empty(), "INKLING_NO_PREFILL does not support images");
            self.reset()?;
            self.gpu.write(self.act.prompt.ptr, inkling_cuda::as_bytes(&p))?;
            self.gpu.write(self.act.step.ptr, inkling_cuda::as_bytes(&[p[0], 0, 0, 0]))?;
            for _ in 0..n - 1 {
                self.launch_graph(Tail::Prefill)?;
            }
            0
        } else {
            self.prepare_prompt(prompt)?
        };
        // the last prompt token runs as the first decode step
        self.gpu.write(self.act.step.ptr, inkling_cuda::as_bytes(&[p[n - 1], (n - 1) as i32, 0, 0]))?;
        // a constraint needs each token before the next mask: one step at a time
        let ahead: usize = if mask.is_some() { 1 } else { 3 };
        let words = self.cfg.n_vocab.div_ceil(32);
        self.gpu.write(self.act.mask_on.ptr, inkling_cuda::as_bytes(&[0i32]))?;
        let mut events = Vec::with_capacity(max_new);
        let mut out = Vec::with_capacity(max_new);
        let mut launched = 0;
        while out.len() < max_new {
            while launched < max_new && launched - out.len() < ahead {
                if let Some(m) = mask.as_mut() {
                    match m() {
                        Some(mut bits) => {
                            bits.resize(words, 0);
                            self.gpu.write(self.act.mask.ptr, inkling_cuda::as_bytes(&bits))?;
                            self.gpu.write(self.act.mask_on.ptr, inkling_cuda::as_bytes(&[1i32]))?;
                        }
                        None => self.gpu.write(self.act.mask_on.ptr, inkling_cuda::as_bytes(&[0i32]))?,
                    }
                }
                self.launch_graph(Tail::Greedy)?;
                events.push(self.gpu.stream.record_event(None)?);
                launched += 1;
            }
            events[out.len()].synchronize()?;
            let t = self.act.hist.read_i32(out.len()) as u32;
            out.push(t);
            if !on_token(t) {
                break;
            }
        }
        self.gpu.write(self.act.mask_on.ptr, inkling_cuda::as_bytes(&[0i32]))?;
        self.gpu.sync()?;
        self.check_cpu()?;
        // the device fed the last prompt token and every sampled token but the last launched one
        let mut fed = prompt.to_vec();
        fed.extend((0..launched - 1).map(|i| self.act.hist.read_i32(i) as u32));
        self.cache.tokens = fed;
        Ok((out, reused))
    }

    /// Run one token and return the greedy next token (argmax on the GPU, 4-byte readback).
    pub fn step_greedy(&mut self, token: u32, pos: usize) -> Result<u32> {
        self.run(token, pos, None)?;
        let nv = self.cfg.n_vocab_unpadded as i32;
        self.gpu.launch("ink_argmax", (1, 1), 1024, &[Arg::Ptr(self.act.logits.ptr), Arg::I32(nv), Arg::Ptr(self.act.sel.ptr)])?;
        let v = self.gpu.read_f32(self.act.sel.ptr, 1)?;
        Ok(v[0].to_bits())
    }

    /// Run one token at position `pos`; returns the logits over the unpadded vocabulary.
    pub fn step(&mut self, token: u32, pos: usize, capture: Option<Capture>) -> Result<Vec<f32>> {
        self.cache = PromptCache::default();
        self.run(token, pos, capture)?;
        self.gpu.read_f32(self.act.logits.ptr, self.cfg.n_vocab_unpadded)
    }

    /// Enqueue one token step (graph replay when possible); results stay on the device.
    fn run(&mut self, token: u32, pos: usize, capture: Option<Capture>) -> Result<()> {
        ensure!(pos < self.n_ctx, "position {pos} exceeds context {}", self.n_ctx);
        self.gpu.write(self.act.step.ptr, inkling_cuda::as_bytes(&[token as i32, pos as i32, 0, 0]))?;
        if capture.is_some() || !self.use_graph {
            self.enqueue_step(capture)?;
        } else {
            self.launch_graph(Tail::None)?;
        }
        Ok(())
    }
}
