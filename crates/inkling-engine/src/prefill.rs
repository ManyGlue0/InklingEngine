//! Batched prompt processing: the forward pass over chunks of up to `PREFILL_CHUNK` tokens, writing the same KV
//! cache and short-conv states as token-by-token decode. Mirrors `Engine::enqueue_step` operation for operation;
//! matmuls run on int8 tensor cores (`ink_gemm_q8`, grouped `ink_moe_*_g`), per-token kernels take a token grid
//! dimension.
//!
//! Layout: every per-token activation is `[T][width]`; multi-query buffers are `[j][T][width]` (T = chunk length).

use anyhow::{Result, ensure};
use inkling_cuda::{Arg, DevBuf, ExpW, GemmArgs, GemmSeg, Gpu, MixArgs, MixArgsQ, MvB4, MvSegB, NormQ4};

use crate::{Engine, F32, weights};

/// Tokens per prefill chunk (also sizes the sliding-window ring: window + chunk - 1 positions must fit).
pub const PREFILL_CHUNK: usize = 2048;
/// Position splits of the batched attention.
const PRE_SPLITS: usize = 4;

/// q8_1 buffer: int8 values + f32 scale per 32.
struct Q8 {
    q: DevBuf,
    d: DevBuf,
}
impl Q8 {
    fn new(gpu: &Gpu, n: usize) -> Result<Self> {
        Ok(Self { q: gpu.alloc(n)?, d: gpu.alloc(n / 32 * F32)? })
    }
    /// (values, scales) at element offset `off` (multiple of 32)
    fn at(&self, off: usize) -> (u64, u64) {
        (self.q.at(off), self.d.at(off / 32 * F32))
    }
}

pub(crate) struct PrefillBufs {
    pub(crate) tokens: DevBuf,
    pub(crate) step: DevBuf,
    x: DevBuf,
    xp: DevBuf,
    xprev: DevBuf,
    blocks: DevBuf, // 8 x [T][D] (block-boundary inputs: 1 prelude + 3 passes x 2 + 1 coda)
    a: DevBuf,      // [3][T][D]
    c: DevBuf,      // [3][T][D]
    nrm: DevBuf,    // [3][T][D]
    nq: Q8,         // [3][T][D]
    msg: DevBuf,    // [4][T][D]
    g: DevBuf,      // [4][T][R]
    qrg: DevBuf,    // [T][q | r | gate]
    sink: DevBuf,   // [T][sink 24 | dynk 32 | dynv 32]
    kraw: DevBuf,   // [T][1024]
    vraw: DevBuf,
    att: DevBuf, // [T][3072]
    attq: Q8,
    proj: DevBuf, // [T][D]
    f: DevBuf,
    f2: DevBuf,
    fnorm: DevBuf,
    fq: Q8,
    rlog: DevBuf, // [T][E]
    sel: DevBuf,  // [T][8] i32
    wts: DevBuf,  // [T][8]
    alist: DevBuf,
    inv: DevBuf,
    tiles: DevBuf, // int4 [max_tiles]
    ntiles: DevBuf,
    hq: Q8,       // [T*8][F]
    ya: DevBuf,   // [T*8][D]
    ysum: DevBuf, // [T][D]
    part: DevBuf,
    cnt: DevBuf,
    logits: Option<DevBuf>, // [LOGIT_ROWS][V] (validation only)
    spart: DevBuf, // split-K partials of ink_sgemm
    ga: DevBuf, // loop-control scratch [T][D] x 2, row norms [T] x 2
    fa: DevBuf,
    ic: DevBuf,
    im: DevBuf,
    // short-chain MoE (ink_moe8b, T <= MOE_B_MAX): expert hidden h [8T][F] float, int8 copy + scales per 16
    bh: DevBuf,
    bhq: DevBuf,
    bhd: DevBuf,
}

/// Options of a batched forward pass: feature capture for the drafter (`feat` != 0: tap j's input residual
/// rows go to feat + j * FEAT_ROWS * D, taps[0..3] are layer visits, tap 3 the final-norm output) and deferred conv
/// states (`hist` != 0: per-slot input history, see spec::hist_offset; chunks of at most spec::HIST_T tokens).
/// `rt` != 0: parallel streams, row r is one token of stream rt[2r] at position rt[2r + 1] (device i32 table; at most
/// MOE_B_MAX rows, distinct streams).
#[derive(Clone, Copy, Default)]
pub(crate) struct ChunkOpts {
    pub feat: u64,
    pub taps: [usize; 3],
    pub hist: u64,
    pub rt: u64,
}

/// Longest chunk the short-chain MoE kernel takes (ink_moe8b TMAX).
pub(crate) const MOE_B_MAX: usize = 8;

const LOGIT_ROWS: usize = 64;
/// Floats of split-K scratch for ink_sgemm.
const SGEMM_PART: usize = 8 << 20;

impl PrefillBufs {
    pub(crate) fn new(gpu: &Gpu, cfg: &inkling_core::Config) -> Result<Self> {
        let tp = PREFILL_CHUNK;
        let (d, kvw, qd, k) = (cfg.n_embd, cfg.n_head_kv * cfg.head_dim, cfg.n_head * cfg.head_dim, cfg.n_expert_used);
        let f = |n: usize| gpu.alloc(n * F32);
        let a = tp * k;
        Ok(Self {
            tokens: f(tp)?,
            step: f(4)?,
            x: f(tp * d)?,
            xp: f(tp * d)?,
            xprev: f(tp * d)?,
            blocks: f(8 * tp * d)?,
            a: f(3 * tp * d)?,
            c: f(3 * tp * d)?,
            nrm: f(3 * tp * d)?,
            nq: Q8::new(gpu, 3 * tp * d)?,
            msg: f(4 * tp * d)?,
            g: f(4 * tp * cfg.gate_rank)?,
            qrg: f(tp * (2 * qd + cfg.n_head * cfg.n_rel))?,
            sink: f(tp * (cfg.n_head + 2 * cfg.n_head_kv * cfg.n_conv))?,
            kraw: f(tp * kvw)?,
            vraw: f(tp * kvw)?,
            att: f(tp * qd)?,
            attq: Q8::new(gpu, tp * qd)?,
            proj: f(tp * d)?,
            f: f(tp * d)?,
            f2: f(tp * d)?,
            fnorm: f(tp * d)?,
            fq: Q8::new(gpu, tp * d)?,
            rlog: f(tp * cfg.n_expert)?,
            sel: f(a)?,
            wts: f(a)?,
            alist: f(a)?,
            inv: f(a)?,
            tiles: gpu.alloc((a / 64 + cfg.n_expert) * 16)?,
            ntiles: f(1)?,
            hq: Q8::new(gpu, a * cfg.n_ff_exp)?,
            ya: f(a * d)?,
            ysum: f(tp * d)?,
            part: f(tp * cfg.n_head_kv * PRE_SPLITS * 3 * (2 + cfg.head_dim))?,
            cnt: f(tp * cfg.n_head_kv)?,
            logits: None,
            spart: f(SGEMM_PART)?,
            ga: f(tp * d)?,
            fa: f(tp * d)?,
            ic: f(tp)?,
            im: f(tp)?,
            bh: f(8 * MOE_B_MAX * cfg.n_ff_exp)?,
            bhq: gpu.alloc(8 * MOE_B_MAX * cfg.n_ff_exp)?,
            bhd: f(8 * MOE_B_MAX * cfg.n_ff_exp / 16)?,
        })
    }
}

impl Engine {
    /// Process `tokens` at positions `pos0..`, writing KV cache and conv states (no logits). Chunked.
    pub(crate) fn prefill(&mut self, tokens: &[u32], pos0: usize) -> Result<()> {
        for (i, ch) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            self.prefill_chunk(ch, pos0 + i * PREFILL_CHUNK, None)?;
        }
        Ok(())
    }

    /// Fresh sequence: prefill `tokens` from position 0 (benchmarks).
    pub fn prefill_fresh(&mut self, tokens: &[u32]) -> Result<()> {
        self.reset()?;
        self.prefill(tokens, 0)
    }

    /// Prefill a fresh sequence and return the (soft-capped) logits of every position, row-major [n][vocab].
    /// Validation / perplexity path.
    pub fn prefill_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(tokens.len() * self.cfg.n_vocab_unpadded);
        self.prefill_each(tokens, &mut |_, row| out.extend_from_slice(row))?;
        Ok(out)
    }

    /// Process `tokens` at positions `pos0..` on top of the current state (one chunk) and return their soft-capped
    /// logits, row-major [n][vocab] (validation of the short-chain path).
    pub fn chunk_rows(&mut self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(tokens.len() * self.cfg.n_vocab_unpadded);
        self.cache = crate::PromptCache::default();
        self.prefill_chunk(tokens, pos0, Some(&mut |_, row: &[f32]| out.extend_from_slice(row)))?;
        Ok(out)
    }

    /// Prefill a fresh sequence and call `f(position, logits)` for every position (long-context evaluation without
    /// holding all logits in memory).
    pub fn prefill_each(&mut self, tokens: &[u32], f: &mut dyn FnMut(usize, &[f32])) -> Result<()> {
        self.reset()?;
        for (i, ch) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            self.prefill_chunk(ch, i * PREFILL_CHUNK, Some(&mut *f))?;
        }
        Ok(())
    }

    fn pre_mix(&self, states: &[u64], q: u64, nq: usize, out: u64, outn: u64, nw: [u64; 4], xq: (u64, u64), t: usize) -> Result<()> {
        ensure!(states.len() <= 6 && nq <= 4, "mix: too many states/queries");
        let mut m = MixArgs { q, out, outn, nw, n: states.len() as i32, nq: nq as i32, d: self.cfg.n_embd as i32, eps: self.cfg.rms_eps, ..Default::default() };
        m.st[..states.len()].copy_from_slice(states);
        let k = match nq { 1 => "ink_mix6_1", 3 => "ink_mix6_3", 4 => "ink_mix6_4", _ => anyhow::bail!("mix: nq = {nq} unsupported") };
        self.gpu.launch(k, (1, t as u32), 768, &[Arg::MixQ(MixArgsQ { m, xq: xq.0, xd: xq.1 })])
    }

    pub(crate) fn pre_norm(&self, items: &[(u64, u64, u64, (u64, u64))], t: usize) -> Result<()> {
        let mut a = NormQ4::default();
        for (j, &(x, w, y, (q, d))) in items.iter().enumerate() {
            a.x[j] = x;
            a.w[j] = w;
            a.y[j] = y;
            a.xq[j] = q;
            a.xd[j] = d;
        }
        self.gpu.launch("ink_rmsnorm_q2", (items.len() as u32, t as u32), 768, &[Arg::NormQ(a), Arg::I32(self.cfg.n_embd as i32), Arg::F32(self.cfg.rms_eps)])
    }

    /// int8 GEMM segments (weights, (xq, xd), y, rows, ldy), shared K.
    pub(crate) fn pre_gemm(&self, segs: &[(weights::Q8M, (u64, u64), u64, usize, usize)], k: usize, t: usize) -> Result<()> {
        if t <= 6 && segs.len() <= 4 && segs.iter().all(|s| s.0.fmt == 0) && std::env::var_os("INKLING_NO_MV_B").is_none() {
            // short chain: every weight row read once for all t tokens
            let mut a = inkling_cuda::MvQB4 { nseg: segs.len() as i32, k: k as i32, t: t as i32, ..Default::default() };
            let mut rows = 0;
            for (i, &(w, (x, xd), y, n, ldy)) in segs.iter().enumerate() {
                a.s[i] = inkling_cuda::MvSegQB { qs: w.qs, d: w.d, xq: x, xd, y, n: n as i32, ldy: ldy as i32 };
                rows += n;
            }
            return self.gpu.launch("ink_mv_q8q_b", (rows.div_ceil(8) as u32, 1), 256, &[Arg::MvQB(a)]);
        }
        if segs.len() > 4 || segs.iter().any(|s| s.0.fmt != 0) || std::env::var_os("INKLING_GEN_DENSE").is_some() {
            ensure!(segs.len() <= 8, "pre_gemm: too many segments");
            let mut a = inkling_cuda::GemmGen { nseg: segs.len() as i32, t: t as i32, k: k as i32, ..Default::default() };
            let mut tiles = 0;
            for (i, &(w, (x, xd), y, n, ldy)) in segs.iter().enumerate() {
                a.s[i] = inkling_cuda::GemmSegG { w: w.qs, wd: w.d, x, xd, y, n: n as i32, ldy: ldy as i32, fmt: w.fmt, pad: 0 };
                tiles += n.div_ceil(64);
            }
            return self.gpu.launch("ink_gemm_gen", (tiles as u32, t.div_ceil(64) as u32), 128, &[Arg::GemmG(a)]);
        }
        let mut a = GemmArgs { nseg: segs.len() as i32, t: t as i32, k: k as i32, ..Default::default() };
        let mut tiles = 0;
        for (i, &(w, (x, xd), y, n, ldy)) in segs.iter().enumerate() {
            a.s[i] = GemmSeg { w: w.qs, wd: w.d, x, xd, y, n: n as i32, ldy: ldy as i32 };
            tiles += n.div_ceil(64);
        }
        self.gpu.launch("ink_gemm_q8", (tiles as u32, t.div_ceil(64) as u32), 128, &[Arg::Gemm(a)])
    }

    /// Y[t][n] = (acc ? Y : 0) + rs[t] * X[t] . W[n]  (small SIMT GEMM; W f32 or f16)
    #[allow(clippy::too_many_arguments)]
    fn sgemm(&self, x: u64, ldx: usize, w: u64, ldw: usize, f16: bool, y: u64, ldy: usize, rs: u64, acc: bool, t: usize, n: usize, k: usize) -> Result<()> {
        // split K until the grid has ~2 blocks per SM (largest divisor of K/16 not above the target), partial
        // sums reduced in a fixed order (deterministic)
        let tiles = n.div_ceil(64) * t.div_ceil(64);
        let target = (2 * self.gpu.info.sm_count as usize).div_ceil(tiles).max(1);
        let ks = k / 16;
        let max_split = SGEMM_PART / (t * n).max(1);
        let s = (1..=ks.min(target).min(max_split.max(1))).rev().find(|&s| ks % s == 0).unwrap_or(1);
        let part = self.pre.as_ref().unwrap().spart.ptr;
        self.gpu.launch3("ink_sgemm", (n.div_ceil(64) as u32, t.div_ceil(64) as u32, s as u32), 256, &[
            Arg::Ptr(x), Arg::I32(ldx as i32), Arg::Ptr(w), Arg::I32(ldw as i32), Arg::I32(f16 as i32), Arg::Ptr(y), Arg::I32(ldy as i32),
            Arg::Ptr(rs), Arg::I32(acc as i32), Arg::I32(t as i32), Arg::I32(n as i32), Arg::I32(k as i32), Arg::Ptr(part),
        ])?;
        if s > 1 {
            self.gpu.launch("ink_sgemm_reduce", ((t * n).div_ceil(256) as u32, 1), 256, &[
                Arg::Ptr(part), Arg::I32(s as i32), Arg::Ptr(y), Arg::I32(ldy as i32), Arg::Ptr(rs), Arg::I32(acc as i32), Arg::I32(t as i32), Arg::I32(n as i32),
            ])?;
        }
        Ok(())
    }

    /// Loop controls for T tokens as small GEMMs: g = rms(c) Wd_c^T + rms(m) Wd_m^T, then
    /// out = (1 - tanh(g Wf^T)) c + tanh(g Wg^T) m. Each entry: (control index, c, m, out), all [T][D].
    fn pre_controls(&self, lp: &weights::LoopW, ctrls: &[(usize, u64, u64, u64)], t: usize) -> Result<()> {
        ensure!(lp.f16, "batched loop controls need F16 loop weights");
        let p = self.pre.as_ref().unwrap();
        let (d, r) = (self.cfg.n_embd, self.cfg.gate_rank);
        if t <= 16 && std::env::var_os("INKLING_NO_CTRL_B").is_none() {
            // short chunk: the decode kernels with a token grid dimension, all controls in one launch each
            let mut a = inkling_cuda::Ctrl4 { n: ctrls.len() as i32, ..Default::default() };
            for (j, &(idx, c, m, out)) in ctrls.iter().enumerate() {
                a.c[j] = c;
                a.m[j] = m;
                a.wd[j] = lp.down[idx];
                a.wg[j] = lp.gain[idx];
                a.wf[j] = lp.forget[idx];
                a.g[j] = p.g.at(j * t * r * F32);
                a.out[j] = out;
            }
            let n = ctrls.len() as u32;
            self.gpu.launch3("ink_ctrl_down2", (r as u32, n, t as u32), 192, &[Arg::Ctrl(a), Arg::I32(d as i32), Arg::I32(r as i32)])?;
            return self.gpu.launch3("ink_ctrl_up_h", ((d / 8) as u32, n, t as u32), 256, &[Arg::Ctrl(a), Arg::I32(d as i32), Arg::I32(r as i32)]);
        }
        let g = p.g.ptr; // [T][R]
        for &(idx, c, m, out) in ctrls {
            self.gpu.launch("ink_rowrms", (t as u32, 1), 256, &[Arg::Ptr(c), Arg::I32(d as i32), Arg::I32(d as i32), Arg::Ptr(p.ic.ptr), Arg::F32(1e-6)])?;
            self.gpu.launch("ink_rowrms", (t as u32, 1), 256, &[Arg::Ptr(m), Arg::I32(d as i32), Arg::I32(d as i32), Arg::Ptr(p.im.ptr), Arg::F32(1e-6)])?;
            let wd = lp.down[idx]; // [R][2D] f16
            self.sgemm(c, d, wd, 2 * d, true, g, r, p.ic.ptr, false, t, r, d)?;
            self.sgemm(m, d, wd + (d * 2) as u64, 2 * d, true, g, r, p.im.ptr, true, t, r, d)?;
            self.sgemm(g, r, lp.gain[idx], r, true, p.ga.ptr, d, 0, false, t, d, r)?;
            self.sgemm(g, r, lp.forget[idx], r, true, p.fa.ptr, d, 0, false, t, d, r)?;
            self.gpu.launch("ink_ctrl_combine", ((t * d).div_ceil(256) as u32, 1), 256, &[
                Arg::Ptr(p.ga.ptr), Arg::Ptr(p.fa.ptr), Arg::Ptr(c), Arg::Ptr(m), Arg::Ptr(out), Arg::I32((t * d) as i32),
            ])?;
        }
        Ok(())
    }

    /// Post conv + residual; `hist` != 0 defers the state update (the inputs are kept there for `commit_conv`).
    /// `rt` != 0: parallel streams (row table, states `cs` floats apart).
    fn pre_conv_res(&self, xin: u64, w: u64, state: u64, x: u64, accumulate: bool, t: usize, hist: u64, rt: u64, cs: usize) -> Result<()> {
        let d = self.cfg.n_embd;
        self.gpu.launch("ink_conv_res_b", (d.div_ceil(256) as u32, t as u32), 256, &[
            Arg::Ptr(xin), Arg::Ptr(w), Arg::Ptr(state), Arg::Ptr(x), Arg::I32(accumulate as i32), Arg::I32(d as i32), Arg::Ptr(rt), Arg::I32(cs as i32),
        ])?;
        self.conv_adv(xin, state, d, t, hist, rt, cs)
    }

    /// Advance a conv state past the chunk, or (hist != 0) keep its inputs for a later commit; with a row table
    /// (`rt` != 0) each row advances its own stream's state by one input.
    fn conv_adv(&self, x: u64, state: u64, ch: usize, t: usize, hist: u64, rt: u64, cs: usize) -> Result<()> {
        if rt != 0 {
            return self.gpu.launch("ink_conv_state_m", (ch.div_ceil(256) as u32, t as u32), 256, &[
                Arg::Ptr(x), Arg::Ptr(rt), Arg::Ptr(state), Arg::I32(cs as i32), Arg::I32(ch as i32),
            ]);
        }
        match hist {
            0 => self.conv_state(x, state, ch, t),
            h if h == x => Ok(()), // already written there
            h => self.copy_n(x, h, t * ch * F32),
        }
    }

    fn conv_state(&self, x: u64, state: u64, ch: usize, t: usize) -> Result<()> {
        self.gpu.launch("ink_conv_state_b", (ch.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(x), Arg::I32(t as i32), Arg::Ptr(state), Arg::I32(ch as i32)])
    }

    pub(crate) fn copy_n(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
        unsafe { inkling_cuda::cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, self.gpu.stream.cu_stream()) }?;
        Ok(())
    }

    /// Grouped expert GEMMs of one layer: directly when its experts are in VRAM; for a CPU-offloaded layer, the
    /// pinned host copy streams in groups of `offload::STREAM_GROUP` experts into two alternating VRAM buffers on
    /// the copy stream, each group's GEMMs (restricted to its expert range) overlapping the next group's copy.
    fn pre_experts(&self, phys: usize, lw: &weights::LayerW, run: &dyn Fn(ExpW, ExpW) -> Result<()>) -> Result<()> {
        let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt.code(), erange: 0 };
        let Some(c) = self.cpu.as_ref().filter(|c| c.plan.on_cpu[phys]) else {
            return run(ew(lw.gate_up), ew(lw.down));
        };
        let cl = c.moe.layers[phys].as_ref().expect("offloaded layer");
        let (gq, gs) = cl.gate_up.row_bytes();
        let (dq, ds) = cl.down.row_bytes();
        // bytes per expert of each plane, and the planes' offsets inside a group buffer
        let per = [gq * cl.gate_up.rows, gs * cl.gate_up.rows, dq * cl.down.rows, ds * cl.down.rows];
        let src = [cl.gate_up.qs, cl.gate_up.sc, cl.down.qs, cl.down.sc];
        let gsz = crate::offload::STREAM_GROUP;
        let mut off = [0usize; 4];
        for i in 1..4 {
            off[i] = off[i - 1] + gsz * per[i - 1];
        }
        // natively stored formats may read a few bytes past the last block of the group
        ensure!(off[3] + gsz * per[3] + weights::NATIVE_PAD <= c.bufs[0].bytes, "prefill streaming buffer too small");
        let ne = self.cfg.n_expert;
        let cs = &c.copy_stream;
        // the buffers may still be in use by the previous offloaded layer's GEMMs
        cs.wait(&self.gpu.stream.record_event(None)?)?;
        let mut done: [Option<inkling_cuda::cudarc::driver::CudaEvent>; 2] = [None, None];
        for (j, e0) in (0..ne).step_by(gsz).enumerate() {
            let e1 = (e0 + gsz).min(ne);
            let b = &c.bufs[j % 2];
            if let Some(ev) = done[j % 2].take() {
                cs.wait(&ev)?;
            }
            for i in 0..4 {
                if per[i] == 0 {
                    continue; // native formats have no scale plane
                }
                let r = unsafe {
                    inkling_cuda::cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        b.at(off[i]),
                        src[i].add(e0 * per[i]) as *const std::ffi::c_void,
                        (e1 - e0) * per[i],
                        cs.cu_stream(),
                    )
                };
                r.result()?;
            }
            self.gpu.stream.wait(&cs.record_event(None)?)?;
            // base pointers shifted so expert e0 lands at the start of its plane (only experts in [e0, e1) are read)
            let base = |i: usize| b.at(off[i]).wrapping_sub((e0 * per[i]) as u64);
            let erange = (e0 | e1 << 16) as i32;
            let gu = ExpW { qs: base(0), sc: base(1), scale: lw.gate_up.scale, fmt: lw.gate_up.fmt.code(), erange };
            let dn = ExpW { qs: base(2), sc: base(3), scale: lw.down.scale, fmt: lw.down.fmt.code(), erange };
            run(gu, dn)?;
            done[j % 2] = Some(self.gpu.stream.record_event(None)?);
        }
        Ok(())
    }

    fn prefill_chunk(&mut self, tokens: &[u32], pos0: usize, logits_out: Option<&mut dyn FnMut(usize, &[f32])>) -> Result<()> {
        let t = tokens.len();
        ensure!(t > 0 && t <= PREFILL_CHUNK, "bad chunk length");
        ensure!(pos0 + t <= self.n_ctx, "prompt exceeds the context ({})", self.n_ctx);
        self.upload_chunk(tokens, pos0)?;
        self.enqueue_chunk(t, pos0, ChunkOpts::default())?;
        if let Some(out) = logits_out {
            self.chunk_logits(t, pos0, out)?;
        }
        Ok(())
    }

    /// Chunk inputs: token ids and the start position (read on the device, so an enqueued chunk is replayable).
    pub(crate) fn upload_chunk(&self, tokens: &[u32], pos0: usize) -> Result<()> {
        let p = self.pre.as_ref().expect("prefill buffers");
        let ids: Vec<i32> = tokens.iter().map(|&x| if x >= crate::EMBED_TOKEN_BASE { 0 } else { x as i32 }).collect();
        self.gpu.write(p.tokens.ptr, inkling_cuda::as_bytes(&ids))?;
        self.gpu.write(p.step.ptr, inkling_cuda::as_bytes(&[0i32, pos0 as i32, 0, 0]))
    }

    /// The forward pass of a chunk of `t` tokens (inputs already uploaded); `pos0` only places image rows.
    pub(crate) fn enqueue_chunk(&self, t: usize, pos0: usize, o: ChunkOpts) -> Result<()> {
        let p = self.pre.as_ref().expect("prefill buffers");
        let cfg = &self.cfg;
        let (d, kvw, qd, r) = (cfg.n_embd, cfg.n_head_kv * cfg.head_dim, cfg.n_head * cfg.head_dim, cfg.gate_rank);
        let qs = 2 * qd + cfg.n_head * cfg.n_rel; // qrg row
        let ss = cfg.n_head + 2 * cfg.n_head_kv * cfg.n_conv; // sink row
        let td = t * d;
        let row = |j: usize| j * td * F32; // j-th [T][D] slab
        let nqv = |j: usize| p.nq.at(j * td);

        if self.w.token_embd_fmt == 8 {
            self.gpu.launch("ink_embed_b", (t as u32, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::Ptr(p.tokens.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(p.x.ptr), Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
            ])?;
        } else {
            self.gpu.launch("ink_embed_gen", (t as u32, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::I32(self.w.token_embd_fmt), Arg::Ptr(p.tokens.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(p.x.ptr),
                Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
            ])?;
        }
        // image embeddings replace their rows as they are (no embedding norm, as in the reference)
        for (epos, e) in &self.embeds {
            let (lo, hi) = ((*epos).max(pos0), (epos + e.n).min(pos0 + t));
            if lo < hi {
                let src = &e.data[(lo - epos) * d..(hi - epos) * d];
                self.gpu.write(p.x.at((lo - pos0) * d * F32), inkling_cuda::as_bytes(src))?;
            }
        }

        let mut blocks: Vec<u64> = Vec::new();
        let mut prelude_blocks: Vec<u64> = Vec::new();
        let mut bank: Vec<u64> = Vec::new();
        let mut n_push = 0usize;

        for (vi, ps) in cfg.schedule().into_iter().enumerate() {
            if o.feat != 0 {
                if let Some(j) = o.taps.iter().position(|&v| v == vi) {
                    self.copy_n(p.x.ptr, o.feat + (j * crate::dspark::FEAT_ROWS * d * F32) as u64, td * F32)?;
                }
            }
            let hist = |site: usize| if o.hist != 0 { o.hist + crate::spec::hist_offset(ps.slot, site, d) as u64 } else { 0 };
            let lw = self.w.layers[ps.phys];
            if ps.phys == cfg.core_start {
                if ps.pass == 0 {
                    self.copy_n(p.x.ptr, p.xp.ptr, td * F32)?;
                    prelude_blocks = blocks.clone();
                } else {
                    bank = blocks[prelude_blocks.len() + 1..].to_vec();
                    self.copy_n(p.x.ptr, p.xprev.ptr, td * F32)?;
                    self.copy_n(p.xp.ptr, p.x.ptr, td * F32)?;
                    blocks = prelude_blocks.clone();
                }
            }
            // parallel streams: stream 0's buffers, the kernels add each row's stream offset
            let slot = if o.rt != 0 { self.slot_base(ps.slot) } else { self.slots[ps.slot] };
            let (ks, vs, cs) = if o.rt != 0 { self.stream_strides(ps.slot) } else { (0, 0, 0) };
            let (conv_k, conv_v) = (slot.conv.ptr, slot.conv.at(3 * kvw * F32));
            let (conv_attn, conv_ffn) = (slot.conv.at(6 * kvw * F32), slot.conv.at((6 * kvw + 3 * d) * F32));
            let looped = ps.pass > 0;
            let lp = lw.lp;

            if looped {
                let mut st = bank.clone();
                st.push(p.xprev.ptr);
                self.pre_mix(&st, lp.unwrap().msg_query, 4, p.msg.ptr, 0, [0; 4], (0, 0), t)?;
            }
            let boundary = ps.phys % cfg.attn_res_block == 0;
            let inp_copy = if boundary {
                ensure!(n_push < 8, "too many block boundaries for the prefill buffers");
                let dst = p.blocks.at(n_push * PREFILL_CHUNK * d * F32);
                n_push += 1;
                self.copy_n(p.x.ptr, dst, td * F32)?;
                Some(dst)
            } else {
                None
            };

            // attention inputs
            let n = [p.nrm.ptr, p.nrm.at(row(1)), p.nrm.at(row(2))];
            let norms = if lw.is_swa { [lw.attn_norm; 3] } else { [lw.q_norm, lw.k_norm, lw.v_norm] };
            let nqry = if lw.is_swa { 1 } else { 3 };
            let shared_input = lw.is_swa && !looped;
            if !looped {
                if lw.attn_res_query != 0 {
                    let mut st = blocks.clone();
                    st.push(p.x.ptr);
                    let nw = if lw.is_swa { [norms[0], 0, 0, 0] } else { [norms[0], norms[1], norms[2], 0] };
                    self.pre_mix(&st, lw.attn_res_query, nqry, p.a.ptr, p.nrm.ptr, nw, nqv(0), t)?;
                } else {
                    let items: Vec<_> = (0..nqry).map(|j| (p.x.ptr, norms[j], n[j], nqv(j))).collect();
                    self.pre_norm(&items, t)?;
                }
            } else {
                let mut st = blocks.clone();
                st.push(p.x.ptr);
                self.pre_mix(&st, lw.attn_res_query, nqry, p.a.ptr, 0, [0; 4], (0, 0), t)?;
                let inp = |j: usize| if lw.is_swa { p.a.ptr } else { p.a.at(row(j)) };
                let ctl: Vec<_> = (0..3).map(|j| (j, inp(j), p.msg.at(row(j)), p.c.at(row(j)))).collect();
                self.pre_controls(&lp.unwrap(), &ctl, t)?;
                let items: Vec<_> = (0..3).map(|j| (p.c.at(row(j)), norms[j], n[j], nqv(j))).collect();
                self.pre_norm(&items, t)?;
            }
            let (iq, ik, iv) = if shared_input { (0, 0, 0) } else { (0, 1, 2) };

            // projections
            let mut segs: Vec<_> = self.qrg_segs(&lw, nqv(iq), p.qrg.ptr).into_iter().map(|(w, x, y, n)| (w, x, y, n, qs)).collect();
            // deferred conv states: the conv inputs are written straight into the history
            let kraw = if o.hist != 0 { hist(0) } else { p.kraw.ptr };
            let vraw = if o.hist != 0 { hist(1) } else { p.vraw.ptr };
            let proj = if o.hist != 0 { hist(2) } else { p.proj.ptr };
            segs.push((lw.wk, nqv(ik), kraw, kvw, kvw));
            segs.push((lw.wv, nqv(iv), vraw, kvw, kvw));
            self.pre_gemm(&segs, d, t)?;
            let nkc = cfg.n_head_kv * cfg.n_conv;
            let mut mb = MvB4 { k: d as i32, ..Default::default() };
            mb.s[0] = MvSegB { w: lw.wsink, x: n[iq], y: p.sink.ptr, n: cfg.n_head as i32, f32w: lw.small_f32 };
            mb.nseg = 1;
            let mut rows = cfg.n_head;
            if !lw.is_swa {
                mb.s[1] = MvSegB { w: lw.kconv_proj, x: n[ik], y: p.sink.at(cfg.n_head * F32), n: nkc as i32, f32w: lw.small_f32 };
                mb.s[2] = MvSegB { w: lw.vconv_proj, x: n[iv], y: p.sink.at((cfg.n_head + nkc) * F32), n: nkc as i32, f32w: lw.small_f32 };
                mb.nseg = 3;
                rows += 2 * nkc;
            }
            self.gpu.launch("ink_mv_bf16b", (rows.div_ceil(8) as u32, t as u32), 256, &[Arg::MvB(mb), Arg::I32(ss as i32)])?;
            // K/V convs + clip/norm + cache write, q/r clip + tau
            self.gpu.launch("ink_attn_prep_b", (64, t as u32), 128, &[
                Arg::Ptr(p.qrg.ptr), Arg::I32(qs as i32), Arg::I32(qd as i32), Arg::Ptr(kraw), Arg::Ptr(vraw),
                Arg::Ptr(lw.q_clip), Arg::Ptr(lw.rel_clip), Arg::Ptr(lw.k_conv), Arg::Ptr(lw.v_conv), Arg::Ptr(conv_k), Arg::Ptr(conv_v),
                Arg::Ptr(p.sink.at(cfg.n_head * F32)), Arg::I32(ss as i32), Arg::I32(!lw.is_swa as i32), Arg::Ptr(p.step.ptr),
                Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(!lw.is_swa as i32),
                Arg::F32(cfg.qk_clip), Arg::F32(cfg.rel_clip), Arg::F32(cfg.log_scale_floor), Arg::F32(cfg.log_scale_alpha), Arg::F32(cfg.rms_eps),
                Arg::I32(self.kv_bits(lw.is_swa).0), Arg::I32(self.kv_bits(lw.is_swa).1), Arg::I32(slot.kv_fmt),
                Arg::Ptr(o.rt), Arg::Ptr(ks as u64), Arg::Ptr(vs as u64), Arg::I32(cs as i32),
            ])?;
            self.conv_adv(kraw, conv_k, kvw, t, hist(0), o.rt, cs)?;
            self.conv_adv(vraw, conv_v, kvw, t, hist(1), o.rt, cs)?;
            let window = if lw.is_swa { cfg.n_swa as i32 } else { 0 };
            let (oq, od) = p.attq.at(0);
            // short chunks (speculative verify): the decode attention per token with decode's position splits (the
            // prefill kernel would run only n_head blocks); INKLING_PRE_ATTN=split|mma|prefill forces a kernel
            let pre_attn = std::env::var("INKLING_PRE_ATTN").ok();
            let short = (t <= 16 && pre_attn.as_deref() != Some("prefill")) || o.rt != 0;
            if short || matches!(pre_attn.as_deref(), Some("split" | "mma")) {
                let splits = if short { self.attn_splits as usize } else { PRE_SPLITS };
                let k = if pre_attn.as_deref() == Some("split") && o.rt == 0 { "ink_attn_split" } else { "ink_attn_mma" };
                self.gpu.launch3(k, (cfg.n_head_kv as u32, splits as u32, t as u32), 128, &[
                Arg::Ptr(p.qrg.ptr), Arg::Ptr(p.qrg.at(qd * F32)), Arg::Ptr(p.sink.ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr),
                Arg::Ptr(lw.rel_bias), Arg::Ptr(p.qrg.at((qd + cfg.n_head * cfg.n_rel) * F32)), Arg::Ptr(p.att.ptr), Arg::Ptr(oq), Arg::Ptr(od),
                Arg::Ptr(p.step.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(window), Arg::I32(cfg.rel_extent as i32),
                Arg::Ptr(p.part.ptr), Arg::Ptr(p.cnt.ptr), Arg::I32(qs as i32), Arg::I32(ss as i32), Arg::I32(slot.kv_fmt),
                Arg::Ptr(o.rt), Arg::Ptr(ks as u64), Arg::Ptr(vs as u64),
            ])?;
            } else {
                self.gpu.launch("ink_attn_prefill", (t.div_ceil(64) as u32, cfg.n_head as u32), 128, &[
                    Arg::Ptr(p.qrg.ptr), Arg::I32(qs as i32), Arg::I32(qd as i32), Arg::I32((qd + cfg.n_head * cfg.n_rel) as i32),
                    Arg::Ptr(p.sink.ptr), Arg::I32(ss as i32), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32),
                    Arg::I32(window), Arg::I32(cfg.rel_extent as i32), Arg::Ptr(lw.rel_bias), Arg::Ptr(p.step.ptr), Arg::I32(t as i32),
                    Arg::Ptr(p.att.ptr), Arg::Ptr(oq), Arg::Ptr(od), Arg::I32(slot.kv_fmt),
                ])?;
            }
            self.pre_gemm(&[(lw.wo, (oq, od), proj, d, d)], qd, t)?;
            if let Some(dst) = inp_copy {
                blocks.push(dst);
            }
            self.pre_conv_res(proj, lw.attn_post_conv, conv_attn, p.x.ptr, inp_copy.is_none(), t, hist(2), o.rt, cs)?;

            // FFN input
            let mut st = blocks.clone();
            st.push(p.x.ptr);
            let fq = p.fq.at(0);
            if looped {
                self.pre_mix(&st, lw.ffn_res_query, 1, p.f.ptr, 0, [0; 4], (0, 0), t)?;
                self.pre_controls(&lp.unwrap(), &[(3, p.f.ptr, p.msg.at(row(3)), p.f2.ptr)], t)?;
                self.pre_norm(&[(p.f2.ptr, lw.ffn_norm, p.fnorm.ptr, fq)], t)?;
            } else {
                self.pre_mix(&st, lw.ffn_res_query, 1, p.f.ptr, p.fnorm.ptr, [lw.ffn_norm, 0, 0, 0], fq, t)?;
            }

            // MoE: router, routing, grouped expert GEMMs, weighted combine, post conv + residual
            let k = cfg.n_expert_used;
            if t <= 16 && std::env::var_os("INKLING_NO_ROUTER_B").is_none() {
                self.gpu.launch("ink_router_b", (cfg.n_expert as u32, t as u32), 192, &[Arg::Ptr(p.fnorm.ptr), Arg::Ptr(lw.router), Arg::Ptr(p.rlog.ptr), Arg::I32(cfg.n_expert as i32)])?;
            } else {
                self.sgemm(p.fnorm.ptr, d, lw.router, d, false, p.rlog.ptr, cfg.n_expert, 0, false, t, cfg.n_expert, d)?;
            }
            self.gpu.launch("ink_topk_b", (t as u32, 1), 32, &[
                Arg::Ptr(p.rlog.ptr), Arg::Ptr(lw.probs_bias), Arg::Ptr(p.sel.ptr), Arg::Ptr(p.wts.ptr), Arg::I32(cfg.n_expert as i32),
                Arg::I32(k as i32), Arg::F32(cfg.expert_weights_scale),
            ])?;
            let on_gpu = self.cpu.as_ref().is_none_or(|c| !c.plan.on_cpu[ps.phys]);
            if t <= MOE_B_MAX && self.moe_blocks > 0 && on_gpu && lw.gate_up.fmt.tuned() && lw.down.fmt.tuned()
                && std::env::var_os("INKLING_NO_MOE_B").is_none()
            {
                // short chain: gate_up + down + expert sum + causal post conv + residual, one persistent kernel
                let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt.code(), erange: 0 };
                self.gpu.launch(if t <= 4 { "ink_moe8b4" } else { "ink_moe8b8" }, (self.moe_blocks, 1), 256, &[
                    Arg::Exp(ew(lw.gate_up)), Arg::Exp(ew(lw.down)), Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(p.sel.ptr), Arg::Ptr(p.wts.ptr),
                    Arg::I32(t as i32), Arg::Ptr(p.bh.ptr), Arg::Ptr(p.bhq.ptr), Arg::Ptr(p.bhd.ptr), Arg::Ptr(p.ya.ptr),
                    Arg::Ptr(lw.ffn_post_conv), Arg::Ptr(conv_ffn), Arg::Ptr(p.x.ptr), Arg::Ptr(hist(3)), Arg::Ptr(self.act.bar.ptr),
                    Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta), Arg::Ptr(self.moe_b_tl.as_ref().map_or(0, |b| b.ptr)),
                    Arg::Ptr(o.rt), Arg::I32(cs as i32),
                ])?;
                continue;
            }
            if t <= MOE_B_MAX && k == 8 && on_gpu && std::env::var_os("INKLING_NO_MOE_B").is_none() {
                // short chain, any other format pairing (natively stored, or one tensor NVFP4/Q8_0): each union expert
                // read once for all its tokens, per-token arithmetic as the decode kernels (ink_moe_*_gen / *7)
                let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt.code(), erange: 0 };
                let (hq, hd) = p.hq.at(0);
                let un = p.alist.ptr; // the union (genb::Union, 641 ints) in the routing scratch
                self.gpu.launch("ink_moe_union_genb", (1, 1), 64, &[Arg::Ptr(p.sel.ptr), Arg::I32(t as i32), Arg::Ptr(un)])?;
                self.gpu.launch("ink_moe_gate_up_genb", ((cfg.n_ff_exp / 16) as u32, (8 * t) as u32), 256, &[
                    Arg::Exp(ew(lw.gate_up)), Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(un), Arg::Ptr(hq), Arg::Ptr(hd),
                    Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
                ])?;
                self.gpu.launch("ink_moe_down_genb", ((d / 64) as u32, (8 * t) as u32), 256, &[
                    Arg::Exp(ew(lw.down)), Arg::Ptr(hq), Arg::Ptr(hd), Arg::Ptr(un), Arg::Ptr(p.ya.ptr),
                ])?;
                self.gpu.launch("ink_moe_sum_genb", ((d / 8) as u32, 1), 256, &[
                    Arg::Exp(ew(lw.down)), Arg::Ptr(p.ya.ptr), Arg::Ptr(p.sel.ptr), Arg::Ptr(p.wts.ptr), Arg::I32(t as i32),
                    Arg::Ptr(lw.ffn_post_conv), Arg::Ptr(conv_ffn), Arg::Ptr(p.x.ptr), Arg::Ptr(hist(3)), Arg::Ptr(o.rt), Arg::I32(cs as i32),
                ])?;
                continue;
            }
            let na = t * k;
            self.gpu.launch("ink_moe_route", (1, 1), 1024, &[
                Arg::Ptr(p.sel.ptr), Arg::I32(na as i32), Arg::Ptr(p.alist.ptr), Arg::Ptr(p.inv.ptr), Arg::Ptr(p.tiles.ptr), Arg::Ptr(p.ntiles.ptr),
            ])?;
            let max_tiles = (na / 64 + cfg.n_expert) as u32;
            let (hq, hd) = p.hq.at(0);
            let experts = |gu: ExpW, dn: ExpW| -> Result<()> {
                self.gpu.launch("ink_moe_gate_up_g", ((cfg.n_ff_exp / 32) as u32, max_tiles), 128, &[
                    Arg::Exp(gu), Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(p.alist.ptr), Arg::Ptr(p.tiles.ptr), Arg::Ptr(p.ntiles.ptr),
                    Arg::Ptr(hq), Arg::Ptr(hd), Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
                ])?;
                self.gpu.launch("ink_moe_down_g", ((d / 64) as u32, max_tiles), 128, &[
                    Arg::Exp(dn), Arg::Ptr(hq), Arg::Ptr(hd), Arg::Ptr(p.tiles.ptr), Arg::Ptr(p.ntiles.ptr), Arg::Ptr(p.ya.ptr),
                ])
            };
            self.pre_experts(ps.phys, &lw, &experts)?;
            self.gpu.launch("ink_moe_combine", (d.div_ceil(256) as u32, t as u32), 256, &[
                Arg::Ptr(p.ya.ptr), Arg::Ptr(p.inv.ptr), Arg::Ptr(p.sel.ptr), Arg::Ptr(p.wts.ptr), Arg::Ptr(lw.down.scale), Arg::Ptr(p.ysum.ptr), Arg::I32(d as i32),
            ])?;
            self.pre_conv_res(p.ysum.ptr, lw.ffn_post_conv, conv_ffn, p.x.ptr, true, t, hist(3), o.rt, cs)?;
        }
        let _ = r;

        // output head input (normalized, q8_1 in fq) of every position; only the logits paths read it
        let mut st = blocks.clone();
        st.push(p.x.ptr);
        self.pre_mix(&st, self.w.output_res_query, 1, p.f.ptr, p.fnorm.ptr, [self.w.output_norm, 0, 0, 0], p.fq.at(0), t)?;
        if o.feat != 0 {
            // the last feature tap: the final-norm output
            self.copy_n(p.fnorm.ptr, o.feat + (3 * crate::dspark::FEAT_ROWS * d * F32) as u64, td * F32)?;
        }
        Ok(())
    }

    pub(crate) fn logit_rows(&mut self) -> Result<u64> {
        if self.pre.as_ref().unwrap().logits.is_none() {
            let b = self.gpu.alloc(LOGIT_ROWS * self.cfg.n_vocab * F32)?;
            self.pre.as_mut().unwrap().logits = Some(b);
        }
        Ok(self.pre.as_ref().unwrap().logits.as_ref().unwrap().ptr)
    }

    /// Output head over rows r0..r0 + rows of the last chunk (rows <= LOGIT_ROWS), raw logits into the row buffer.
    pub(crate) fn enqueue_head(&self, lg: u64, r0: usize, rows: usize) -> Result<()> {
        let p = self.pre.as_ref().unwrap();
        let nv = self.cfg.n_vocab;
        self.pre_gemm(&[(self.w.output, p.fq.at(r0 * self.cfg.n_embd), lg, nv, nv)], self.cfg.n_embd, rows)
    }

    /// Output head (validation path): soft-capped logits of every position of the last chunk.
    fn chunk_logits(&mut self, t: usize, pos0: usize, out: &mut dyn FnMut(usize, &[f32])) -> Result<()> {
        let lg = self.logit_rows()?;
        let (nv, nvu, cap) = (self.cfg.n_vocab, self.cfg.n_vocab_unpadded, self.cfg.final_softcap);
        for r0 in (0..t).step_by(LOGIT_ROWS) {
            let rows = LOGIT_ROWS.min(t - r0);
            self.enqueue_head(lg, r0, rows)?;
            let v = self.gpu.read_f32(lg, rows * nv)?;
            let mut row = vec![0f32; nvu];
            for rr in 0..rows {
                for (o, &z) in row.iter_mut().zip(&v[rr * nv..rr * nv + nvu]) {
                    *o = cap * (z / cap).tanh();
                }
                out(pos0 + r0 + rr, &row);
            }
        }
        Ok(())
    }

    /// Timing study for speculative decoding: milliseconds of one decode step and of one t-token verify pass
    /// (the batched forward pass plus the output head on all t rows, replayed as a CUDA graph) at position `pos`.
    /// The state must already hold `pos` tokens; conv states drift during the measurement (timing only).
    pub fn bench_verify(&mut self, pos: usize, ts: &[usize], reps: usize) -> Result<(f64, Vec<f64>)> {
        ensure!(self.cpu.is_none(), "bench_verify: no CPU offload");
        ensure!(pos + ts.iter().max().copied().unwrap_or(1) <= self.n_ctx, "bench_verify: context too small");
        let lg = self.logit_rows()?;
        // decode: graph replay, one step after another
        for _ in 0..3 {
            self.run(1000, pos, None)?;
        }
        self.gpu.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            self.run(1000, pos, None)?;
        }
        self.gpu.sync()?;
        let dec = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let mut out = Vec::new();
        for &t in ts {
            let toks: Vec<u32> = (0..t as u32).map(|i| 1000 + i).collect();
            self.upload_chunk(&toks, pos)?;
            self.gpu.sync()?;
            self.gpu.stream.begin_capture(inkling_cuda::cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let r = self.enqueue_chunk(t, pos, ChunkOpts::default()).and_then(|_| self.enqueue_head(lg, 0, t));
            let g = self.gpu.stream.end_capture(inkling_cuda::cudarc::driver::sys::CUgraphInstantiate_flags(0))?;
            r?;
            let g = g.expect("graph");
            for _ in 0..3 {
                g.launch()?;
            }
            self.gpu.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                g.launch()?;
            }
            self.gpu.sync()?;
            out.push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
        self.cache = crate::PromptCache::default();
        Ok((dec, out))
    }
}
