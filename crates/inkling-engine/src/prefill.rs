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
    tokens: DevBuf,
    step: DevBuf,
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
}

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

    fn pre_norm(&self, items: &[(u64, u64, u64, (u64, u64))], t: usize) -> Result<()> {
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
    fn pre_gemm(&self, segs: &[(weights::Q8M, (u64, u64), u64, usize, usize)], k: usize, t: usize) -> Result<()> {
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

    fn pre_conv_res(&self, xin: u64, w: u64, state: u64, x: u64, accumulate: bool, t: usize) -> Result<()> {
        let d = self.cfg.n_embd;
        self.gpu.launch("ink_conv_res_b", (d.div_ceil(256) as u32, t as u32), 256, &[
            Arg::Ptr(xin), Arg::Ptr(w), Arg::Ptr(state), Arg::Ptr(x), Arg::I32(accumulate as i32), Arg::I32(d as i32),
        ])?;
        self.conv_state(xin, state, d, t)
    }

    fn conv_state(&self, x: u64, state: u64, ch: usize, t: usize) -> Result<()> {
        self.gpu.launch("ink_conv_state_b", (ch.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(x), Arg::I32(t as i32), Arg::Ptr(state), Arg::I32(ch as i32)])
    }

    fn copy_n(&self, src: u64, dst: u64, bytes: usize) -> Result<()> {
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
        let p = self.pre.as_ref().expect("prefill buffers");
        let cfg = &self.cfg;
        let (d, kvw, qd, r) = (cfg.n_embd, cfg.n_head_kv * cfg.head_dim, cfg.n_head * cfg.head_dim, cfg.gate_rank);
        let qs = 2 * qd + cfg.n_head * cfg.n_rel; // qrg row
        let ss = cfg.n_head + 2 * cfg.n_head_kv * cfg.n_conv; // sink row
        let td = t * d;
        let row = |j: usize| j * td * F32; // j-th [T][D] slab
        let nqv = |j: usize| p.nq.at(j * td);

        let ids: Vec<i32> = tokens.iter().map(|&x| if x >= crate::EMBED_TOKEN_BASE { 0 } else { x as i32 }).collect();
        self.gpu.write(p.tokens.ptr, inkling_cuda::as_bytes(&ids))?;
        self.gpu.write(p.step.ptr, inkling_cuda::as_bytes(&[0i32, pos0 as i32, 0, 0]))?;
        self.gpu.launch("ink_embed_b", (t as u32, 1), 256, &[
            Arg::Ptr(self.w.token_embd), Arg::Ptr(p.tokens.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(p.x.ptr), Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
        ])?;
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

        for ps in cfg.schedule() {
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
            let slot = &self.slots[ps.slot];
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
            self.pre_gemm(&[
                (lw.wqrg, nqv(iq), p.qrg.ptr, qs, qs),
                (lw.wk, nqv(ik), p.kraw.ptr, kvw, kvw),
                (lw.wv, nqv(iv), p.vraw.ptr, kvw, kvw),
            ], d, t)?;
            let nkc = cfg.n_head_kv * cfg.n_conv;
            let mut mb = MvB4 { k: d as i32, ..Default::default() };
            mb.s[0] = MvSegB { w: lw.wsink, x: n[iq], y: p.sink.ptr, n: cfg.n_head as i32 };
            mb.nseg = 1;
            let mut rows = cfg.n_head;
            if !lw.is_swa {
                mb.s[1] = MvSegB { w: lw.kconv_proj, x: n[ik], y: p.sink.at(cfg.n_head * F32), n: nkc as i32 };
                mb.s[2] = MvSegB { w: lw.vconv_proj, x: n[iv], y: p.sink.at((cfg.n_head + nkc) * F32), n: nkc as i32 };
                mb.nseg = 3;
                rows += 2 * nkc;
            }
            self.gpu.launch("ink_mv_bf16b", (rows.div_ceil(8) as u32, t as u32), 256, &[Arg::MvB(mb), Arg::I32(ss as i32)])?;
            // K/V convs + clip/norm + cache write, q/r clip + tau
            self.gpu.launch("ink_attn_prep_b", (64, t as u32), 128, &[
                Arg::Ptr(p.qrg.ptr), Arg::I32(qs as i32), Arg::I32(qd as i32), Arg::Ptr(p.kraw.ptr), Arg::Ptr(p.vraw.ptr),
                Arg::Ptr(lw.q_clip), Arg::Ptr(lw.rel_clip), Arg::Ptr(lw.k_conv), Arg::Ptr(lw.v_conv), Arg::Ptr(conv_k), Arg::Ptr(conv_v),
                Arg::Ptr(p.sink.at(cfg.n_head * F32)), Arg::I32(ss as i32), Arg::I32(!lw.is_swa as i32), Arg::Ptr(p.step.ptr),
                Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(!lw.is_swa as i32),
                Arg::F32(cfg.qk_clip), Arg::F32(cfg.rel_clip), Arg::F32(cfg.log_scale_floor), Arg::F32(cfg.log_scale_alpha), Arg::F32(cfg.rms_eps),
                Arg::I32(self.kv_bits(lw.is_swa).0), Arg::I32(self.kv_bits(lw.is_swa).1), Arg::I32(slot.kv_fmt),
            ])?;
            self.conv_state(p.kraw.ptr, conv_k, kvw, t)?;
            self.conv_state(p.vraw.ptr, conv_v, kvw, t)?;
            let window = if lw.is_swa { cfg.n_swa as i32 } else { 0 };
            let (oq, od) = p.attq.at(0);
let pre_attn = std::env::var("INKLING_PRE_ATTN").ok();
            if let Some(k @ ("split" | "mma")) = pre_attn.as_deref() {
                            self.gpu.launch3(if k == "mma" { "ink_attn_mma" } else { "ink_attn_split" }, (cfg.n_head_kv as u32, PRE_SPLITS as u32, t as u32), 128, &[
                Arg::Ptr(p.qrg.ptr), Arg::Ptr(p.qrg.at(qd * F32)), Arg::Ptr(p.sink.ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr),
                Arg::Ptr(lw.rel_bias), Arg::Ptr(p.qrg.at((qd + cfg.n_head * cfg.n_rel) * F32)), Arg::Ptr(p.att.ptr), Arg::Ptr(oq), Arg::Ptr(od),
                Arg::Ptr(p.step.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(window), Arg::I32(cfg.rel_extent as i32),
                Arg::Ptr(p.part.ptr), Arg::Ptr(p.cnt.ptr), Arg::I32(qs as i32), Arg::I32(ss as i32), Arg::I32(slot.kv_fmt),
            ])?;
            } else {
                self.gpu.launch("ink_attn_prefill", (t.div_ceil(64) as u32, cfg.n_head as u32), 128, &[
                    Arg::Ptr(p.qrg.ptr), Arg::I32(qs as i32), Arg::I32(qd as i32), Arg::I32((qd + cfg.n_head * cfg.n_rel) as i32),
                    Arg::Ptr(p.sink.ptr), Arg::I32(ss as i32), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32),
                    Arg::I32(window), Arg::I32(cfg.rel_extent as i32), Arg::Ptr(lw.rel_bias), Arg::Ptr(p.step.ptr), Arg::I32(t as i32),
                    Arg::Ptr(p.att.ptr), Arg::Ptr(oq), Arg::Ptr(od), Arg::I32(slot.kv_fmt),
                ])?;
            }
            self.pre_gemm(&[(lw.wo, (oq, od), p.proj.ptr, d, d)], qd, t)?;
            if let Some(dst) = inp_copy {
                blocks.push(dst);
            }
            self.pre_conv_res(p.proj.ptr, lw.attn_post_conv, conv_attn, p.x.ptr, inp_copy.is_none(), t)?;

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
            self.sgemm(p.fnorm.ptr, d, lw.router, d, false, p.rlog.ptr, cfg.n_expert, 0, false, t, cfg.n_expert, d)?;
            self.gpu.launch("ink_topk_b", (t as u32, 1), 32, &[
                Arg::Ptr(p.rlog.ptr), Arg::Ptr(lw.probs_bias), Arg::Ptr(p.sel.ptr), Arg::Ptr(p.wts.ptr), Arg::I32(cfg.n_expert as i32),
                Arg::I32(k as i32), Arg::F32(cfg.expert_weights_scale),
            ])?;
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
            self.pre_conv_res(p.ysum.ptr, lw.ffn_post_conv, conv_ffn, p.x.ptr, true, t)?;
        }
        let _ = r;

        // output head (validation path): logits of every position
        if let Some(out) = logits_out {
            let mut st = blocks.clone();
            st.push(p.x.ptr);
            let fq = p.fq.at(0);
            self.pre_mix(&st, self.w.output_res_query, 1, p.f.ptr, p.fnorm.ptr, [self.w.output_norm, 0, 0, 0], fq, t)?;
            if self.pre.as_ref().unwrap().logits.is_none() {
                let b = self.gpu.alloc(LOGIT_ROWS * self.cfg.n_vocab * F32)?;
                self.pre.as_mut().unwrap().logits = Some(b);
            }
            let p = self.pre.as_ref().unwrap();
            let lg = p.logits.as_ref().unwrap();
            let (nv, nvu, cap) = (self.cfg.n_vocab, self.cfg.n_vocab_unpadded, self.cfg.final_softcap);
            for r0 in (0..t).step_by(LOGIT_ROWS) {
                let rows = LOGIT_ROWS.min(t - r0);
                let (xq, xd) = p.fq.at(r0 * d);
                self.pre_gemm(&[(self.w.output, (xq, xd), lg.ptr, nv, nv)], d, rows)?;
                let v = self.gpu.read_f32(lg.ptr, rows * nv)?;
                let mut row = vec![0f32; nvu];
                for rr in 0..rows {
                    for (o, &z) in row.iter_mut().zip(&v[rr * nv..rr * nv + nvu]) {
                        *o = cap * (z / cap).tanh();
                    }
                    out(pos0 + r0 + rr, &row);
                }
            }
        }
        Ok(())
    }
}
