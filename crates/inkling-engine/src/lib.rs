//! InklingEngine forward pass. The decode step (docs/MODEL.md) is recorded once into a CUDA graph and replayed
//! for every token; per-token inputs (token id, position) live in a device buffer written before each replay.
//!
//! Activations feeding the quantized matvecs are int8 (q8_1 blocks of 32, as llama.cpp's mmvq), produced by the
//! kernel that computes them (depth-attention mix, norms, attention output, expert gate_up).

pub mod repack;
pub mod weights;

use anyhow::{Result, ensure};
use inkling_core::{Config, Gguf};
use inkling_cuda::cudarc::driver::{CudaGraph, sys};
use inkling_cuda::{Arg, Ctrl4, DevBuf, ExpW, Gpu, MixArgs, MixArgsQ, MvB4, MvQ4, MvSegB, MvSegQ, NormQ4};

pub use weights::{ExpertFmt, LayerW, Weights};

const F32: usize = 4;

/// Per-slot recurrent state: KV cache + short-conv histories.
struct Slot {
    k: DevBuf,
    v: DevBuf,
    kv_cap: usize,
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
    hq: Q8Buf,      // q8_1 expert hidden [k][F]
    sel: DevBuf,
    wts: DevBuf,
    rlog: DevBuf,
    counter: DevBuf,
    part: DevBuf,   // expert partial outputs [k][D]
    ffn_out: DevBuf,
    xp: DevBuf,
    xprev: DevBuf,
    blocks: DevBuf, // block storage [16][D], indexed by push order within a token
    logits: DevBuf,
    step: DevBuf,   // i32 [token, pos]
}

pub struct Engine {
    pub gpu: Gpu,
    pub cfg: Config,
    pub w: Weights,
    slots: Vec<Slot>,
    act: Act,
    pub n_ctx: usize,
    graph: Option<CudaGraph>,
    /// disable graph replay (debug captures need per-layer host callbacks)
    pub use_graph: bool,
}

/// Optional per-pass hook, called after each layer pass with (slot, engine); use `Engine::debug_buf` to read state.
pub type Capture<'a> = &'a mut dyn FnMut(usize, &Engine);

impl Engine {
    pub fn load(model: &str, n_ctx: usize) -> Result<Self> {
        let g = Gguf::open(model)?;
        let cfg = Config::from_gguf(&g)?;
        // several kernels are specialised for Inkling 2 Mini's dimensions
        ensure!(
            cfg.n_embd == 2304 && cfg.n_ff_exp == 1152 && cfg.n_expert_used <= 8 && cfg.n_expert <= 128 && cfg.head_dim == 128
                && cfg.n_head == 24 && cfg.n_head_kv == 8 && cfg.n_rel == 16,
            "unsupported model dimensions"
        );
        let gpu = Gpu::new(0)?;
        let w = Weights::load(&gpu, &g, &cfg)?;
        let d = cfg.n_embd;
        let kvw = cfg.n_head_kv * cfg.head_dim;
        let mut slot_phys = vec![0; cfg.n_slots()];
        for p in cfg.schedule() {
            slot_phys[p.slot] = p.phys;
        }
        let mut slots = Vec::with_capacity(cfg.n_slots());
        for s in 0..cfg.n_slots() {
            let cap = if cfg.is_swa[slot_phys[s]] { cfg.n_swa.min(n_ctx) } else { n_ctx };
            slots.push(Slot {
                k: gpu.alloc(cap * kvw * 2)?,
                v: gpu.alloc(cap * kvw * 2)?,
                kv_cap: cap,
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
            hq: Q8Buf::new(&gpu, k * cfg.n_ff_exp)?,
            sel: f(k)?,
            wts: f(k)?,
            rlog: f(cfg.n_expert)?,
            counter: f(1)?,
            part: f(k * d)?,
            ffn_out: f(d)?,
            xp: f(d)?,
            xprev: f(d)?,
            blocks: f(16 * d)?,
            logits: f(cfg.n_vocab)?,
            step: f(2)?,
        };
        gpu.sync()?;
        let use_graph = std::env::var_os("INKLING_NO_GRAPH").is_none();
        Ok(Self { gpu, cfg, w, slots, act, n_ctx, graph: None, use_graph })
    }

    /// Zero all KV caches and conv states (start a new sequence).
    pub fn reset(&mut self) -> Result<()> {
        for s in &self.slots {
            for b in [&s.k, &s.v, &s.conv] {
                unsafe { inkling_cuda::cudarc::driver::result::memset_d8_async(b.ptr, 0, b.bytes, self.gpu.stream.cu_stream()) }?;
            }
        }
        Ok(())
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
        self.gpu.launch("ink_mix4", (1, 1), 512, &[Arg::MixQ(MixArgsQ { m, xq: xq.0, xd: xq.1 })])
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
        self.gpu.launch("ink_rmsnorm_q", (items.len() as u32, 1), 256, &[Arg::NormQ(a), Arg::I32(self.cfg.n_embd as i32), Arg::F32(self.cfg.rms_eps)])
    }

    /// Q8 matvecs on q8_1 activations: segments (weights, (xq, xd), y, rows), all with K inputs.
    fn mv(&self, segs: &[(weights::Q8M, (u64, u64), u64, usize)], k: usize) -> Result<()> {
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
        self.gpu.launch("ink_ctrl_down", ((r / 8) as u32, n), 256, &[Arg::Ctrl(a), Arg::I32(d), Arg::I32(r as i32)])?;
        self.gpu.launch("ink_ctrl_up", ((self.cfg.n_embd / 8) as u32, n), 256, &[Arg::Ctrl(a), Arg::I32(d), Arg::I32(r as i32)])
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

        self.gpu.launch("ink_embed", (1, 1), 256, &[
            Arg::Ptr(self.w.token_embd), Arg::Ptr(a.step.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(a.x.ptr), Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
        ])?;

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
            self.mv(&[
                (lw.wqrg, nqv(iq), a.qrg.ptr, 2 * qd + cfg.n_head * cfg.n_rel),
                (lw.wk, nqv(ik), a.kraw.ptr, kvw),
                (lw.wv, nqv(iv), a.vraw.ptr, kvw),
            ], d)?;
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
            ])?;
            let window = if lw.is_swa { cfg.n_swa as i32 } else { 0 };
            let (oq, od) = a.attq.at(0, qd);
            self.gpu.launch("ink_attn_decode2", (cfg.n_head as u32, 1), 128, &[
                Arg::Ptr(q_ptr), Arg::Ptr(r_ptr), Arg::Ptr(sink_ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::Ptr(lw.rel_bias),
                Arg::Ptr(gate_ptr), Arg::Ptr(a.att.ptr), Arg::Ptr(oq), Arg::Ptr(od),
                Arg::Ptr(a.step.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(window), Arg::I32(cfg.rel_extent as i32),
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
            self.gpu.launch("ink_router2", ((cfg.n_expert / 8) as u32, 1), 256, &[
                Arg::Ptr(a.fnorm.ptr), Arg::Ptr(lw.router), Arg::Ptr(lw.probs_bias), Arg::Ptr(a.rlog.ptr), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr),
                Arg::Ptr(a.counter.ptr), Arg::I32(d as i32), Arg::I32(cfg.n_expert as i32), Arg::I32(k as i32), Arg::F32(cfg.expert_weights_scale),
            ])?;
            let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt as i32 };
            let (hq, hd) = a.hq.at(0, k * cfg.n_ff_exp);
            self.gpu.launch("ink_moe_gate_up6", ((cfg.n_ff_exp / 32) as u32, k as u32), 256, &[
                Arg::Exp(ew(lw.gate_up)), Arg::Ptr(fq.0), Arg::Ptr(fq.1), Arg::Ptr(a.sel.ptr), Arg::Ptr(hq), Arg::Ptr(hd),
                Arg::I32(cfg.n_ff_exp as i32), Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
            ])?;
            self.gpu.launch("ink_moe_down6", ((d / 32) as u32, k as u32), 256, &[
                Arg::Exp(ew(lw.down)), Arg::Ptr(hq), Arg::Ptr(hd), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr), Arg::Ptr(a.part.ptr), Arg::I32(d as i32),
            ])?;
            self.conv_res(a.part.ptr, k, lw.ffn_post_conv, conv_ffn, a.x.ptr, a.ffn_out.ptr, true)?;

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
        self.run(token, pos, capture)?;
        self.gpu.read_f32(self.act.logits.ptr, self.cfg.n_vocab_unpadded)
    }

    /// Enqueue one token step (graph replay when possible); results stay on the device.
    fn run(&mut self, token: u32, pos: usize, capture: Option<Capture>) -> Result<()> {
        ensure!(pos < self.n_ctx, "position {pos} exceeds context {}", self.n_ctx);
        self.gpu.write(self.act.step.ptr, inkling_cuda::as_bytes(&[token as i32, pos as i32]))?;
        if capture.is_some() || !self.use_graph {
            self.enqueue_step(capture)?;
        } else {
            if self.graph.is_none() {
                self.gpu.sync()?;
                self.gpu.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let r = self.enqueue_step(None);
                let g = self.gpu.stream.end_capture(sys::CUgraphInstantiate_flags(0))?;
                r?;
                self.graph = g;
            }
            self.graph.as_ref().expect("graph").launch()?;
        }
        Ok(())
    }
}
