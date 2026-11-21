//! InklingEngine forward pass. The decode step (docs/MODEL.md) is recorded once into a CUDA graph and replayed
//! for every token; per-token inputs (token id, position) live in a device buffer written before each replay.

pub mod repack;
pub mod weights;

use anyhow::{Result, ensure};
use inkling_core::{Config, Gguf};
use inkling_cuda::cudarc::driver::{CudaGraph, sys};
use inkling_cuda::{Arg, Ctrl4, DevBuf, ExpW, Gpu, MixArgs, Mv4, MvB4, MvSeg, MvSegB};

pub use weights::{ExpertFmt, LayerW, Weights};

const F32: usize = 4;

/// Per-slot recurrent state: KV cache + short-conv histories.
struct Slot {
    k: DevBuf,
    v: DevBuf,
    kv_cap: usize,
    conv: DevBuf, // [k: 3x1024][v: 3x1024][attn: 3xD][ffn: 3xD] floats
}

/// Scratch activations (all F32).
struct Act {
    x: DevBuf,
    a: DevBuf,     // mix outputs [3][D]
    c: DevBuf,     // control outputs [3][D]
    nrm: DevBuf,   // normalized attention inputs [3][D]
    msg: DevBuf,   // loop messages [4][D]
    g: DevBuf,     // control bottleneck [4][R]
    qrg: DevBuf,   // q (3072) | r (384) | gate (3072)
    sink: DevBuf,  // 24 | dynk 32 | dynv 32
    kraw: DevBuf,
    vraw: DevBuf,
    k: DevBuf,
    v: DevBuf,
    att: DevBuf,
    proj: DevBuf,
    attn_out: DevBuf,
    f: DevBuf,
    f2: DevBuf,
    fnorm: DevBuf,
    sel: DevBuf,
    wts: DevBuf,
    rlog: DevBuf,
    h: DevBuf,
    y: DevBuf,
    part: DevBuf,  // expert partial outputs [k][D]
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
        // the v3 expert kernels are specialised for Inkling 2 Mini's dimensions
        ensure!(cfg.n_embd == 2304 && cfg.n_ff_exp == 1152 && cfg.n_expert_used <= 8, "unsupported model dimensions");
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
        let act = Act {
            x: f(d)?,
            a: f(3 * d)?,
            c: f(3 * d)?,
            nrm: f(3 * d)?,
            msg: f(4 * d)?,
            g: f(4 * cfg.gate_rank)?,
            qrg: f(2 * qd + cfg.n_head * cfg.n_rel)?,
            sink: f(cfg.n_head + 2 * cfg.n_head_kv * cfg.n_conv)?,
            kraw: f(kvw)?,
            vraw: f(kvw)?,
            k: f(kvw)?,
            v: f(kvw)?,
            att: f(qd)?,
            proj: f(d)?,
            attn_out: f(d)?,
            f: f(d)?,
            f2: f(d)?,
            fnorm: f(d)?,
            sel: f(cfg.n_expert_used)?,
            wts: f(cfg.n_expert_used)?,
            rlog: f(cfg.n_expert)?,
            h: f(cfg.n_expert_used * cfg.n_ff_exp)?,
            y: f(d)?,
            part: f(cfg.n_expert_used * d)?,
            ffn_out: f(d)?,
            xp: f(d)?,
            xprev: f(d)?,
            blocks: f(16 * d)?,
            logits: f(cfg.n_vocab)?,
            step: f(2)?,
        };
        gpu.sync()?;
        Ok(Self { gpu, cfg, w, slots, act, n_ctx, graph: None, use_graph: std::env::var_os("INKLING_NO_GRAPH").is_none() })
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
            "k" => (a.k.ptr, c.n_head_kv * c.head_dim),
            "v" => (a.v.ptr, c.n_head_kv * c.head_dim),
            "nrm" => (a.nrm.ptr, 3 * c.n_embd),
            "att" => (a.att.ptr, qd),
            "attn_out" => (a.attn_out.ptr, c.n_embd),
            "fnorm" => (a.fnorm.ptr, c.n_embd),
            "rlog" => (a.rlog.ptr, c.n_expert),
            "wts" => (a.wts.ptr, c.n_expert_used),
            "y" => (a.y.ptr, c.n_embd),
            "ffn_out" => (a.ffn_out.ptr, c.n_embd),
            "msg" => (a.msg.ptr, 4 * c.n_embd),
            _ => anyhow::bail!("unknown debug buffer {name}"),
        };
        self.gpu.read_f32(ptr, n)
    }

    // ---- launch helpers ------------------------------------------------------------------------------------

    fn mix(&self, states: &[u64], q: u64, nq: usize, out: u64, outn: u64, nw: [u64; 4]) -> Result<()> {
        ensure!(states.len() <= 6 && nq <= 4, "mix: too many states/queries");
        let mut m = MixArgs { q, out, outn, nw, n: states.len() as i32, nq: nq as i32, d: self.cfg.n_embd as i32, eps: self.cfg.rms_eps, ..Default::default() };
        m.st[..states.len()].copy_from_slice(states);
        self.gpu.launch("ink_mix3", (1, 1), 512, &[Arg::Mix(m)])
    }

    fn rmsnorm(&self, x: u64, w: u64, y: u64) -> Result<()> {
        self.gpu.launch("ink_rmsnorm", (1, 1), 256, &[Arg::Ptr(x), Arg::Ptr(w), Arg::Ptr(y), Arg::I32(self.cfg.n_embd as i32), Arg::F32(self.cfg.rms_eps)])
    }

    fn mv_q8(&self, segs: &[(weights::Q8M, u64, u64, usize)], k: usize) -> Result<()> {
        let mut a = Mv4 { nseg: segs.len() as i32, k: k as i32, ..Default::default() };
        let mut rows = 0;
        for (i, &(w, x, y, n)) in segs.iter().enumerate() {
            a.s[i] = MvSeg { qs: w.qs, d: w.d, x, y, n: n as i32 };
            rows += n;
        }
        self.gpu.launch("ink_mv_q8", (rows.div_ceil(8) as u32, 1), 256, &[Arg::Mv(a)])
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

    fn conv(&self, x: u64, w: u64, state: u64, dynamic: u64, out: u64, ch: usize) -> Result<()> {
        self.gpu.launch("ink_conv", (ch.div_ceil(256) as u32, 1), 256, &[
            Arg::Ptr(x), Arg::Ptr(w), Arg::Ptr(state), Arg::Ptr(dynamic), Arg::Ptr(out), Arg::I32(ch as i32), Arg::I32(self.cfg.head_dim as i32),
        ])
    }

    /// Post conv over the sum of `nparts` inputs (D apart), written to x (accumulate) or replacing x.
    fn conv_res(&self, xin: u64, nparts: usize, w: u64, state: u64, x: u64, dbg: u64, accumulate: bool) -> Result<()> {
        let d = self.cfg.n_embd;
        self.gpu.launch("ink_conv_res", (d.div_ceil(256) as u32, 1), 256, &[
            Arg::Ptr(xin), Arg::I32(nparts as i32), Arg::I32(d as i32), Arg::Ptr(w), Arg::Ptr(state), Arg::Ptr(x), Arg::Ptr(dbg),
            Arg::I32(accumulate as i32), Arg::I32(d as i32),
        ])
    }

    fn ew(&self, kernel: &str, args: &[Arg], n: usize) -> Result<()> {
        let mut a = args.to_vec();
        a.push(Arg::I32(n as i32));
        self.gpu.launch(kernel, (n.div_ceil(256) as u32, 1), 256, &a)
    }

    /// Loop controls j in `ctrls`: out_j = control(c_j, m_j) with the layer's weights for control index idx_j.
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

        self.gpu.launch("ink_embed", (1, 1), 256, &[
            Arg::Ptr(self.w.token_embd), Arg::Ptr(a.step.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(a.x.ptr), Arg::I32(d as i32), Arg::F32(cfg.rms_eps),
        ])?;

        let mut blocks: Vec<u64> = Vec::new();
        let mut prelude_blocks: Vec<u64> = Vec::new();
        let mut bank: Vec<u64> = Vec::new();
        let mut n_push = 0usize;
        let row = |j: usize| j * d * F32;

        for p in cfg.schedule() {
            let lw = self.w.layers[p.phys];
            if p.phys == cfg.core_start {
                if p.pass == 0 {
                    self.ew("ink_copy", &[Arg::Ptr(a.x.ptr), Arg::Ptr(a.xp.ptr)], d)?;
                    prelude_blocks = blocks.clone();
                } else {
                    bank = blocks[prelude_blocks.len() + 1..].to_vec();
                    self.ew("ink_copy", &[Arg::Ptr(a.x.ptr), Arg::Ptr(a.xprev.ptr)], d)?;
                    self.ew("ink_copy", &[Arg::Ptr(a.xp.ptr), Arg::Ptr(a.x.ptr)], d)?;
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
                self.mix(&st, lp.unwrap().msg_query, 4, a.msg.ptr, 0, [0; 4])?;
            }

            let boundary = p.phys % cfg.attn_res_block == 0;
            let inp_copy = if boundary {
                let dst = a.blocks.at(n_push * d * F32);
                n_push += 1;
                self.ew("ink_copy", &[Arg::Ptr(a.x.ptr), Arg::Ptr(dst)], d)?;
                Some(dst)
            } else {
                None
            };

            // attention inputs -> normalized nq/nk/nv
            let n = [a.nrm.ptr, a.nrm.at(row(1)), a.nrm.at(row(2))];
            let norms = if lw.is_swa { [lw.attn_norm; 3] } else { [lw.q_norm, lw.k_norm, lw.v_norm] };
            let nq = if lw.is_swa { 1 } else { 3 };
            let (xq, xk, xv);
            if !looped {
                if lw.attn_res_query != 0 {
                    let mut st = blocks.clone();
                    st.push(a.x.ptr);
                    let nw = if lw.is_swa { [norms[0], 0, 0, 0] } else { [norms[0], norms[1], norms[2], 0] };
                    self.mix(&st, lw.attn_res_query, nq, a.a.ptr, a.nrm.ptr, nw)?;
                } else {
                    for j in 0..nq {
                        self.rmsnorm(a.x.ptr, norms[j], n[j])?;
                    }
                }
                (xq, xk, xv) = if lw.is_swa { (n[0], n[0], n[0]) } else { (n[0], n[1], n[2]) };
            } else {
                let mut st = blocks.clone();
                st.push(a.x.ptr);
                self.mix(&st, lw.attn_res_query, nq, a.a.ptr, 0, [0; 4])?;
                let inp = |j: usize| if lw.is_swa { a.a.ptr } else { a.a.at(row(j)) };
                let ctl: Vec<(usize, u64, u64, u64)> = (0..3).map(|j| (j, inp(j), a.msg.at(row(j)), a.c.at(row(j)))).collect();
                self.controls(&lp.unwrap(), &ctl)?;
                for j in 0..3 {
                    self.rmsnorm(a.c.at(row(j)), norms[j], n[j])?;
                }
                (xq, xk, xv) = (n[0], n[1], n[2]);
            }

            // projections: q|r|gate, k, v in one launch; sink (+ dynamic conv taps) in another
            self.mv_q8(&[(lw.wqrg, xq, a.qrg.ptr, 2 * qd + cfg.n_head * cfg.n_rel), (lw.wk, xk, a.kraw.ptr, kvw), (lw.wv, xv, a.vraw.ptr, kvw)], d)?;
            let (dk, dv) = if lw.is_swa {
                self.mv_bf16(&[(lw.wsink, xq, sink_ptr, cfg.n_head)], d)?;
                (0, 0)
            } else {
                self.mv_bf16(&[(lw.wsink, xq, sink_ptr, cfg.n_head), (lw.kconv_proj, xk, dynk_ptr, nkc), (lw.vconv_proj, xv, dynv_ptr, nkc)], d)?;
                (dynk_ptr, dynv_ptr)
            };
            self.conv(a.kraw.ptr, lw.k_conv, conv_k, dk, a.k.ptr, kvw)?;
            self.conv(a.vraw.ptr, lw.v_conv, conv_v, dv, a.v.ptr, kvw)?;
            self.gpu.launch("ink_attn_prep", (64, 1), 32, &[
                Arg::Ptr(q_ptr), Arg::Ptr(a.k.ptr), Arg::Ptr(a.v.ptr), Arg::Ptr(r_ptr), Arg::Ptr(lw.q_clip), Arg::Ptr(lw.rel_clip),
                Arg::Ptr(a.step.ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(!lw.is_swa as i32),
                Arg::F32(cfg.qk_clip), Arg::F32(cfg.rel_clip), Arg::F32(cfg.log_scale_floor), Arg::F32(cfg.log_scale_alpha), Arg::F32(cfg.rms_eps),
            ])?;
            let window = if lw.is_swa { cfg.n_swa as i32 } else { 0 };
            self.gpu.launch("ink_attn_decode", (cfg.n_head as u32, 1), 128, &[
                Arg::Ptr(q_ptr), Arg::Ptr(r_ptr), Arg::Ptr(sink_ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::Ptr(lw.rel_bias),
                Arg::Ptr(a.att.ptr), Arg::Ptr(a.step.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(window), Arg::I32(cfg.rel_extent as i32),
            ])?;
            self.ew("ink_gate_sigmoid", &[Arg::Ptr(a.att.ptr), Arg::Ptr(gate_ptr)], qd)?;
            self.mv_q8(&[(lw.wo, a.att.ptr, a.proj.ptr, d)], qd)?;
            // post conv fused with the residual: block boundary starts a new stream (x = attn), else x += attn
            if let Some(dst) = inp_copy {
                blocks.push(dst);
            }
            self.conv_res(a.proj.ptr, 1, lw.attn_post_conv, conv_attn, a.x.ptr, a.attn_out.ptr, inp_copy.is_none())?;

            // FFN input
            let mut st = blocks.clone();
            st.push(a.x.ptr);
            if looped {
                self.mix(&st, lw.ffn_res_query, 1, a.f.ptr, 0, [0; 4])?;
                self.controls(&lp.unwrap(), &[(3, a.f.ptr, a.msg.at(row(3)), a.f2.ptr)])?;
                self.rmsnorm(a.f2.ptr, lw.ffn_norm, a.fnorm.ptr)?;
            } else {
                self.mix(&st, lw.ffn_res_query, 1, a.f.ptr, a.fnorm.ptr, [lw.ffn_norm, 0, 0, 0])?;
            }

            // MoE
            self.gpu.launch("ink_matvec_f32", ((cfg.n_expert / 8) as u32, 1), 256, &[
                Arg::Ptr(lw.router), Arg::Ptr(a.fnorm.ptr), Arg::Ptr(a.rlog.ptr), Arg::I32(d as i32), Arg::I32(cfg.n_expert as i32),
            ])?;
            self.gpu.launch("ink_topk", (1, 1), 32, &[
                Arg::Ptr(a.rlog.ptr), Arg::Ptr(lw.probs_bias), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr),
                Arg::I32(cfg.n_expert as i32), Arg::I32(cfg.n_expert_used as i32), Arg::F32(cfg.expert_weights_scale),
            ])?;
            let ew = |m: weights::ExpM| ExpW { qs: m.qs, sc: m.sc, scale: m.scale, fmt: m.fmt as i32 };
            let k = cfg.n_expert_used;
            self.gpu.launch("ink_moe_gate_up5", ((cfg.n_ff_exp / 32) as u32, k as u32), 256, &[
                Arg::Exp(ew(lw.gate_up)), Arg::Ptr(a.fnorm.ptr), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.h.ptr),
                Arg::I32(cfg.n_ff_exp as i32), Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
            ])?;
            self.gpu.launch("ink_moe_down5", ((d / 32) as u32, k as u32), 256, &[
                Arg::Exp(ew(lw.down)), Arg::Ptr(a.h.ptr), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr), Arg::Ptr(a.part.ptr), Arg::I32(d as i32),
            ])?;
            // sum of the k expert partials -> post conv -> x += ffn
            self.conv_res(a.part.ptr, k, lw.ffn_post_conv, conv_ffn, a.x.ptr, a.ffn_out.ptr, true)?;

            if let Some(cb) = capture.as_mut() {
                cb(p.slot, self);
            }
        }

        // output head
        let mut st = blocks.clone();
        st.push(a.x.ptr);
        self.mix(&st, self.w.output_res_query, 1, a.f.ptr, a.fnorm.ptr, [self.w.output_norm, 0, 0, 0])?;
        self.mv_q8(&[(self.w.output, a.fnorm.ptr, a.logits.ptr, cfg.n_vocab)], d)?;
        let nv = cfg.n_vocab_unpadded;
        self.gpu.launch("ink_softcap", (nv.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(a.logits.ptr), Arg::I32(nv as i32), Arg::F32(cfg.final_softcap)])
    }

    /// Run one token at position `pos`; returns the logits over the unpadded vocabulary.
    pub fn step(&mut self, token: u32, pos: usize, capture: Option<Capture>) -> Result<Vec<f32>> {
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
        self.gpu.read_f32(self.act.logits.ptr, self.cfg.n_vocab_unpadded)
    }
}
