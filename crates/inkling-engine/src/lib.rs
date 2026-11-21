//! InklingEngine forward pass (M3 reference implementation: one kernel per spec stage, see docs/MODEL.md).

pub mod weights;

use anyhow::{Result, ensure};
use inkling_core::{Config, Gguf};
use inkling_cuda::{Arg, DevBuf, Gpu, Ptrs8};

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
    q: DevBuf,     // 3072
    r: DevBuf,     // 384
    gate: DevBuf,  // 3072
    sink: DevBuf,  // 24
    kraw: DevBuf,  // 1024
    vraw: DevBuf,  // 1024
    k: DevBuf,
    v: DevBuf,
    dynk: DevBuf,  // 32
    dynv: DevBuf,
    att: DevBuf,   // 3072
    proj: DevBuf,  // D
    attn_out: DevBuf,
    f: DevBuf,
    f2: DevBuf,
    fnorm: DevBuf,
    sel: DevBuf,   // i32 [8]
    wts: DevBuf,   // [8]
    rlog: DevBuf,  // router logits [E]
    h: DevBuf,     // [8][F]
    y: DevBuf,     // D
    ffn_out: DevBuf,
    xp: DevBuf,    // prelude output
    xprev: DevBuf, // previous core pass output
    blocks: DevBuf, // block storage [16][D], indexed by push order within a token
    logits: DevBuf,
    step: DevBuf,  // i32 [token, pos]
}

pub struct Engine {
    pub gpu: Gpu,
    pub cfg: Config,
    pub w: Weights,
    slots: Vec<Slot>,
    act: Act,
    pub n_ctx: usize,
}

/// Optional per-pass hook, called after each layer pass with (slot, engine); use `Engine::debug_buf` to read state.
pub type Capture<'a> = &'a mut dyn FnMut(usize, &Engine);

impl Engine {
    pub fn load(model: &str, n_ctx: usize) -> Result<Self> {
        let g = Gguf::open(model)?;
        let cfg = Config::from_gguf(&g)?;
        let gpu = Gpu::new(0)?;
        let w = Weights::load(&gpu, &g, &cfg)?;
        let d = cfg.n_embd;
        let kvw = cfg.n_head_kv * cfg.head_dim;
        let mut slots = Vec::with_capacity(cfg.n_slots());
        let sched = cfg.schedule();
        let mut slot_phys = vec![0; cfg.n_slots()];
        for p in &sched {
            slot_phys[p.slot] = p.phys;
        }
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
            q: f(qd)?,
            r: f(cfg.n_head * cfg.n_rel)?,
            gate: f(qd)?,
            sink: f(cfg.n_head)?,
            kraw: f(kvw)?,
            vraw: f(kvw)?,
            k: f(kvw)?,
            v: f(kvw)?,
            dynk: f(cfg.n_head_kv * cfg.n_conv)?,
            dynv: f(cfg.n_head_kv * cfg.n_conv)?,
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
            ffn_out: f(d)?,
            xp: f(d)?,
            xprev: f(d)?,
            blocks: f(16 * d)?,
            logits: f(cfg.n_vocab)?,
            step: f(2)?,
        };
        gpu.sync()?;
        Ok(Self { gpu, cfg, w, slots, act, n_ctx })
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

    fn mix(&self, states: &[u64], q: u64, nq: usize, out: u64) -> Result<()> {
        ensure!(states.len() <= 8 && nq <= 4, "mix: too many states/queries");
        let mut p = Ptrs8::default();
        p.0[..states.len()].copy_from_slice(states);
        self.gpu.launch("ink_mix", (1, 1), 256, &[
            Arg::Ptrs(p), Arg::I32(states.len() as i32), Arg::Ptr(q), Arg::I32(nq as i32), Arg::Ptr(out), Arg::I32(self.cfg.n_embd as i32),
        ])
    }

    fn rmsnorm(&self, x: u64, w: u64, y: u64) -> Result<()> {
        self.gpu.launch("ink_rmsnorm", (1, 1), 256, &[Arg::Ptr(x), Arg::Ptr(w), Arg::Ptr(y), Arg::I32(self.cfg.n_embd as i32), Arg::F32(self.cfg.rms_eps)])
    }

    fn mv(&self, kernel: &str, w: u64, x: u64, y: u64, k: usize, n: usize) -> Result<()> {
        self.gpu.launch(kernel, (n.div_ceil(8) as u32, 1), 256, &[Arg::Ptr(w), Arg::Ptr(x), Arg::Ptr(y), Arg::I32(k as i32), Arg::I32(n as i32)])
    }

    fn conv(&self, x: u64, w: u64, state: u64, dynamic: u64, out: u64, ch: usize) -> Result<()> {
        self.gpu.launch("ink_conv", (ch.div_ceil(256) as u32, 1), 256, &[
            Arg::Ptr(x), Arg::Ptr(w), Arg::Ptr(state), Arg::Ptr(dynamic), Arg::Ptr(out), Arg::I32(ch as i32), Arg::I32(self.cfg.head_dim as i32),
        ])
    }

    fn ew(&self, kernel: &str, args: &[Arg], n: usize) -> Result<()> {
        let mut a = args.to_vec();
        a.push(Arg::I32(n as i32));
        self.gpu.launch(kernel, (n.div_ceil(256) as u32, 1), 256, &a)
    }

    fn control(&self, c: u64, m: u64, lp: &weights::LoopW, ctrl: usize, out: u64) -> Result<()> {
        self.gpu.launch("ink_control", (1, 1), 1024, &[
            Arg::Ptr(c), Arg::Ptr(m), Arg::Ptr(lp.down[ctrl]), Arg::Ptr(lp.gain[ctrl]), Arg::Ptr(lp.forget[ctrl]), Arg::Ptr(out),
            Arg::I32(self.cfg.n_embd as i32), Arg::I32(self.cfg.gate_rank as i32),
        ])
    }

    /// Read a scratch buffer by name (debugging / validation).
    pub fn debug_buf(&self, name: &str) -> Result<Vec<f32>> {
        let a = &self.act;
        let c = &self.cfg;
        let (b, n) = match name {
            "x" => (&a.x, c.n_embd),
            "q" => (&a.q, c.n_head * c.head_dim),
            "k" => (&a.k, c.n_head_kv * c.head_dim),
            "v" => (&a.v, c.n_head_kv * c.head_dim),
            "r" => (&a.r, c.n_head * c.n_rel),
            "nrm" => (&a.nrm, 3 * c.n_embd),
            "a" => (&a.a, 3 * c.n_embd),
            "att" => (&a.att, c.n_head * c.head_dim),
            "attn_out" => (&a.attn_out, c.n_embd),
            "f" => (&a.f, c.n_embd),
            "fnorm" => (&a.fnorm, c.n_embd),
            "rlog" => (&a.rlog, c.n_expert),
            "wts" => (&a.wts, c.n_expert_used),
            "y" => (&a.y, c.n_embd),
            "ffn_out" => (&a.ffn_out, c.n_embd),
            "msg" => (&a.msg, 4 * c.n_embd),
            "sink" => (&a.sink, c.n_head),
            _ => anyhow::bail!("unknown debug buffer {name}"),
        };
        self.gpu.read_f32(b.ptr, n)
    }

    /// Run one token at position `pos`; returns the logits over the unpadded vocabulary.
    pub fn step(&mut self, token: u32, pos: usize, mut capture: Option<Capture>) -> Result<Vec<f32>> {
        ensure!(pos < self.n_ctx, "position {pos} exceeds context {}", self.n_ctx);
        let cfg = &self.cfg;
        let d = cfg.n_embd;
        let a = &self.act;
        self.gpu.write(a.step.ptr, inkling_cuda::as_bytes(&[token as i32, pos as i32]))?;
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
            let conv_k = slot.conv.ptr;
            let kvw = cfg.n_head_kv * cfg.head_dim;
            let conv_v = slot.conv.at(3 * kvw * F32);
            let conv_attn = slot.conv.at(6 * kvw * F32);
            let conv_ffn = slot.conv.at((6 * kvw + 3 * d) * F32);

            // loop messages from the previous pass
            let looped = p.pass > 0;
            if looped {
                let mut st = bank.clone();
                st.push(a.xprev.ptr);
                self.mix(&st, lw.lp.unwrap().msg_query, 4, a.msg.ptr)?;
            }

            // a block boundary stores the layer input before it is overwritten
            let boundary = p.phys % cfg.attn_res_block == 0;
            let inp_copy = if boundary {
                let dst = a.blocks.at(n_push * d * F32);
                n_push += 1;
                self.ew("ink_copy", &[Arg::Ptr(a.x.ptr), Arg::Ptr(dst)], d)?;
                Some(dst)
            } else {
                None
            };

            // attention inputs
            let nq = if lw.is_swa { 1 } else { 3 };
            let mut inp = [a.x.ptr; 3];
            if lw.attn_res_query != 0 {
                let mut st = blocks.clone();
                st.push(a.x.ptr);
                self.mix(&st, lw.attn_res_query, nq, a.a.ptr)?;
                for j in 0..3 {
                    inp[j] = a.a.at(if lw.is_swa { 0 } else { j * d * F32 });
                }
            }
            if looped {
                let lp = lw.lp.unwrap();
                for j in 0..3 {
                    self.control(inp[j], a.msg.at(j * d * F32), &lp, j, a.c.at(j * d * F32))?;
                    inp[j] = a.c.at(j * d * F32);
                }
            }
            let n = [a.nrm.ptr, a.nrm.at(d * F32), a.nrm.at(2 * d * F32)];
            let (nq_, nk, nv);
            if lw.is_swa && !looped {
                self.rmsnorm(inp[0], lw.attn_norm, n[0])?;
                (nq_, nk, nv) = (n[0], n[0], n[0]);
            } else if lw.is_swa {
                for j in 0..3 {
                    self.rmsnorm(inp[j], lw.attn_norm, n[j])?;
                }
                (nq_, nk, nv) = (n[0], n[1], n[2]);
            } else {
                self.rmsnorm(inp[0], lw.q_norm, n[0])?;
                self.rmsnorm(inp[1], lw.k_norm, n[1])?;
                self.rmsnorm(inp[2], lw.v_norm, n[2])?;
                (nq_, nk, nv) = (n[0], n[1], n[2]);
            }

            // projections
            let qd = cfg.n_head * cfg.head_dim;
            self.mv("ink_matvec_q8", lw.wq, nq_, a.q.ptr, d, qd)?;
            self.mv("ink_matvec_q8", lw.wrel, nq_, a.r.ptr, d, cfg.n_head * cfg.n_rel)?;
            self.mv("ink_matvec_q8", lw.wgate, nq_, a.gate.ptr, d, qd)?;
            self.mv("ink_matvec_bf16", lw.wsink, nq_, a.sink.ptr, d, cfg.n_head)?;
            self.mv("ink_matvec_q8", lw.wk, nk, a.kraw.ptr, d, kvw)?;
            self.mv("ink_matvec_q8", lw.wv, nv, a.vraw.ptr, d, kvw)?;
            let (dk, dv) = if lw.is_swa {
                (0, 0)
            } else {
                self.mv("ink_matvec_bf16", lw.kconv_proj, nk, a.dynk.ptr, d, cfg.n_head_kv * cfg.n_conv)?;
                self.mv("ink_matvec_bf16", lw.vconv_proj, nv, a.dynv.ptr, d, cfg.n_head_kv * cfg.n_conv)?;
                (a.dynk.ptr, a.dynv.ptr)
            };
            self.conv(a.kraw.ptr, lw.k_conv, conv_k, dk, a.k.ptr, kvw)?;
            self.conv(a.vraw.ptr, lw.v_conv, conv_v, dv, a.v.ptr, kvw)?;
            self.gpu.launch("ink_attn_prep", (64, 1), 32, &[
                Arg::Ptr(a.q.ptr), Arg::Ptr(a.k.ptr), Arg::Ptr(a.v.ptr), Arg::Ptr(a.r.ptr), Arg::Ptr(lw.q_clip), Arg::Ptr(lw.rel_clip),
                Arg::Ptr(a.step.ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(!lw.is_swa as i32),
                Arg::F32(cfg.qk_clip), Arg::F32(cfg.rel_clip), Arg::F32(cfg.log_scale_floor), Arg::F32(cfg.log_scale_alpha), Arg::F32(cfg.rms_eps),
            ])?;
            let window = if lw.is_swa { cfg.n_swa as i32 } else { 0 };
            self.gpu.launch("ink_attn_decode", (cfg.n_head as u32, 1), 128, &[
                Arg::Ptr(a.q.ptr), Arg::Ptr(a.r.ptr), Arg::Ptr(a.sink.ptr), Arg::Ptr(slot.k.ptr), Arg::Ptr(slot.v.ptr), Arg::Ptr(lw.rel_bias),
                Arg::Ptr(a.att.ptr), Arg::Ptr(a.step.ptr), Arg::I32(slot.kv_cap as i32), Arg::I32(window), Arg::I32(cfg.rel_extent as i32),
            ])?;
            self.ew("ink_gate_sigmoid", &[Arg::Ptr(a.att.ptr), Arg::Ptr(a.gate.ptr)], qd)?;
            self.mv("ink_matvec_q8", lw.wo, a.att.ptr, a.proj.ptr, qd, d)?;
            self.conv(a.proj.ptr, lw.attn_post_conv, conv_attn, 0, a.attn_out.ptr, d)?;

            // residual stream / block bookkeeping
            if let Some(dst) = inp_copy {
                blocks.push(dst);
                self.ew("ink_copy", &[Arg::Ptr(a.attn_out.ptr), Arg::Ptr(a.x.ptr)], d)?;
            } else {
                self.ew("ink_add", &[Arg::Ptr(a.x.ptr), Arg::Ptr(a.attn_out.ptr), Arg::Ptr(a.x.ptr)], d)?;
            }

            // FFN input: depth attention (+ loop control), norm
            let mut st = blocks.clone();
            st.push(a.x.ptr);
            self.mix(&st, lw.ffn_res_query, 1, a.f.ptr)?;
            let mut fin = a.f.ptr;
            if looped {
                self.control(a.f.ptr, a.msg.at(3 * d * F32), &lw.lp.unwrap(), 3, a.f2.ptr)?;
                fin = a.f2.ptr;
            }
            self.rmsnorm(fin, lw.ffn_norm, a.fnorm.ptr)?;

            // MoE
            self.gpu.launch("ink_router", (1, 1), 1024, &[
                Arg::Ptr(a.fnorm.ptr), Arg::Ptr(lw.router), Arg::Ptr(lw.probs_bias), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr), Arg::Ptr(a.rlog.ptr),
                Arg::I32(d as i32), Arg::I32(cfg.n_expert as i32), Arg::I32(cfg.n_expert_used as i32), Arg::F32(cfg.expert_weights_scale),
            ])?;
            let fmt = lw.expert_fmt as i32;
            self.gpu.launch("ink_moe_gate_up", ((cfg.n_ff_exp / 8) as u32, cfg.n_expert_used as u32), 256, &[
                Arg::Ptr(lw.gate_up), Arg::I32(fmt), Arg::Ptr(lw.gate_up_scale), Arg::Ptr(a.fnorm.ptr), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.h.ptr),
                Arg::I32(d as i32), Arg::I32(cfg.n_ff_exp as i32), Arg::F32(cfg.situ_beta), Arg::F32(cfg.situ_linear_beta),
            ])?;
            self.gpu.launch("ink_moe_down", ((d / 8) as u32, 1), 256, &[
                Arg::Ptr(lw.down), Arg::I32(fmt), Arg::Ptr(lw.down_scale), Arg::Ptr(a.h.ptr), Arg::Ptr(a.sel.ptr), Arg::Ptr(a.wts.ptr), Arg::Ptr(a.y.ptr),
                Arg::I32(cfg.n_ff_exp as i32), Arg::I32(d as i32), Arg::I32(cfg.n_expert_used as i32),
            ])?;
            self.conv(a.y.ptr, lw.ffn_post_conv, conv_ffn, 0, a.ffn_out.ptr, d)?;
            self.ew("ink_add", &[Arg::Ptr(a.x.ptr), Arg::Ptr(a.ffn_out.ptr), Arg::Ptr(a.x.ptr)], d)?;

            if let Some(cb) = capture.as_mut() {
                cb(p.slot, self);
            }
        }

        // output head
        let mut st = blocks.clone();
        st.push(a.x.ptr);
        self.mix(&st, self.w.output_res_query, 1, a.f.ptr)?;
        self.rmsnorm(a.f.ptr, self.w.output_norm, a.fnorm.ptr)?;
        self.mv("ink_matvec_q8", self.w.output, a.fnorm.ptr, a.logits.ptr, d, cfg.n_vocab)?;
        let nv = cfg.n_vocab_unpadded;
        self.gpu.launch("ink_softcap", (nv.div_ceil(256) as u32, 1), 256, &[Arg::Ptr(a.logits.ptr), Arg::I32(nv as i32), Arg::F32(cfg.final_softcap)])?;
        self.gpu.read_f32(a.logits.ptr, nv)
    }
}
