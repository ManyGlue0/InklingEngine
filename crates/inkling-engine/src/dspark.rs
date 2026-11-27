//! DSpark drafter for speculative decoding: Inkling 2 Mini's own 3-layer block drafter (Unsloth / llama.cpp GGUF,
//! architecture `dflash` with the DSpark Markov head). Reference: llama.cpp src/models/dflash.cpp + dspark.cpp and
//! common/speculative.cpp (DSpark driver), conversion/inkling2.py.
//!
//! * KV injection: every token the target processes contributes its features (residual stream at layer visits 5, 23
//!   and 41, and the final-norm output), mixed per drafter layer into context K/V at the token's position; the same
//!   mix with the Markov weights gives f_rnn, the Markov head's context of the last committed token.
//! * Draft: block [anchor, mask x k] at positions pos0.. (anchor = the sampled token the target has not processed)
//!   through the 3 layers, non-causal attention over the context before the anchor (layers 0 and 2: the 1024
//!   positions before it) plus the block; logits from the target's output head; then the recurrent Markov head walks
//!   the k draft slots, each slot's token feeding the next.

use anyhow::{Context, Result, ensure};
use inkling_core::Gguf;
use inkling_cuda::{Arg, DevBuf};

use crate::weights::{Loader, Q8M, f32_rows};
use crate::{Engine, F32};

const D: usize = 2304;
const NQ: usize = 24;
const KVW: usize = 1024;
const FF: usize = 9216;
const RANK: usize = 256;
/// Longest draft (block size 8 minus the anchor).
pub const MAX_DRAFT: usize = 7;
/// Rows of the injection scratch and of each feature tap (one prefill chunk).
pub(crate) const FEAT_ROWS: usize = crate::prefill::PREFILL_CHUNK;
const INJ_ROWS: usize = FEAT_ROWS;

struct DLayer {
    swa: bool,
    attn_norm: u64,
    wq: Q8M,
    wkv: Q8M, // k | v rows
    wo: Q8M,
    q_norm: u64,
    k_norm: u64,
    ffn_norm: u64,
    wgu: Q8M, // gate | up rows
    wdown: Q8M,
    ctx_mix: [f32; 4],
    ctx_norm: u64,
    wkv_ctx: Q8M, // k_ctx | v_ctx rows
    k_ctx_norm: u64,
    /// KV ring positions: the sliding-window layers keep the window + one block, the full layer the context
    cap: usize,
    kc: DevBuf, // f16 [cap][1024]
    vc: DevBuf,
}

struct Q8 {
    q: DevBuf,
    d: DevBuf,
}

impl Q8 {
    fn new(gpu: &inkling_cuda::Gpu, n: usize) -> Result<Self> {
        Ok(Self { q: gpu.alloc(n)?, d: gpu.alloc(n / 32 * F32)? })
    }
    fn at(&self, off: usize) -> (u64, u64) {
        (self.q.at(off), self.d.at(off / 32 * F32))
    }
}

pub struct DSpark {
    _bufs: std::collections::HashMap<String, DevBuf>,
    layers: Vec<DLayer>,
    /// target layer visits whose input residual is a feature tap; the last tap is the final-norm output
    pub taps: Vec<usize>,
    n_swa: usize,
    eps: f32,
    rope_base: f32,
    softcap: f32,
    output_norm: u64,
    mask_embd: u64,
    joint_w: u64,
    joint_b: u64,
    markov_mix: [f32; 4],
    markov_norm: u64,
    w1: u64,
    w2: u64,
    b2: u64,
    // block scratch [8][..]
    pub(crate) tokens: DevBuf, // i32 [16]: anchor, then the drafted tokens
    x: DevBuf,
    xn: DevBuf,
    xq: Q8,
    q: DevBuf,
    kv: DevBuf,
    attq: Q8,
    proj: DevBuf,
    gu: DevBuf,
    hq: Q8,
    hid: DevBuf,
    hidq: Q8,
    logits: DevBuf, // [8][n_vocab]
    joint: DevBuf,
    state: DevBuf,
    markov: DevBuf,
    col: DevBuf,
    // injection scratch
    mix: DevBuf,
    mixq: Q8,
    kvc: DevBuf,
    frnn: DevBuf,
    /// f_rnn saved with the prompt-cache snapshot
    frnn_snap: DevBuf,
    /// target features of the last chunk, [4][INJ_ROWS][D] (tap-major)
    pub feat: DevBuf,
}

impl DSpark {
    pub fn load(gpu: &inkling_cuda::Gpu, path: &str, n_ctx: usize, n_vocab: usize, n_vocab_unpadded: usize) -> Result<Self> {
        let g = Gguf::open(path).with_context(|| format!("opening the drafter {path}"))?;
        ensure!(g.str("general.architecture")? == "dflash", "{path}: not a DSpark drafter (architecture is not dflash)");
        ensure!(g.tensor("markov_w1.weight").is_some() && g.tensor("joint_proj.weight").is_some(), "{path}: no DSpark Markov head");
        ensure!(g.tensor("markov_ctx_mix.weight").is_some(), "{path}: older DSpark export without the Markov context (use the mainline-format GGUF)");
        let n_layer = g.u64("dflash.block_count")? as usize;
        ensure!(g.u64("dflash.embedding_length")? as usize == D && g.u64("dflash.feed_forward_length")? as usize == FF, "{path}: unexpected drafter size");
        ensure!(g.u64("dflash.block_size")? as usize == MAX_DRAFT + 1, "{path}: unexpected block size");
        let taps: Vec<usize> = g.get("dflash.target_layers").and_then(|v| v.as_arr()).context("dflash.target_layers")?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
        ensure!(taps.len() == 4, "{path}: expected 4 target layers");
        let swa_pat: Vec<bool> =
            g.get("dflash.attention.sliding_window_pattern").and_then(|v| v.as_arr()).context("sliding_window_pattern")?.iter().map(|v| v.as_u64() == Some(1)).collect();
        let n_swa = g.u64("dflash.attention.sliding_window")? as usize;
        let eps = g.f64("dflash.attention.layer_norm_rms_epsilon")? as f32;
        let rope_base = g.f64("dflash.rope.freq_base")? as f32;
        let softcap = g.f64("dflash.final_logit_softcapping")? as f32;
        ensure!(g.require("markov_w1.weight")?.dims[1] as usize == n_vocab_unpadded, "{path}: drafter vocabulary does not match the model");

        let mut l = Loader { gpu, g: &g, bufs: Default::default(), bytes: 0 };
        let mix4 = |name: &str| -> Result<[f32; 4]> {
            let v = f32_rows(&g, name)?;
            ensure!(v.len() == 4, "{name}: expected 4 weights");
            Ok([v[0], v[1], v[2], v[3]])
        };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let n = |s: &str| format!("blk.{i}.{s}");
            let swa = swa_pat.get(i).copied().unwrap_or(false);
            let cap = if swa { (n_swa + crate::prefill::PREFILL_CHUNK + MAX_DRAFT + 1).min(n_ctx) } else { n_ctx };
            layers.push(DLayer {
                swa,
                cap,
                attn_norm: l.raw(&n("attn_norm.weight"))?,
                wq: l.q8(&[n("attn_q.weight")])?,
                wkv: l.q8(&[n("attn_k.weight"), n("attn_v.weight")])?,
                wo: l.q8(&[n("attn_output.weight")])?,
                q_norm: l.raw(&n("attn_q_norm.weight"))?,
                k_norm: l.raw(&n("attn_k_norm.weight"))?,
                ffn_norm: l.raw(&n("ffn_norm.weight"))?,
                wgu: l.q8(&[n("ffn_gate.weight"), n("ffn_up.weight")])?,
                wdown: l.q8(&[n("ffn_down.weight")])?,
                ctx_mix: mix4(&n("ctx_mix.weight"))?,
                ctx_norm: l.raw(&n("ctx_norm.weight"))?,
                wkv_ctx: l.q8(&[n("attn_k_ctx.weight"), n("attn_v_ctx.weight")])?,
                k_ctx_norm: l.raw(&n("attn_k_ctx_norm.weight"))?,
                kc: gpu.alloc(cap * KVW * 2)?,
                vc: gpu.alloc(cap * KVW * 2)?,
            });
        }
        let output_norm = l.raw("output_norm.weight")?;
        let mask_embd = l.raw("mask_embd.weight")?;
        let joint_w = l.raw("joint_proj.weight")?;
        let joint_b = l.raw("joint_proj.bias")?;
        let markov_norm = l.raw("markov_ctx_norm.weight")?;
        let w1 = l.raw("markov_w1.weight")?;
        let w2 = l.raw("markov_w2.weight")?;
        let b2 = l.raw("markov_w2.bias")?;
        let markov_mix = mix4("markov_ctx_mix.weight")?;
        let bufs = std::mem::take(&mut l.bufs);
        let f = |n: usize| gpu.alloc(n * F32);
        let b = MAX_DRAFT + 1;
        Ok(Self {
            _bufs: bufs,
            layers,
            taps,
            n_swa,
            eps,
            rope_base,
            softcap,
            output_norm,
            mask_embd,
            joint_w,
            joint_b,
            markov_mix,
            markov_norm,
            w1,
            w2,
            b2,
            tokens: f(16)?,
            x: f(b * D)?,
            xn: f(b * D)?,
            xq: Q8::new(gpu, b * D)?,
            q: f(b * NQ * 128)?,
            kv: f(b * 2 * KVW)?,
            attq: Q8::new(gpu, b * NQ * 128)?,
            proj: f(b * D)?,
            gu: f(b * 2 * FF)?,
            hq: Q8::new(gpu, b * FF)?,
            hid: f(b * D)?,
            hidq: Q8::new(gpu, b * D)?,
            logits: f(b * n_vocab)?,
            joint: f(3 * RANK)?,
            state: f(RANK)?,
            markov: f(RANK)?,
            col: f(n_vocab)?,
            mix: f(INJ_ROWS * D)?,
            mixq: Q8::new(gpu, INJ_ROWS * D)?,
            kvc: f(INJ_ROWS * 2 * KVW)?,
            frnn: f(D)?,
            frnn_snap: f(D)?,
            feat: f(4 * INJ_ROWS * D)?,
        })
    }

    /// VRAM of the drafter's KV cache per context token (the one full-attention layer, f16 K and V; the
    /// sliding-window layers are fixed rings).
    pub fn kv_bytes_per_token() -> usize {
        2 * KVW * 2
    }
}

impl Engine {
    fn ds(&self) -> &DSpark {
        match self.draft.as_ref() {
            Some(crate::spec::Drafter::DSpark(d)) => d,
            _ => panic!("no DSpark drafter"),
        }
    }

    /// Inject the target features of `n` consecutive tokens at positions step[1].. of the batched pass (rows 0..n of
    /// the feature buffer, whose rows are `stride` apart per tap) into the drafter's KV cache; f_rnn becomes that of
    /// the last token. Replayable (no host data).
    pub(crate) fn dsp_inject(&self, n: usize, stride: usize) -> Result<()> {
        let step = self.pre.as_ref().expect("prefill buffers").step.ptr;
        ensure!(n > 0 && n <= stride && stride <= INJ_ROWS, "ds_inject: bad row count");
        let ds = self.ds();
        let g = &self.gpu;
        for ly in &ds.layers {
            let a = ly.ctx_mix;
            let (mq, md) = ds.mixq.at(0);
            g.launch("ink_ds_mix4", (n as u32, 1), 768, &[
                Arg::Ptr(ds.feat.ptr), Arg::I32(stride as i32), Arg::F32(a[0]), Arg::F32(a[1]), Arg::F32(a[2]), Arg::F32(a[3]),
                Arg::Ptr(ly.ctx_norm), Arg::F32(ds.eps), Arg::Ptr(ds.mix.ptr), Arg::Ptr(mq), Arg::Ptr(md),
            ])?;
            self.pre_gemm(&[(ly.wkv_ctx, (mq, md), ds.kvc.ptr, 2 * KVW, 2 * KVW)], D, n)?;
            g.launch("ink_ds_qk_rope", (16, n as u32), 128, &[
                Arg::Ptr(0), Arg::I32(0), Arg::Ptr(ds.kvc.ptr), Arg::I32((2 * KVW) as i32), Arg::I32(0), Arg::Ptr(0), Arg::Ptr(ly.k_ctx_norm),
                Arg::Ptr(step), Arg::F32(ds.eps), Arg::F32(ds.rope_base), Arg::Ptr(ly.kc.ptr), Arg::Ptr(ly.vc.ptr), Arg::I32(ly.cap as i32),
            ])?;
        }
        // f_rnn of the last token
        let a = ds.markov_mix;
        g.launch("ink_ds_mix4", (1, 1), 768, &[
            Arg::Ptr(ds.feat.at((n - 1) * D * F32)), Arg::I32(stride as i32), Arg::F32(a[0]), Arg::F32(a[1]), Arg::F32(a[2]), Arg::F32(a[3]),
            Arg::Ptr(ds.markov_norm), Arg::F32(ds.eps), Arg::Ptr(ds.frnn.ptr), Arg::Ptr(0), Arg::Ptr(0),
        ])
    }

    /// Save (`save`) or restore the drafter state that the prompt-cache snapshot needs besides its KV cache.
    pub(crate) fn dsp_snapshot(&self, save: bool) -> Result<()> {
        let ds = self.ds();
        let (src, dst) = if save { (ds.frnn.ptr, ds.frnn_snap.ptr) } else { (ds.frnn_snap.ptr, ds.frnn.ptr) };
        self.copy_n(src, dst, D * F32)
    }

    /// Positions the drafter may process after a snapshot before its sliding-window rings lose what the snapshot
    /// point needs.
    pub(crate) fn dsp_snap_margin(&self) -> usize {
        let ds = self.ds();
        ds.layers.iter().filter(|l| l.swa && l.cap < self.n_ctx).map(|l| l.cap - ds.n_swa - MAX_DRAFT).min().unwrap_or(usize::MAX)
    }

    /// Draft `k` tokens after the anchor (ds_tokens()[0], the sampled token at position step[1] that the target has
    /// not processed), greedy; [anchor, d1..dk] then becomes the input of the next batched pass (its token buffer).
    /// Replayable (no host data).
    pub(crate) fn dsp_draft(&self, k: usize) -> Result<()> {
        ensure!((1..=MAX_DRAFT).contains(&k), "draft length must be 1..={MAX_DRAFT}");
        let ds = self.ds();
        let g = &self.gpu;
        let t = k + 1;
        let cfg = &self.cfg;
        let pre = self.pre.as_ref().expect("prefill buffers");
        let step = pre.step.ptr;
        // block input: anchor embedding (normed with the target's embed norm), then the mask rows
        if self.w.token_embd_fmt == 8 {
            g.launch("ink_embed_b", (1, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::Ptr(ds.tokens.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(ds.x.ptr), Arg::I32(D as i32), Arg::F32(cfg.rms_eps),
            ])?;
        } else {
            g.launch("ink_embed_gen", (1, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::I32(self.w.token_embd_fmt), Arg::Ptr(ds.tokens.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(ds.x.ptr),
                Arg::I32(D as i32), Arg::F32(cfg.rms_eps),
            ])?;
        }
        self.copy_n(ds.mask_embd, ds.x.at(D * F32), k * D * F32)?;
        for ly in &ds.layers {
            let (xq, xd) = ds.xq.at(0);
            self.pre_norm(&[(ds.x.ptr, ly.attn_norm, ds.xn.ptr, (xq, xd))], t)?;
            self.pre_gemm(&[(ly.wq, (xq, xd), ds.q.ptr, NQ * 128, NQ * 128), (ly.wkv, (xq, xd), ds.kv.ptr, 2 * KVW, 2 * KVW)], D, t)?;
            g.launch("ink_ds_qk_rope", ((NQ + 16) as u32, t as u32), 128, &[
                Arg::Ptr(ds.q.ptr), Arg::I32((NQ * 128) as i32), Arg::Ptr(ds.kv.ptr), Arg::I32((2 * KVW) as i32), Arg::I32(NQ as i32), Arg::Ptr(ly.q_norm),
                Arg::Ptr(ly.k_norm), Arg::Ptr(step), Arg::F32(ds.eps), Arg::F32(ds.rope_base), Arg::Ptr(ly.kc.ptr), Arg::Ptr(ly.vc.ptr), Arg::I32(ly.cap as i32),
            ])?;
            let (aq, ad) = ds.attq.at(0);
            let win = if ly.swa { ds.n_swa as i32 - 1 } else { 0 };
            g.launch("ink_ds_attn", (NQ as u32, t as u32), 128, &[
                Arg::Ptr(ds.q.ptr), Arg::I32((NQ * 128) as i32), Arg::I32(NQ as i32), Arg::Ptr(ly.kc.ptr), Arg::Ptr(ly.vc.ptr), Arg::I32(ly.cap as i32), Arg::Ptr(step),
                Arg::I32(t as i32), Arg::I32(win), Arg::Ptr(aq), Arg::Ptr(ad),
            ])?;
            self.pre_gemm(&[(ly.wo, (aq, ad), ds.proj.ptr, D, D)], NQ * 128, t)?;
            g.launch("ink_ds_add", ((t * D).div_ceil(256) as u32, 1), 256, &[Arg::Ptr(ds.x.ptr), Arg::Ptr(ds.proj.ptr), Arg::I32((t * D) as i32)])?;
            self.pre_norm(&[(ds.x.ptr, ly.ffn_norm, ds.xn.ptr, (xq, xd))], t)?;
            self.pre_gemm(&[(ly.wgu, (xq, xd), ds.gu.ptr, 2 * FF, 2 * FF)], D, t)?;
            let (hq, hd) = ds.hq.at(0);
            g.launch("ink_ds_swiglu_q8", ((FF / 256) as u32, t as u32), 256, &[Arg::Ptr(ds.gu.ptr), Arg::I32(FF as i32), Arg::Ptr(hq), Arg::Ptr(hd)])?;
            self.pre_gemm(&[(ly.wdown, (hq, hd), ds.proj.ptr, D, D)], FF, t)?;
            g.launch("ink_ds_add", ((t * D).div_ceil(256) as u32, 1), 256, &[Arg::Ptr(ds.x.ptr), Arg::Ptr(ds.proj.ptr), Arg::I32((t * D) as i32)])?;
        }
        // final norm -> hidden (the Markov head's input) and base logits through the target's output head
        let (hq, hd) = ds.hidq.at(0);
        self.pre_norm(&[(ds.x.ptr, ds.output_norm, ds.hid.ptr, (hq, hd))], t)?;
        let nv = cfg.n_vocab;
        self.pre_gemm(&[(self.w.output, (hq, hd), ds.logits.ptr, nv, nv)], D, t)?;
        // Markov head over the draft slots 1..=k
        let nvu = cfg.n_vocab_unpadded;
        for i in 1..=k {
            let first = (i == 1) as i32;
            g.launch("ink_ds_joint", ((3 * RANK / 8) as u32, 1), 256, &[
                Arg::Ptr(ds.state.ptr), Arg::I32(first), Arg::Ptr(ds.w1), Arg::Ptr(ds.tokens.at((i - 1) * 4)), Arg::Ptr(ds.hid.at(i * D * F32)), Arg::Ptr(ds.frnn.ptr),
                Arg::Ptr(ds.joint_w), Arg::Ptr(ds.joint_b), Arg::Ptr(ds.joint.ptr),
            ])?;
            g.launch("ink_ds_rnn", (1, 1), RANK as u32, &[Arg::Ptr(ds.state.ptr), Arg::I32(first), Arg::Ptr(ds.joint.ptr), Arg::Ptr(ds.markov.ptr)])?;
            g.launch("ink_ds_bias", (nvu.div_ceil(8) as u32, 1), 256, &[
                Arg::Ptr(ds.w2), Arg::Ptr(ds.b2), Arg::Ptr(ds.markov.ptr), Arg::Ptr(ds.logits.at(i * nv * F32)), Arg::F32(ds.softcap), Arg::I32(nvu as i32),
                Arg::Ptr(ds.col.ptr),
            ])?;
            g.launch("ink_argmax", (1, 1), 1024, &[Arg::Ptr(ds.col.ptr), Arg::I32(nvu as i32), Arg::Ptr(ds.tokens.at(i * 4))])?;
        }
        // [anchor, d1..dk] -> the batched pass input
        self.copy_n(ds.tokens.ptr, pre.tokens.ptr, t * 4)
    }
}
