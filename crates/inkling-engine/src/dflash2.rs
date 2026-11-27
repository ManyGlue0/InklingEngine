//! DFlash2 drafter for speculative decoding (Inco AI's block-diffusion drafter for Inkling 2 Mini, safetensors
//! BF16). Reference: llama.cpp src/models/dflash.cpp + dflash2.cpp and the draft-dflash2 driver.
//!
//! * KV injection: the four target features of every committed token, concatenated, go through `fc` and
//!   `hidden_norm`; each layer's own K/V projections (with k_norm and RoPE) write the context K/V.
//! * Draft: block [anchor, mask x k] (the target's normalized token embeddings) through 5 Qwen3-style layers whose
//!   attention and MLP inputs and outputs pass a dynamic two-tap causal convolution over the block; non-causal
//!   attention over the context (sliding window 2048) and the block; soft-capped logits from the target's head. The
//!   candidate selector then walks the top-16 lattice greedily: slot p's token is the candidate maximizing
//!   succ . (pred[previous token] * gate[p]) + logit.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use inkling_cuda::{Arg, DevBuf, Gpu};

use crate::weights::Q8M;
use crate::{Engine, F32};

const D: usize = 2304;
const KVW: usize = 1024;
const RANK: usize = 256;
const TOPK: usize = 16;
pub const MAX_DRAFT: usize = 7;
const FEAT_ROWS: usize = crate::dspark::FEAT_ROWS;

/// A safetensors file: header (name -> dtype, shape, byte range) over the whole file in memory.
struct SafeTensors {
    data: Vec<u8>,
    base: usize,
    meta: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    fn open(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        ensure!(data.len() >= 8, "{}: not a safetensors file", path.display());
        let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        ensure!(8 + n <= data.len(), "{}: bad safetensors header", path.display());
        let h: serde_json::Value = serde_json::from_slice(&data[8..8 + n]).context("safetensors header")?;
        let mut meta = HashMap::new();
        for (k, v) in h.as_object().context("safetensors header")? {
            if k == "__metadata__" {
                continue;
            }
            let dt = v["dtype"].as_str().context("dtype")?.to_owned();
            let shape: Vec<usize> = v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let off = v["data_offsets"].as_array().context("data_offsets")?;
            meta.insert(k.clone(), (dt, shape, off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize));
        }
        Ok(Self { data, base: 8 + n, meta })
    }

    /// BF16 tensor as f32 values and its shape.
    fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (dt, shape, a, b) = self.meta.get(name).with_context(|| format!("DFlash2 drafter: missing tensor {name}"))?;
        ensure!(dt == "BF16", "{name}: expected BF16, got {dt}");
        let raw = &self.data[self.base + a..self.base + b];
        Ok((raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(), shape.clone()))
    }

    fn bf16_raw(&self, name: &str) -> Result<(&[u8], Vec<usize>)> {
        let (dt, shape, a, b) = self.meta.get(name).with_context(|| format!("DFlash2 drafter: missing tensor {name}"))?;
        ensure!(dt == "BF16", "{name}: expected BF16, got {dt}");
        Ok((&self.data[self.base + a..self.base + b], shape.clone()))
    }
}

/// Uploads, quantizing matrices to the engine's repacked Q8_0 (int8 rows + f16 scale per 32).
struct Up<'a> {
    gpu: &'a Gpu,
    st: &'a SafeTensors,
    bufs: Vec<DevBuf>,
}

impl Up<'_> {
    fn put(&mut self, bytes: &[u8]) -> Result<u64> {
        let b = self.gpu.upload(bytes)?;
        let p = b.ptr;
        self.bufs.push(b);
        Ok(p)
    }
    /// One or more [rows][K] BF16 matrices with the same K, row-concatenated, as Q8.
    fn q8(&mut self, names: &[&str]) -> Result<Q8M> {
        let (mut qs, mut sc) = (Vec::new(), Vec::new());
        let mut k0 = 0;
        for n in names {
            let (v, shape) = self.st.f32(n)?;
            ensure!(shape.len() == 2, "{n}: expected a matrix");
            let k = shape[1];
            ensure!(k % 32 == 0 && (k0 == 0 || k0 == k), "{n}: bad inner size");
            k0 = k;
            for g in v.chunks_exact(32) {
                let amax = g.iter().fold(0f32, |m, x| m.max(x.abs()));
                let d = amax / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                qs.extend(g.iter().map(|x| (x * id).round().clamp(-127.0, 127.0) as i8 as u8));
                sc.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            }
        }
        Ok(Q8M { qs: self.put(&qs)?, d: self.put(&sc)?, fmt: 0 })
    }
    fn f32(&mut self, name: &str) -> Result<u64> {
        let (v, _) = self.st.f32(name)?;
        self.put(inkling_cuda::as_bytes(&v))
    }
    fn bf16(&mut self, name: &str) -> Result<u64> {
        let (raw, _) = self.st.bf16_raw(name)?;
        let raw = raw.to_vec();
        self.put(&raw)
    }
}

struct FLayer {
    in_ln: u64,
    post_ln: u64,
    wq: Q8M,
    wkv: Q8M, // k | v
    wo: Q8M,
    q_norm: u64,
    k_norm: u64,
    wgu: Q8M, // gate | up
    wdown: Q8M,
    conv_a: Q8M, // attention conv kernel projection [576][D]
    base_a: u64, // f32 [2][2][D]
    conv_f: Q8M,
    base_f: u64,
    cap: usize,
    kc: DevBuf,
    vc: DevBuf,
}

struct Q8 {
    q: DevBuf,
    d: DevBuf,
}

impl Q8 {
    fn new(gpu: &Gpu, n: usize) -> Result<Self> {
        Ok(Self { q: gpu.alloc(n)?, d: gpu.alloc(n / 32 * F32)? })
    }
    fn at(&self, off: usize) -> (u64, u64) {
        (self.q.at(off), self.d.at(off / 32 * F32))
    }
}

pub struct DFlash2 {
    _bufs: Vec<DevBuf>,
    layers: Vec<FLayer>,
    pub taps: Vec<usize>,
    nq: usize,
    ff: usize,
    n_swa: usize,
    eps: f32,
    rope_base: f32,
    softcap: f32,
    fc: Q8M,
    hidden_norm: u64,
    norm: u64,
    sel_hidden: u64,
    pred: u64,
    succ: u64,
    // block scratch [8][..]
    pub tokens: DevBuf, // i32 [16]: anchor, then the drafted tokens
    etok: DevBuf,       // i32 [8]: anchor, mask x 7 (embedding input)
    x: DevBuf,
    xn: DevBuf,
    xq: Q8,
    dynb: DevBuf, // [8][576]
    cv: DevBuf,
    q: DevBuf,
    kv: DevBuf,
    attq: Q8,
    proj: DevBuf,
    gu: DevBuf,
    hq: Q8,
    hid: DevBuf,
    hidq: Q8,
    logits: DevBuf,
    cand: DevBuf,
    val: DevBuf,
    gate: DevBuf,
    // injection scratch
    fq: Q8,       // [INJ][4 D]
    ginp: DevBuf, // [INJ][D]
    gq: Q8,
    kvc: DevBuf,
    pub feat: DevBuf,
}

impl DFlash2 {
    /// `path`: the drafter's directory (config.json + model.safetensors) or its model.safetensors.
    pub fn load(gpu: &Gpu, path: &str, n_ctx: usize, n_vocab: usize, n_vocab_unpadded: usize) -> Result<Self> {
        let p = std::path::Path::new(path);
        let dir = if p.is_dir() { p.to_path_buf() } else { p.parent().map(|d| d.to_path_buf()).unwrap_or_default() };
        let st_path = if p.is_dir() { dir.join("model.safetensors") } else { p.to_path_buf() };
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("DFlash2 drafter: config.json")?)?;
        ensure!(cfg["architectures"][0].as_str() == Some("DFlash2DraftModel"), "{path}: not a DFlash2 drafter");
        let dcfg = &cfg["dflash_config"];
        let u = |v: &serde_json::Value, what: &str| v.as_u64().map(|x| x as usize).with_context(|| format!("DFlash2 config: {what}"));
        ensure!(u(&cfg["hidden_size"], "hidden_size")? == D && u(&cfg["head_dim"], "head_dim")? == 128 && u(&cfg["num_key_value_heads"], "kv heads")? == 8, "DFlash2: unexpected dimensions");
        ensure!(u(&dcfg["block_size"], "block_size")? == MAX_DRAFT + 1 && u(&dcfg["selector_top_k"], "top_k")? == TOPK && u(&dcfg["selector_rank"], "rank")? == RANK, "DFlash2: unexpected block / selector");
        ensure!(u(&dcfg["conv_kernel_size"], "conv")? == 2 && u(&dcfg["conv_group_size"], "group")? == 16, "DFlash2: unexpected convolution");
        let taps: Vec<usize> = dcfg["target_layer_ids"].as_array().context("target_layer_ids")?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
        ensure!(taps.len() == 4, "DFlash2: expected 4 target layers");
        let n_layer = u(&cfg["num_hidden_layers"], "layers")?;
        let nq = u(&cfg["num_attention_heads"], "heads")?;
        let ff = u(&cfg["intermediate_size"], "intermediate_size")?;
        ensure!(ff % 256 == 0 && nq % 8 == 0, "DFlash2: unexpected sizes");
        let n_swa = if cfg["use_sliding_window"].as_bool() == Some(true) { u(&cfg["sliding_window"], "sliding_window")? } else { 0 };
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let rope_base = cfg["rope_parameters"]["rope_theta"].as_f64().or(cfg["rope_theta"].as_f64()).unwrap_or(10000.0) as f32;
        let softcap = dcfg["final_logit_softcapping"].as_f64().unwrap_or(0.0) as f32;
        let mask_id = u(&dcfg["mask_token_id"], "mask_token_id")?;
        ensure!(mask_id < n_vocab_unpadded, "DFlash2: mask token outside the vocabulary");

        let st = SafeTensors::open(&st_path)?;
        let (_, cb) = st.bf16_raw("candidate_selector.predecessor_codebook")?;
        ensure!(cb[0] >= n_vocab_unpadded && cb[1] == RANK, "DFlash2: codebook does not match the model's vocabulary");
        let mut up = Up { gpu, st: &st, bufs: Vec::new() };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let n = |s: &str| format!("layers.{i}.{s}");
            let cap = if n_swa > 0 { (n_swa + crate::prefill::PREFILL_CHUNK + MAX_DRAFT + 1).min(n_ctx) } else { n_ctx };
            layers.push(FLayer {
                in_ln: up.f32(&n("input_layernorm.weight"))?,
                post_ln: up.f32(&n("post_attention_layernorm.weight"))?,
                wq: up.q8(&[&n("self_attn.q_proj.weight")])?,
                wkv: up.q8(&[&n("self_attn.k_proj.weight"), &n("self_attn.v_proj.weight")])?,
                wo: up.q8(&[&n("self_attn.o_proj.weight")])?,
                q_norm: up.f32(&n("self_attn.q_norm.weight"))?,
                k_norm: up.f32(&n("self_attn.k_norm.weight"))?,
                wgu: up.q8(&[&n("mlp.gate_proj.weight"), &n("mlp.up_proj.weight")])?,
                wdown: up.q8(&[&n("mlp.down_proj.weight")])?,
                conv_a: up.q8(&[&n("attention_conv.kernel_projection.weight")])?,
                base_a: up.f32(&n("attention_conv.base_kernel"))?,
                conv_f: up.q8(&[&n("mlp_conv.kernel_projection.weight")])?,
                base_f: up.f32(&n("mlp_conv.base_kernel"))?,
                cap,
                kc: gpu.alloc(cap * KVW * 2)?,
                vc: gpu.alloc(cap * KVW * 2)?,
            });
        }
        let fc = up.q8(&["fc.weight"])?;
        let hidden_norm = up.f32("hidden_norm.weight")?;
        let norm = up.f32("norm.weight")?;
        let sel_hidden = up.bf16("candidate_selector.hidden_projection.weight")?;
        let pred = up.bf16("candidate_selector.predecessor_codebook")?;
        let succ = up.bf16("candidate_selector.successor_codebook")?;
        let bufs = std::mem::take(&mut up.bufs);
        drop(st);
        let f = |n: usize| gpu.alloc(n * F32);
        let b = MAX_DRAFT + 1;
        let etok = f(b)?;
        let mut e = vec![mask_id as i32; b];
        e[0] = 0;
        gpu.write(etok.ptr, inkling_cuda::as_bytes(&e))?;
        let _ = n_vocab;
        Ok(Self {
            _bufs: bufs,
            layers,
            taps,
            nq,
            ff,
            n_swa,
            eps,
            rope_base,
            softcap,
            fc,
            hidden_norm,
            norm,
            sel_hidden,
            pred,
            succ,
            tokens: f(16)?,
            etok,
            x: f(b * D)?,
            xn: f(b * D)?,
            xq: Q8::new(gpu, b * D)?,
            dynb: f(b * 576)?,
            cv: f(b * D)?,
            q: f(b * nq * 128)?,
            kv: f(b * 2 * KVW)?,
            attq: Q8::new(gpu, b * nq * 128)?,
            proj: f(b * D)?,
            gu: f(b * 2 * ff)?,
            hq: Q8::new(gpu, b * ff)?,
            hid: f(b * D)?,
            hidq: Q8::new(gpu, b * D)?,
            logits: f(b * n_vocab)?,
            cand: f(b * TOPK)?,
            val: f(b * TOPK)?,
            gate: f(b * RANK)?,
            fq: Q8::new(gpu, FEAT_ROWS * 4 * D)?,
            ginp: f(FEAT_ROWS * D)?,
            gq: Q8::new(gpu, FEAT_ROWS * D)?,
            kvc: f(FEAT_ROWS * 2 * KVW)?,
            feat: f(4 * FEAT_ROWS * D)?,
        })
    }

    pub fn snap_margin(&self, n_ctx: usize) -> usize {
        self.layers.iter().filter(|l| l.cap < n_ctx).map(|l| l.cap - self.n_swa - MAX_DRAFT).min().unwrap_or(usize::MAX)
    }
}

impl Engine {
    fn df(&self) -> &DFlash2 {
        match self.draft.as_ref() {
            Some(crate::spec::Drafter::DFlash2(d)) => d,
            _ => panic!("no DFlash2 drafter"),
        }
    }

    /// KV injection of `n` tokens at positions step[1].. (rows 0..n of the feature buffer). Replayable.
    pub(crate) fn df_inject(&self, n: usize, stride: usize) -> Result<()> {
        ensure!(n > 0 && n <= stride && stride <= FEAT_ROWS, "df_inject: bad row count");
        let df = self.df();
        let g = &self.gpu;
        let step = self.pre.as_ref().expect("prefill buffers").step.ptr;
        let (fq, fd) = df.fq.at(0);
        g.launch("ink_df_feat_q8", (4, n as u32), 768, &[Arg::Ptr(df.feat.ptr), Arg::I32(stride as i32), Arg::Ptr(fq), Arg::Ptr(fd)])?;
        self.pre_gemm(&[(df.fc, (fq, fd), df.ginp.ptr, D, D)], 4 * D, n)?;
        let (gq, gd) = df.gq.at(0);
        self.pre_norm(&[(df.ginp.ptr, df.hidden_norm, df.kvc.ptr, (gq, gd))], n)?;
        for ly in &df.layers {
            self.pre_gemm(&[(ly.wkv, (gq, gd), df.kvc.ptr, 2 * KVW, 2 * KVW)], D, n)?;
            g.launch("ink_ds_qk_rope", (16, n as u32), 128, &[
                Arg::Ptr(0), Arg::I32(0), Arg::Ptr(df.kvc.ptr), Arg::I32((2 * KVW) as i32), Arg::I32(0), Arg::Ptr(0), Arg::Ptr(ly.k_norm),
                Arg::Ptr(step), Arg::F32(df.eps), Arg::F32(df.rope_base), Arg::Ptr(ly.kc.ptr), Arg::Ptr(ly.vc.ptr), Arg::I32(ly.cap as i32),
            ])?;
        }
        Ok(())
    }

    /// Draft `k` tokens after the anchor (tokens[0]); [anchor, d1..dk] becomes the next batched pass's input.
    /// Replayable.
    pub(crate) fn df_draft(&self, k: usize) -> Result<()> {
        if !(1..=MAX_DRAFT).contains(&k) {
            bail!("draft length must be 1..={MAX_DRAFT}");
        }
        let df = self.df();
        let g = &self.gpu;
        let t = k + 1;
        let cfg = &self.cfg;
        let pre = self.pre.as_ref().expect("prefill buffers");
        let step = pre.step.ptr;
        let nq = df.nq;
        let qd = nq * 128;
        // block input: [anchor, mask x k] through the target's (normalized) token embeddings
        self.copy_n(df.tokens.ptr, df.etok.ptr, 4)?;
        if self.w.token_embd_fmt == 8 {
            g.launch("ink_embed_b", (t as u32, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::Ptr(df.etok.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(df.x.ptr), Arg::I32(D as i32), Arg::F32(cfg.rms_eps),
            ])?;
        } else {
            g.launch("ink_embed_gen", (t as u32, 1), 256, &[
                Arg::Ptr(self.w.token_embd), Arg::I32(self.w.token_embd_fmt), Arg::Ptr(df.etok.ptr), Arg::Ptr(self.w.token_embd_norm), Arg::Ptr(df.x.ptr),
                Arg::I32(D as i32), Arg::F32(cfg.rms_eps),
            ])?;
        }
        let (xq, xd) = df.xq.at(0);
        let (aq, ad) = df.attq.at(0);
        let (hq, hd) = df.hq.at(0);
        let win = df.n_swa as i32;
        for ly in &df.layers {
            // attention: norm -> dynamic kernels -> conv (side 0) -> q, k, v
            self.pre_norm(&[(df.x.ptr, ly.in_ln, df.xn.ptr, (xq, xd))], t)?;
            self.pre_gemm(&[(ly.conv_a, (xq, xd), df.dynb.ptr, 576, 576)], D, t)?;
            g.launch("ink_df_conv", (t as u32, 1), 768, &[
                Arg::Ptr(df.xn.ptr), Arg::Ptr(df.dynb.ptr), Arg::Ptr(ly.base_a), Arg::I32(0), Arg::Ptr(df.cv.ptr), Arg::I32(0), Arg::Ptr(xq), Arg::Ptr(xd),
            ])?;
            self.pre_gemm(&[(ly.wq, (xq, xd), df.q.ptr, qd, qd), (ly.wkv, (xq, xd), df.kv.ptr, 2 * KVW, 2 * KVW)], D, t)?;
            g.launch("ink_ds_qk_rope", ((nq + 16) as u32, t as u32), 128, &[
                Arg::Ptr(df.q.ptr), Arg::I32(qd as i32), Arg::Ptr(df.kv.ptr), Arg::I32((2 * KVW) as i32), Arg::I32(nq as i32), Arg::Ptr(ly.q_norm),
                Arg::Ptr(ly.k_norm), Arg::Ptr(step), Arg::F32(df.eps), Arg::F32(df.rope_base), Arg::Ptr(ly.kc.ptr), Arg::Ptr(ly.vc.ptr), Arg::I32(ly.cap as i32),
            ])?;
            g.launch("ink_ds_attn", (nq as u32, t as u32), 128, &[
                Arg::Ptr(df.q.ptr), Arg::I32(qd as i32), Arg::I32(nq as i32), Arg::Ptr(ly.kc.ptr), Arg::Ptr(ly.vc.ptr), Arg::I32(ly.cap as i32), Arg::Ptr(step),
                Arg::I32(t as i32), Arg::I32(win), Arg::Ptr(aq), Arg::Ptr(ad),
            ])?;
            self.pre_gemm(&[(ly.wo, (aq, ad), df.proj.ptr, D, D)], qd, t)?;
            // conv (side 1) of the attention output, added to the residual
            g.launch("ink_df_conv", (t as u32, 1), 768, &[
                Arg::Ptr(df.proj.ptr), Arg::Ptr(df.dynb.ptr), Arg::Ptr(ly.base_a), Arg::I32(1), Arg::Ptr(df.x.ptr), Arg::I32(1), Arg::Ptr(0), Arg::Ptr(0),
            ])?;
            // MLP, the same way
            self.pre_norm(&[(df.x.ptr, ly.post_ln, df.xn.ptr, (xq, xd))], t)?;
            self.pre_gemm(&[(ly.conv_f, (xq, xd), df.dynb.ptr, 576, 576)], D, t)?;
            g.launch("ink_df_conv", (t as u32, 1), 768, &[
                Arg::Ptr(df.xn.ptr), Arg::Ptr(df.dynb.ptr), Arg::Ptr(ly.base_f), Arg::I32(0), Arg::Ptr(df.cv.ptr), Arg::I32(0), Arg::Ptr(xq), Arg::Ptr(xd),
            ])?;
            self.pre_gemm(&[(ly.wgu, (xq, xd), df.gu.ptr, 2 * df.ff, 2 * df.ff)], D, t)?;
            g.launch("ink_ds_swiglu_q8", ((df.ff / 256) as u32, t as u32), 256, &[Arg::Ptr(df.gu.ptr), Arg::I32(df.ff as i32), Arg::Ptr(hq), Arg::Ptr(hd)])?;
            self.pre_gemm(&[(ly.wdown, (hq, hd), df.proj.ptr, D, D)], df.ff, t)?;
            g.launch("ink_df_conv", (t as u32, 1), 768, &[
                Arg::Ptr(df.proj.ptr), Arg::Ptr(df.dynb.ptr), Arg::Ptr(ly.base_f), Arg::I32(1), Arg::Ptr(df.x.ptr), Arg::I32(1), Arg::Ptr(0), Arg::Ptr(0),
            ])?;
        }
        // final norm -> hidden; logits of the draft slots through the target's head; selector lattice walk
        let (oq, od) = df.hidq.at(0);
        self.pre_norm(&[(df.x.ptr, df.norm, df.hid.ptr, (oq, od))], t)?;
        let nv = cfg.n_vocab;
        self.pre_gemm(&[(self.w.output, (oq, od), df.logits.ptr, nv, nv)], D, t)?;
        g.launch("ink_df_topk16", (t as u32, 1), 1024, &[
            Arg::Ptr(df.logits.ptr), Arg::I32(nv as i32), Arg::I32(cfg.n_vocab_unpadded as i32), Arg::F32(df.softcap), Arg::Ptr(df.cand.ptr), Arg::Ptr(df.val.ptr),
        ])?;
        g.launch("ink_df_gate", ((RANK / 8) as u32, t as u32), 256, &[Arg::Ptr(df.sel_hidden), Arg::Ptr(df.hid.ptr), Arg::Ptr(df.gate.ptr)])?;
        g.launch("ink_df_select", (1, 1), 512, &[
            Arg::Ptr(df.cand.ptr), Arg::Ptr(df.val.ptr), Arg::Ptr(df.gate.ptr), Arg::Ptr(df.pred), Arg::Ptr(df.succ), Arg::Ptr(df.tokens.ptr), Arg::I32(k as i32),
        ])?;
        self.copy_n(df.tokens.ptr, pre.tokens.ptr, t * 4)
    }
}
