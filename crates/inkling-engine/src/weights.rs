//! Upload GGUF tensors to the GPU and resolve them into per-layer pointer tables.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use inkling_core::{Config, GgmlType, Gguf};
use inkling_cuda::{DevBuf, Gpu};

/// Expert weight format code passed to kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFmt {
    Q8_0 = 0,
    Nvfp4 = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct LoopW {
    /// q, k, v, ffn controls: down [R][2D], gain [D][R], forget [D][R] (F32)
    pub down: [u64; 4],
    pub gain: [u64; 4],
    pub forget: [u64; 4],
    /// message queries [4][D] rows (q, k, v, ffn), assembled at load
    pub msg_query: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerW {
    pub is_swa: bool,
    pub attn_norm: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub v_norm: u64,
    pub wq: u64,
    pub wk: u64,
    pub wv: u64,
    pub wo: u64,
    pub wgate: u64,
    pub wrel: u64,
    pub wsink: u64,
    pub kconv_proj: u64,
    pub vconv_proj: u64,
    pub q_clip: u64,
    pub rel_clip: u64,
    pub rel_bias: u64,
    pub k_conv: u64,
    pub v_conv: u64,
    pub attn_post_conv: u64,
    pub ffn_post_conv: u64,
    /// 0 for layer 0 (no depth attention before attention)
    pub attn_res_query: u64,
    pub ffn_res_query: u64,
    pub ffn_norm: u64,
    pub router: u64,
    pub probs_bias: u64,
    pub gate_up: u64,
    pub gate_up_scale: u64,
    pub down: u64,
    pub down_scale: u64,
    pub expert_fmt: ExpertFmt,
    pub lp: Option<LoopW>,
}

pub struct Weights {
    pub bufs: HashMap<String, DevBuf>,
    pub token_embd: u64,
    pub token_embd_norm: u64,
    pub output: u64,
    pub output_norm: u64,
    pub output_res_query: u64,
    pub layers: Vec<LayerW>,
    pub bytes: usize,
}

impl Weights {
    pub fn load(gpu: &Gpu, g: &Gguf, cfg: &Config) -> Result<Self> {
        let mut bufs = HashMap::with_capacity(g.tensors.len());
        let mut bytes = 0;
        for t in &g.tensors {
            match t.ty {
                GgmlType::F32 | GgmlType::BF16 | GgmlType::Q8_0 | GgmlType::NVFP4 => {}
                other => bail!("tensor {} has unsupported type {other:?}", t.name),
            }
            let b = gpu.upload(g.data(t)).with_context(|| format!("uploading {}", t.name))?;
            bytes += b.bytes;
            bufs.insert(t.name.clone(), b);
        }
        let p = |name: &str| -> Result<u64> { bufs.get(name).map(|b| b.ptr).with_context(|| format!("missing tensor {name}")) };
        let opt = |name: &str| -> u64 { bufs.get(name).map(|b| b.ptr).unwrap_or(0) };

        let d = cfg.n_embd;
        let mut layers = Vec::with_capacity(cfg.n_phys);
        let mut extra = Vec::new();
        for i in 0..cfg.n_phys {
            let n = |s: &str| format!("blk.{i}.{s}");
            let gu = g.require(&n("ffn_gate_up_exps.weight"))?;
            let expert_fmt = match gu.ty {
                GgmlType::NVFP4 => ExpertFmt::Nvfp4,
                GgmlType::Q8_0 => ExpertFmt::Q8_0,
                t => bail!("layer {i}: unsupported expert type {t:?}"),
            };
            ensure!(g.require(&n("ffn_down_exps.weight"))?.ty == gu.ty, "layer {i}: gate_up/down types differ");
            let lp = if (cfg.core_start..cfg.core_end).contains(&i) {
                let names = ["attn", "k", "v", "mlp"];
                let mut lw = LoopW { down: [0; 4], gain: [0; 4], forget: [0; 4], msg_query: 0 };
                for (c, nm) in names.iter().enumerate() {
                    lw.down[c] = p(&n(&format!("loop_{nm}_down.weight")))?;
                    lw.gain[c] = p(&n(&format!("loop_{nm}_gain.weight")))?;
                    lw.forget[c] = p(&n(&format!("loop_{nm}_forget.weight")))?;
                }
                // message queries of core layer ci: rows [res[2ci], kv[2ci], kv[2ci+1], res[2ci+1]]
                let ci = i - cfg.core_start;
                let res = f32_rows(g, "loop_res_query.weight")?;
                let kv = f32_rows(g, "loop_kv_res_query.weight")?;
                let mut q = Vec::with_capacity(4 * d);
                q.extend_from_slice(&res[2 * ci * d..(2 * ci + 1) * d]);
                q.extend_from_slice(&kv[2 * ci * d..(2 * ci + 2) * d]);
                q.extend_from_slice(&res[(2 * ci + 1) * d..(2 * ci + 2) * d]);
                let b = gpu.upload_f32(&q)?;
                lw.msg_query = b.ptr;
                extra.push((format!("blk.{i}.msg_query"), b));
                Some(lw)
            } else {
                None
            };
            layers.push(LayerW {
                is_swa: cfg.is_swa[i],
                attn_norm: opt(&n("attn_norm.weight")),
                q_norm: opt(&n("attn_q_inp_norm.weight")),
                k_norm: opt(&n("attn_k_inp_norm.weight")),
                v_norm: opt(&n("attn_v_inp_norm.weight")),
                wq: p(&n("attn_q.weight"))?,
                wk: p(&n("attn_k.weight"))?,
                wv: p(&n("attn_v.weight"))?,
                wo: p(&n("attn_output.weight"))?,
                wgate: p(&n("attn_gate.weight"))?,
                wrel: p(&n("attn_rel.weight"))?,
                wsink: p(&n("attn_sink_proj.weight"))?,
                kconv_proj: opt(&n("attn_k_conv_proj.weight")),
                vconv_proj: opt(&n("attn_v_conv_proj.weight")),
                q_clip: p(&n("attn_q_clip.weight"))?,
                rel_clip: p(&n("attn_rel_clip.weight"))?,
                rel_bias: p(&n("attn_rel_bias.weight"))?,
                k_conv: p(&n("attn_k_conv.weight"))?,
                v_conv: p(&n("attn_v_conv.weight"))?,
                attn_post_conv: p(&n("attn_post_conv.weight"))?,
                ffn_post_conv: p(&n("ffn_post_conv.weight"))?,
                attn_res_query: opt(&n("attn_res_query.weight")),
                ffn_res_query: p(&n("ffn_res_query.weight"))?,
                ffn_norm: p(&n("ffn_norm.weight"))?,
                router: p(&n("ffn_gate_inp.weight"))?,
                probs_bias: opt(&n("exp_probs_b.bias")),
                gate_up: p(&n("ffn_gate_up_exps.weight"))?,
                gate_up_scale: opt(&n("ffn_gate_up_exps.scale")),
                down: p(&n("ffn_down_exps.weight"))?,
                down_scale: opt(&n("ffn_down_exps.scale")),
                expert_fmt,
                lp,
            });
        }
        let mut me = Self {
            token_embd: p("token_embd.weight")?,
            token_embd_norm: p("token_embd_norm.weight")?,
            output: p("output.weight")?,
            output_norm: p("output_norm.weight")?,
            output_res_query: p("output_res_query.weight")?,
            layers,
            bytes,
            bufs,
        };
        for (k, b) in extra {
            me.bytes += b.bytes;
            me.bufs.insert(k, b);
        }
        Ok(me)
    }
}

fn f32_rows(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.require(name)?;
    ensure!(t.ty == GgmlType::F32, "{name} is not F32");
    Ok(g.data(t).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}
