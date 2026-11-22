//! Upload GGUF tensors to the GPU (large matrices repacked, see repack.rs) and resolve per-layer pointer tables.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use inkling_core::{Config, GgmlType, Gguf};
use inkling_cuda::{DevBuf, Gpu, HostMapped};

use crate::cpu_moe::{CpuExp, CpuLayer};
use crate::repack::{self, Soa};

/// Expert weight format code passed to kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFmt {
    Q8_0 = 0,
    Nvfp4 = 1,
}

/// Repacked Q8 matrix (qs int8 [rows][K], d f16 [rows][K/32]).
#[derive(Debug, Clone, Copy, Default)]
pub struct Q8M {
    pub qs: u64,
    pub d: u64,
}

/// Repacked expert tensor [E][rows][K].
#[derive(Debug, Clone, Copy)]
pub struct ExpM {
    pub qs: u64,
    pub sc: u64,
    pub scale: u64,
    pub fmt: ExpertFmt,
}

#[derive(Debug, Clone, Copy)]
pub struct LoopW {
    /// weights stored as F16 (default) or F32 (INKLING_LOOP_F32=1)
    pub f16: bool,
    /// q, k, v, ffn controls: down [R][2D], gain [D][R], forget [D][R]
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
    /// q (H*dh rows) | rel (H*R rows) | gate (H*dh rows), one matrix
    pub wqrg: Q8M,
    pub wk: Q8M,
    pub wv: Q8M,
    pub wo: Q8M,
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
    pub gate_up: ExpM,
    pub down: ExpM,
    pub lp: Option<LoopW>,
}

pub struct Weights {
    pub bufs: HashMap<String, DevBuf>,
    /// tensors kept in pinned host memory that kernels read over PCIe (token embeddings: one row per token)
    pub host_bufs: Vec<HostMapped>,
    pub token_embd: u64,
    pub token_embd_norm: u64,
    pub output: Q8M,
    pub output_norm: u64,
    pub output_res_query: u64,
    pub layers: Vec<LayerW>,
    /// CPU-offloaded MoE layers (expert planes in `host_bufs`), by physical layer
    pub cpu: Vec<Option<CpuLayer>>,
    pub bytes: usize,
}

struct Loader<'a> {
    gpu: &'a Gpu,
    g: &'a Gguf,
    bufs: HashMap<String, DevBuf>,
    bytes: usize,
}

impl Loader<'_> {
    fn put(&mut self, key: String, data: &[u8]) -> Result<u64> {
        let b = self.gpu.upload(data).with_context(|| format!("uploading {key}"))?;
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.insert(key, b);
        Ok(p)
    }

    /// Small tensors (F32/BF16) uploaded as-is.
    fn raw(&mut self, name: &str) -> Result<u64> {
        let t = self.g.require(name)?;
        ensure!(matches!(t.ty, GgmlType::F32 | GgmlType::BF16 | GgmlType::Q8_0), "{name}: unexpected type {:?}", t.ty);
        let data = self.g.data(t).to_vec();
        self.put(name.to_owned(), &data)
    }

    /// F32 tensor converted to F16 on upload.
    fn f16(&mut self, name: &str) -> Result<u64> {
        let t = self.g.require(name)?;
        ensure!(t.ty == GgmlType::F32, "{name}: expected F32");
        let h: Vec<u16> = self.g.data(t).chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_bits()).collect();
        self.put(format!("{name}#f16"), inkling_cuda::as_bytes(&h))
    }

    fn raw_opt(&mut self, name: &str) -> Result<u64> {
        if self.g.tensor(name).is_some() { self.raw(name) } else { Ok(0) }
    }

    /// One or more Q8_0 2-D tensors with the same K, row-concatenated and repacked.
    fn q8(&mut self, names: &[String]) -> Result<Q8M> {
        let mut soa = Soa { qs: Vec::new(), sc: Vec::new() };
        let mut k0 = 0;
        for n in names {
            let t = self.g.require(n)?;
            ensure!(t.ty == GgmlType::Q8_0 && t.dims.len() == 2, "{n}: expected 2-D Q8_0");
            let (k, rows) = (t.dims[0] as usize, t.dims[1] as usize);
            ensure!(k0 == 0 || k0 == k, "{n}: K mismatch in concatenation");
            k0 = k;
            let s = repack::q8_soa(self.g.data(t), rows, k)?;
            soa.qs.extend_from_slice(&s.qs);
            soa.sc.extend_from_slice(&s.sc);
        }
        let key = names.join("+");
        Ok(Q8M { qs: self.put(format!("{key}#qs"), &soa.qs)?, d: self.put(format!("{key}#d"), &soa.sc)? })
    }

    fn experts(&mut self, name: &str, scale: &str) -> Result<ExpM> {
        let t = self.g.require(name)?;
        ensure!(t.dims.len() == 3, "{name}: expected 3-D expert tensor");
        let (k, rows, e) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
        let (soa, fmt) = match t.ty {
            GgmlType::NVFP4 => (repack::nvfp4_soa(self.g.data(t), rows * e, k)?, ExpertFmt::Nvfp4),
            GgmlType::Q8_0 => (repack::q8_soa(self.g.data(t), rows * e, k)?, ExpertFmt::Q8_0),
            ty => bail!("{name}: unsupported expert type {ty:?}"),
        };
        let qs = self.put(format!("{name}#qs"), &soa.qs)?;
        let sc = self.put(format!("{name}#sc"), &soa.sc)?;
        let scale = self.raw_opt(scale)?;
        Ok(ExpM { qs, sc, scale, fmt })
    }

    /// Expert tensor repacked into pinned host memory (CPU offload). The returned ExpM points at the mapped planes
    /// (a GPU kernel can still read them over PCIe); the per-expert scales are also uploaded.
    fn experts_host(&mut self, name: &str, scale: &str, host: &mut Vec<HostMapped>) -> Result<(ExpM, CpuExp, Option<Vec<f32>>)> {
        let t = self.g.require(name)?;
        ensure!(t.dims.len() == 3, "{name}: expected 3-D expert tensor");
        let (k, rows, e) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
        let fmt = match t.ty {
            GgmlType::NVFP4 => ExpertFmt::Nvfp4,
            GgmlType::Q8_0 => ExpertFmt::Q8_0,
            ty => bail!("{name}: unsupported expert type {ty:?}"),
        };
        let qb = if fmt == ExpertFmt::Nvfp4 { rows * e * k / 2 } else { rows * e * k };
        let mut hq = self.gpu.alloc_host_mapped_uninit(qb).with_context(|| format!("pinning {} MiB of system RAM for {name}", qb >> 20))?;
        let mut hs = self.gpu.alloc_host_mapped_uninit(rows * e * k / 16).with_context(|| format!("pinning system RAM for {name}"))?;
        match fmt {
            ExpertFmt::Nvfp4 => repack::nvfp4_soa_into(self.g.data(t), rows * e, k, hq.as_mut_slice(), hs.as_mut_slice())?,
            ExpertFmt::Q8_0 => repack::q8_soa_into(self.g.data(t), rows * e, k, hq.as_mut_slice(), hs.as_mut_slice())?,
        }
        let scale_dev = self.raw_opt(scale)?;
        let scale_host = if self.g.tensor(scale).is_some() { Some(f32_rows(self.g, scale)?) } else { None };
        let m = ExpM { qs: hq.dev, sc: hs.dev, scale: scale_dev, fmt };
        let c = CpuExp { qs: hq.host_ptr(), sc: hs.host_ptr(), fmt, rows, k };
        host.push(hq);
        host.push(hs);
        Ok((m, c, scale_host))
    }
}

impl Weights {
    /// Upload the model; layers with `on_cpu[i]` keep their experts in pinned system RAM instead.
    pub fn load(gpu: &Gpu, g: &Gguf, cfg: &Config, on_cpu: &[bool]) -> Result<Self> {
        let mut l = Loader { gpu, g, bufs: HashMap::new(), bytes: 0 };
        let mut host_bufs = Vec::new();
        let mut cpu = Vec::with_capacity(cfg.n_phys);
        let d = cfg.n_embd;
        let res = f32_rows(g, "loop_res_query.weight")?;
        let kv = f32_rows(g, "loop_kv_res_query.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_phys);
        for i in 0..cfg.n_phys {
            let n = |s: &str| format!("blk.{i}.{s}");
            let lp = if (cfg.core_start..cfg.core_end).contains(&i) {
                let f16 = std::env::var_os("INKLING_LOOP_F32").is_none();
                let mut lw = LoopW { f16, down: [0; 4], gain: [0; 4], forget: [0; 4], msg_query: 0 };
                for (c, nm) in ["attn", "k", "v", "mlp"].iter().enumerate() {
                    for (dst, kind) in [(&mut lw.down[c], "down"), (&mut lw.gain[c], "gain"), (&mut lw.forget[c], "forget")] {
                        let name = n(&format!("loop_{nm}_{kind}.weight"));
                        *dst = if f16 { l.f16(&name)? } else { l.raw(&name)? };
                    }
                }
                // message queries of core layer ci: rows [res[2ci], kv[2ci], kv[2ci+1], res[2ci+1]] (q, k, v, ffn)
                let ci = i - cfg.core_start;
                let mut q = Vec::with_capacity(4 * d);
                q.extend_from_slice(&res[2 * ci * d..(2 * ci + 1) * d]);
                q.extend_from_slice(&kv[2 * ci * d..(2 * ci + 2) * d]);
                q.extend_from_slice(&res[(2 * ci + 1) * d..(2 * ci + 2) * d]);
                lw.msg_query = l.put(n("msg_query"), inkling_cuda::as_bytes(&q))?;
                Some(lw)
            } else {
                None
            };
            let (gate_up, down) = if on_cpu.get(i).copied().unwrap_or(false) {
                let (gu, cgu, sgu) = l.experts_host(&n("ffn_gate_up_exps.weight"), &n("ffn_gate_up_exps.scale"), &mut host_bufs)?;
                let (dn, cdn, sdn) = l.experts_host(&n("ffn_down_exps.weight"), &n("ffn_down_exps.scale"), &mut host_bufs)?;
                cpu.push(Some(CpuLayer { gate_up: cgu, down: cdn, gu_scale: sgu, down_scale: sdn }));
                (gu, dn)
            } else {
                cpu.push(None);
                (l.experts(&n("ffn_gate_up_exps.weight"), &n("ffn_gate_up_exps.scale"))?, l.experts(&n("ffn_down_exps.weight"), &n("ffn_down_exps.scale"))?)
            };
            layers.push(LayerW {
                is_swa: cfg.is_swa[i],
                attn_norm: l.raw_opt(&n("attn_norm.weight"))?,
                q_norm: l.raw_opt(&n("attn_q_inp_norm.weight"))?,
                k_norm: l.raw_opt(&n("attn_k_inp_norm.weight"))?,
                v_norm: l.raw_opt(&n("attn_v_inp_norm.weight"))?,
                wqrg: l.q8(&[n("attn_q.weight"), n("attn_rel.weight"), n("attn_gate.weight")])?,
                wk: l.q8(&[n("attn_k.weight")])?,
                wv: l.q8(&[n("attn_v.weight")])?,
                wo: l.q8(&[n("attn_output.weight")])?,
                wsink: l.raw(&n("attn_sink_proj.weight"))?,
                kconv_proj: l.raw_opt(&n("attn_k_conv_proj.weight"))?,
                vconv_proj: l.raw_opt(&n("attn_v_conv_proj.weight"))?,
                q_clip: l.raw(&n("attn_q_clip.weight"))?,
                rel_clip: l.raw(&n("attn_rel_clip.weight"))?,
                rel_bias: l.raw(&n("attn_rel_bias.weight"))?,
                k_conv: l.raw(&n("attn_k_conv.weight"))?,
                v_conv: l.raw(&n("attn_v_conv.weight"))?,
                attn_post_conv: l.raw(&n("attn_post_conv.weight"))?,
                ffn_post_conv: l.raw(&n("ffn_post_conv.weight"))?,
                attn_res_query: l.raw_opt(&n("attn_res_query.weight"))?,
                ffn_res_query: l.raw(&n("ffn_res_query.weight"))?,
                ffn_norm: l.raw(&n("ffn_norm.weight"))?,
                router: l.raw(&n("ffn_gate_inp.weight"))?,
                probs_bias: l.raw_opt(&n("exp_probs_b.bias"))?,
                gate_up,
                down,
                lp,
            });
        }
        // the token embedding table is only gathered (one row per token): keep it in host memory, readable by the GPU
        // over PCIe, and save ~0.5 GB of VRAM (INKLING_EMBED_GPU=1 keeps it on the GPU)
        let token_embd = if embed_on_host() {
            let t = g.require("token_embd.weight")?;
            let mut h = gpu.alloc_host_mapped(t.n_bytes)?;
            h.as_mut_slice().copy_from_slice(g.data(t));
            let p = h.dev;
            host_bufs.push(h);
            p
        } else {
            l.raw("token_embd.weight")?
        };
        let token_embd_norm = l.raw("token_embd_norm.weight")?;
        let output = l.q8(&["output.weight".to_owned()])?;
        let output_norm = l.raw("output_norm.weight")?;
        let output_res_query = l.raw("output_res_query.weight")?;
        Ok(Self { host_bufs, token_embd, token_embd_norm, output, output_norm, output_res_query, layers, cpu, bytes: l.bytes, bufs: l.bufs })
    }
}

/// Whether the token embedding table lives in host memory (default) instead of VRAM.
pub fn embed_on_host() -> bool {
    std::env::var_os("INKLING_EMBED_GPU").is_none()
}

fn f32_rows(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.require(name)?;
    ensure!(t.ty == GgmlType::F32, "{name} is not F32");
    Ok(g.data(t).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}
