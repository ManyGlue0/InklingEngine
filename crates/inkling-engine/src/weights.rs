//! Upload GGUF tensors to the GPU (large matrices repacked, see repack.rs) and resolve per-layer pointer tables.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use inkling_core::{Config, GgmlType, Gguf};
use inkling_cuda::{DevBuf, Gpu, HostMapped};

use crate::cpu_moe::{CpuExp, CpuLayer};
use crate::repack::{self, Soa};

/// Expert weight format. Q8_0 and NVFP4 are repacked for the tuned kernels; other formats stay in their GGUF
/// block layout and run on the generic kernels (codecs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFmt {
    Q8_0,
    Nvfp4,
    /// ggml type id, native layout
    Native(u32),
}

impl ExpertFmt {
    /// code passed to kernels (ExpW.fmt): 0 = repacked Q8_0, 1 = repacked NVFP4, else the ggml type id
    pub fn code(self) -> i32 {
        match self {
            ExpertFmt::Q8_0 => 0,
            ExpertFmt::Nvfp4 => 1,
            ExpertFmt::Native(id) => id as i32,
        }
    }
    /// runs on the tuned (repacked) kernels
    pub fn tuned(self) -> bool {
        !matches!(self, ExpertFmt::Native(_))
    }
}

/// Host memory for offloaded experts: pinned and GPU-mapped when the driver grants it (fast asynchronous copies
/// for prompt processing), else ordinary pageable memory (the CPU path is the same; copies are slower).
pub enum HostMem {
    Pinned(HostMapped),
    Plain(Vec<u8>),
}

impl HostMem {
    fn alloc(gpu: &Gpu, bytes: usize, what: &str) -> Result<Self> {
        // INKLING_NO_PIN=1 skips pinning (tests the fallback)
        let pinned = if std::env::var_os("INKLING_NO_PIN").is_some() { Err(anyhow::anyhow!("pinning disabled")) } else { gpu.alloc_host_mapped_uninit(bytes) };
        match pinned {
            Ok(h) => Ok(HostMem::Pinned(h)),
            Err(_) => {
                let mut v = Vec::new();
                v.try_reserve_exact(bytes).map_err(|_| anyhow::anyhow!("not enough system RAM for {} MiB of {what}", bytes >> 20))?;
                v.resize(bytes, 0);
                Ok(HostMem::Plain(v))
            }
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            HostMem::Pinned(h) => h.as_mut_slice(),
            HostMem::Plain(v) => v.as_mut_slice(),
        }
    }
    pub fn host_ptr(&mut self) -> *mut u8 {
        match self {
            HostMem::Pinned(h) => h.host_ptr(),
            HostMem::Plain(v) => v.as_mut_ptr(),
        }
    }
    /// device address of the mapping (0 for pageable memory: kernels cannot read it)
    pub fn dev(&self) -> u64 {
        match self {
            HostMem::Pinned(h) => h.dev,
            HostMem::Plain(_) => 0,
        }
    }
    pub fn pinned(&self) -> bool {
        matches!(self, HostMem::Pinned(_))
    }
}

/// Slack after natively stored weights: the generic kernels' unaligned 32-bit loads may read past the last block.
pub const NATIVE_PAD: usize = 64;

/// Dense weight matrix: fmt 0 = repacked Q8_0 (qs int8 [rows][K], d f16 [rows][K/32]); otherwise a ggml type id
/// with the matrix in its GGUF block layout at qs (d unused), run by the generic kernels (quant_gen.cu).
#[derive(Debug, Clone, Copy, Default)]
pub struct Q8M {
    pub qs: u64,
    pub d: u64,
    pub fmt: i32,
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
    /// q (H*dh rows) | rel (H*R rows) | gate (H*dh rows): one fused matrix when all three are Q8_0 (nqrg = 1),
    /// else the three matrices (nqrg = 3)
    pub wqrg: [Q8M; 3],
    pub nqrg: usize,
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
    pub host_bufs: Vec<HostMem>,
    pub token_embd: u64,
    /// ggml type id of the embedding table (8 = Q8_0)
    pub token_embd_fmt: i32,
    pub token_embd_norm: u64,
    pub output: Q8M,
    pub output_norm: u64,
    pub output_res_query: u64,
    pub layers: Vec<LayerW>,
    /// CPU-offloaded MoE layers (expert planes in `host_bufs`), by physical layer
    pub cpu: Vec<Option<CpuLayer>>,
    pub bytes: usize,
}

pub(crate) struct Loader<'a> {
    pub(crate) gpu: &'a Gpu,
    pub(crate) g: &'a Gguf,
    pub(crate) bufs: HashMap<String, DevBuf>,
    pub(crate) bytes: usize,
}

impl Loader<'_> {
    /// Upload with NATIVE_PAD zero bytes of slack.
    fn put_padded(&mut self, key: String, data: &[u8]) -> Result<u64> {
        let mut v = Vec::with_capacity(data.len() + NATIVE_PAD);
        v.extend_from_slice(data);
        v.resize(data.len() + NATIVE_PAD, 0);
        self.put(key, &v)
    }

    fn put(&mut self, key: String, data: &[u8]) -> Result<u64> {
        let b = self.gpu.upload(data).with_context(|| format!("uploading {key}"))?;
        let p = b.ptr;
        self.bytes += b.bytes;
        self.bufs.insert(key, b);
        Ok(p)
    }

    /// Small tensors (F32/BF16) uploaded as-is.
    pub(crate) fn raw(&mut self, name: &str) -> Result<u64> {
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

    /// Small matrix dequantized on the host (F32, BF16, Q8_0 or a native codec) and uploaded as F16 or F32.
    fn small(&mut self, name: &str, f16: bool) -> Result<u64> {
        let t = self.g.require(name)?;
        if t.ty == GgmlType::F32 {
            return if f16 { self.f16(name) } else { self.raw(name) };
        }
        let v = tensor_f32(self.g, name)?;
        if f16 {
            let h: Vec<u16> = v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            self.put(format!("{name}#f16"), inkling_cuda::as_bytes(&h))
        } else {
            self.put(format!("{name}#f32"), inkling_cuda::as_bytes(&v))
        }
    }

    fn raw_opt(&mut self, name: &str) -> Result<u64> {
        if self.g.tensor(name).is_some() { self.raw(name) } else { Ok(0) }
    }

    /// One or more Q8_0 2-D tensors with the same K, row-concatenated and repacked.
    pub(crate) fn q8(&mut self, names: &[String]) -> Result<Q8M> {
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
        Ok(Q8M { qs: self.put(format!("{key}#qs"), &soa.qs)?, d: self.put(format!("{key}#d"), &soa.sc)?, fmt: 0 })
    }

    /// A 2-D weight matrix in any supported format.
    fn mat(&mut self, name: &str) -> Result<Q8M> {
        let t = self.g.require(name)?;
        ensure!(t.dims.len() == 2, "{name}: expected a 2-D matrix");
        match t.ty {
            GgmlType::Q8_0 => self.q8(&[name.to_owned()]),
            ty if crate::codecs::supported(ty.id()) => {
                Ok(Q8M { qs: self.put_padded(format!("{name}#native"), self.g.data(t))?, d: 0, fmt: ty.id() as i32 })
            }
            ty => bail!("{name}: matrix type {} is not supported yet", ty.name()),
        }
    }

    /// Matrices that share an input: fused into one when all are Q8_0, else loaded one by one.
    fn mats3(&mut self, names: [String; 3]) -> Result<([Q8M; 3], usize)> {
        if names.iter().all(|n| self.g.tensor(n).is_some_and(|t| t.ty == GgmlType::Q8_0)) {
            Ok(([self.q8(&names)?, Q8M::default(), Q8M::default()], 1))
        } else {
            Ok(([self.mat(&names[0])?, self.mat(&names[1])?, self.mat(&names[2])?], 3))
        }
    }

    fn experts(&mut self, name: &str, scale: &str) -> Result<ExpM> {
        let t = self.g.require(name)?;
        ensure!(t.dims.len() == 3, "{name}: expected 3-D expert tensor");
        let (k, rows, e) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
        let (soa, fmt) = match t.ty {
            GgmlType::NVFP4 => (repack::nvfp4_soa(self.g.data(t), rows * e, k)?, ExpertFmt::Nvfp4),
            GgmlType::Q8_0 => (repack::q8_soa(self.g.data(t), rows * e, k)?, ExpertFmt::Q8_0),
            ty if crate::codecs::supported(ty.id()) => {
                let _ = (rows, e, k);
                let qs = self.put_padded(format!("{name}#native"), self.g.data(t))?;
                let scale = self.raw_opt(scale)?;
                return Ok(ExpM { qs, sc: 0, scale, fmt: ExpertFmt::Native(ty.id()) });
            }
            ty => bail!("{name}: expert type {} is not supported yet", ty.name()),
        };
        let qs = self.put(format!("{name}#qs"), &soa.qs)?;
        let sc = self.put(format!("{name}#sc"), &soa.sc)?;
        let scale = self.raw_opt(scale)?;
        Ok(ExpM { qs, sc, scale, fmt })
    }

    /// Expert tensor repacked into pinned host memory (CPU offload). The returned ExpM points at the mapped planes
    /// (a GPU kernel can still read them over PCIe); the per-expert scales are also uploaded.
    fn experts_host(&mut self, name: &str, scale: &str, host: &mut Vec<HostMem>) -> Result<(ExpM, CpuExp, Option<Vec<f32>>)> {
        let t = self.g.require(name)?;
        ensure!(t.dims.len() == 3, "{name}: expected 3-D expert tensor");
        let (k, rows, e) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
        let fmt = match t.ty {
            GgmlType::NVFP4 => ExpertFmt::Nvfp4,
            GgmlType::Q8_0 => ExpertFmt::Q8_0,
            ty if crate::codecs::supported(ty.id()) => ExpertFmt::Native(ty.id()),
            ty => bail!("{name}: expert type {} is not supported yet", ty.name()),
        };
        if let ExpertFmt::Native(_) = fmt {
            // native layout: one plane, a pinned copy of the file's blocks (plus slack)
            let mut hq = HostMem::alloc(self.gpu, t.n_bytes + NATIVE_PAD, name)?;
            hq.as_mut_slice()[..t.n_bytes].copy_from_slice(self.g.data(t));
            hq.as_mut_slice()[t.n_bytes..].fill(0);
            let scale_dev = self.raw_opt(scale)?;
            let scale_host = if self.g.tensor(scale).is_some() { Some(f32_rows(self.g, scale)?) } else { None };
            let m = ExpM { qs: hq.dev(), sc: 0, scale: scale_dev, fmt };
            let c = CpuExp { qs: hq.host_ptr(), sc: std::ptr::null(), fmt, rows, k };
            host.push(hq);
            return Ok((m, c, scale_host));
        }
        let qb = if fmt == ExpertFmt::Nvfp4 { rows * e * k / 2 } else { rows * e * k };
        let mut hq = HostMem::alloc(self.gpu, qb, name)?;
        let mut hs = HostMem::alloc(self.gpu, rows * e * k / 16, name)?;
        match fmt {
            ExpertFmt::Nvfp4 => repack::nvfp4_soa_into(self.g.data(t), rows * e, k, hq.as_mut_slice(), hs.as_mut_slice())?,
            ExpertFmt::Q8_0 => repack::q8_soa_into(self.g.data(t), rows * e, k, hq.as_mut_slice(), hs.as_mut_slice())?,
            ExpertFmt::Native(_) => unreachable!("handled above"),
        }
        let scale_dev = self.raw_opt(scale)?;
        let scale_host = if self.g.tensor(scale).is_some() { Some(f32_rows(self.g, scale)?) } else { None };
        let m = ExpM { qs: hq.dev(), sc: hs.dev(), scale: scale_dev, fmt };
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
        // merged upstream layout: per-layer `loop_msg_query` and `loop_{q,k,v,ffn}_ctrl_*`; legacy: two global query
        // tables and `loop_{attn,k,v,mlp}_*`
        let merged = g.tensor(&format!("blk.{}.loop_msg_query.weight", cfg.core_start)).is_some();
        let (res, kv) = if merged {
            (Vec::new(), Vec::new())
        } else {
            (f32_rows(g, "loop_res_query.weight")?, f32_rows(g, "loop_kv_res_query.weight")?)
        };

        let mut layers = Vec::with_capacity(cfg.n_phys);
        for i in 0..cfg.n_phys {
            let n = |s: &str| format!("blk.{i}.{s}");
            let lp = if (cfg.core_start..cfg.core_end).contains(&i) {
                let f16 = std::env::var_os("INKLING_LOOP_F32").is_none();
                let mut lw = LoopW { f16, down: [0; 4], gain: [0; 4], forget: [0; 4], msg_query: 0 };
                let names = if merged { ["q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"] } else { ["attn", "k", "v", "mlp"] };
                for (c, nm) in names.iter().enumerate() {
                    for (dst, kind) in [(&mut lw.down[c], "down"), (&mut lw.gain[c], "gain"), (&mut lw.forget[c], "forget")] {
                        let name = n(&format!("loop_{nm}_{kind}.weight"));
                        *dst = l.small(&name, f16)?;
                    }
                }
                // message queries of core layer ci (q, k, v, ffn); legacy rows [res[2ci], kv[2ci], kv[2ci+1], res[2ci+1]]
                let q = if merged {
                    let q = f32_rows(g, &n("loop_msg_query.weight"))?;
                    ensure!(q.len() == 4 * d, "{}: expected {d} x 4", n("loop_msg_query.weight"));
                    q
                } else {
                    let ci = i - cfg.core_start;
                    let mut q = Vec::with_capacity(4 * d);
                    q.extend_from_slice(&res[2 * ci * d..(2 * ci + 1) * d]);
                    q.extend_from_slice(&kv[2 * ci * d..(2 * ci + 2) * d]);
                    q.extend_from_slice(&res[(2 * ci + 1) * d..(2 * ci + 2) * d]);
                    q
                };
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
            let (wqrg, nqrg) = l.mats3([n("attn_q.weight"), n("attn_rel.weight"), n("attn_gate.weight")])?;
            layers.push(LayerW {
                is_swa: cfg.is_swa[i],
                attn_norm: l.raw_opt(&n("attn_norm.weight"))?,
                q_norm: l.raw_opt(&n("attn_q_inp_norm.weight"))?,
                k_norm: l.raw_opt(&n("attn_k_inp_norm.weight"))?,
                v_norm: l.raw_opt(&n("attn_v_inp_norm.weight"))?,
                wqrg,
                nqrg,
                wk: l.mat(&n("attn_k.weight"))?,
                wv: l.mat(&n("attn_v.weight"))?,
                wo: l.mat(&n("attn_output.weight"))?,
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
            // this layer is on the GPU / in pinned memory: let its file pages go (see Gguf::release)
            let prefix = format!("blk.{i}.");
            for t in g.tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
                g.release(t);
            }
        }
        // the token embedding table is only gathered (one row per token): keep it in host memory, readable by the GPU
        // over PCIe, and save ~0.5 GB of VRAM (INKLING_EMBED_GPU=1 keeps it on the GPU)
        let te = g.require("token_embd.weight")?;
        let token_embd_fmt = te.ty.id() as i32;
        ensure!(te.ty == GgmlType::Q8_0 || crate::codecs::supported(te.ty.id()), "token_embd: type {} is not supported yet", te.ty.name());
        // the GPU reads it over PCIe, so it must be pinned; if the driver refuses, keep it in VRAM instead
        let pinned_embd = if embed_on_host() { gpu.alloc_host_mapped(te.n_bytes + NATIVE_PAD).ok() } else { None };
        let token_embd = if let Some(mut h) = pinned_embd {
            h.as_mut_slice()[..te.n_bytes].copy_from_slice(g.data(te));
            let p = h.dev;
            host_bufs.push(HostMem::Pinned(h));
            p
        } else {
            l.put_padded("token_embd.weight".to_owned(), g.data(te))?
        };
        let token_embd_norm = l.raw("token_embd_norm.weight")?;
        let output = l.mat("output.weight")?;
        let output_norm = l.raw("output_norm.weight")?;
        let output_res_query = l.raw("output_res_query.weight")?;
        for t in &g.tensors {
            g.release(t);
        }
        Ok(Self { host_bufs, token_embd, token_embd_fmt, token_embd_norm, output, output_norm, output_res_query, layers, cpu, bytes: l.bytes, bufs: l.bufs })
    }
}

/// Whether the token embedding table lives in host memory (default) instead of VRAM.
pub fn embed_on_host() -> bool {
    std::env::var_os("INKLING_EMBED_GPU").is_none()
}

/// Any 2-D tensor of a small, dequantizable type as F32 values in row order.
pub(crate) fn tensor_f32(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.require(name)?;
    let data = g.data(t);
    let k = t.dims[0] as usize;
    Ok(match t.ty {
        GgmlType::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        GgmlType::BF16 => data.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        GgmlType::Q8_0 => data
            .chunks_exact(34)
            .flat_map(|b| {
                let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                b[2..].iter().map(move |&q| d * q as i8 as f32)
            })
            .collect(),
        ty if crate::codecs::supported(ty.id()) => {
            let rb = crate::codecs::row_bytes(ty.id(), k);
            data.chunks_exact(rb).flat_map(|r| crate::codecs::dequant_row(ty.id(), r, k)).collect()
        }
        ty => bail!("{name}: type {} is not supported here", ty.name()),
    })
}

pub(crate) fn f32_rows(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.require(name)?;
    ensure!(t.ty == GgmlType::F32, "{name} is not F32");
    Ok(g.data(t).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}
