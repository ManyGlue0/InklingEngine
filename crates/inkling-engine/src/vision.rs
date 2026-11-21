//! Vision encoder (the `mmproj` GGUF, projector type `inkling2v`) and image preprocessing.
//!
//! ViT with a CLS token, learned position grid (bilinearly resized to the patch grid), 2-D RoPE and layer scale;
//! the CLS row is dropped, 2x2 patches are merged and a 2-layer GELU MLP projects them to the LM width. Mirrors
//! llama.cpp's `mtmd_graph_inkling2v` + `mtmd_image_preprocessor_inkling2v`: the image is padded bottom-right to a
//! multiple of 40 px (patch 20 x merge 2) and only resized when it would exceed the image-token cap.
//!
//! Weights are converted to f16 for the tensor-core GEMMs; norms, biases, layer scales and embeddings stay f32.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use inkling_core::Gguf;
use inkling_cuda::{Arg, DevBuf, Gpu, as_bytes};

use crate::Engine;

/// Image-token cap per image (merged tokens), as in llama.cpp.
pub const MAX_IMAGE_TOKENS: usize = 4096;

struct VitLayer {
    ln1_w: DevBuf,
    ln1_b: DevBuf,
    qkv_w: DevBuf, // f16 [3072][1024]
    qkv_b: DevBuf, // f32 [3072]
    o_w: DevBuf,
    o_b: DevBuf,
    ls1: DevBuf,
    ln2_w: DevBuf,
    ln2_b: DevBuf,
    up_w: DevBuf,
    up_b: DevBuf,
    down_w: DevBuf,
    down_b: DevBuf,
    ls2: DevBuf,
}

/// Vision weights on the GPU.
pub struct Vision {
    layers: Vec<VitLayer>,
    patch_w: DevBuf, // f16 [D][KP] (K = 3 * ps * ps in (c, y, x) order, zero-padded to KP)
    pos: DevBuf,     // f32 [G * G][D]
    cls: DevBuf,
    cls_pos: DevBuf,
    pre_w: DevBuf,
    pre_b: DevBuf,
    mm1: DevBuf, // f16 [4D][4D]
    mm2: DevBuf, // f16 [n_out][4D]
    pub patch: usize,
    pub merge: usize,
    d: usize,
    ff: usize,
    grid: usize,
    kp: usize,
    eps: f32,
    pub n_out: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

/// An RGB8 image.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl Image {
    /// Decode PNG / JPEG / WebP / GIF / BMP bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory(bytes).context("decoding image")?.to_rgb8();
        Ok(Self { width: img.width() as usize, height: img.height() as usize, rgb: img.into_raw() })
    }

    /// 64-bit FNV-1a of the dimensions and pixels (prompt-cache identity).
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in (self.width as u64).to_le_bytes().iter().chain((self.height as u64).to_le_bytes().iter()).chain(self.rgb.iter()) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

/// Image embeddings ready for the LM: `n` rows of the LM width.
#[derive(Clone)]
pub struct ImageEmbed {
    pub n: usize,
    pub data: Arc<Vec<f32>>,
    pub hash: u64,
}

fn f16_bytes(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_bits()).collect()
}

impl Vision {
    pub fn load(gpu: &Gpu, path: &str) -> Result<Self> {
        let g = Gguf::open(path)?;
        ensure!(g.str("general.architecture")? == "clip", "{path}: not an mmproj (clip) GGUF");
        let proj = g.str("clip.vision.projector_type")?;
        ensure!(proj == "inkling2v", "{path}: projector type `{proj}` is not inkling2v");
        let d = g.u64("clip.vision.embedding_length")? as usize;
        let ff = g.u64("clip.vision.feed_forward_length")? as usize;
        let n_layer = g.u64("clip.vision.block_count")? as usize;
        let heads = g.u64("clip.vision.attention.head_count")? as usize;
        let patch = g.u64("clip.vision.patch_size")? as usize;
        let merge = g.u64("clip.vision.spatial_merge_size")? as usize;
        let eps = g.f64("clip.vision.attention.layer_norm_eps").or_else(|_| g.f64("clip.vision.attention.layer_norm_epsilon"))? as f32;
        ensure!(d == 1024 && heads == 16 && merge == 2 && ff % 128 == 0, "unsupported vision dimensions");
        let arr3 = |k: &str| -> Result<[f32; 3]> {
            let a = g.get(k).and_then(|v| v.as_arr()).with_context(|| format!("{k} missing"))?;
            ensure!(a.len() == 3, "{k}: expected 3 values");
            Ok([a[0].as_f64().unwrap_or(0.5) as f32, a[1].as_f64().unwrap_or(0.5) as f32, a[2].as_f64().unwrap_or(0.5) as f32])
        };
        let raw = |name: &str| -> Result<&[u8]> {
            let t = g.require(name)?;
            ensure!(matches!(t.ty, inkling_core::gguf::GgmlType::F32), "{name}: expected an F32 mmproj tensor");
            Ok(g.data(t))
        };
        let f32b = |name: &str| -> Result<DevBuf> { gpu.upload(raw(name)?) };
        let f16b = |name: &str| -> Result<DevBuf> { gpu.upload(as_bytes(&f16_bytes(raw(name)?))) };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let n = |s: &str| format!("v.blk.{i}.{s}");
            let mut qkv: Vec<u16> = Vec::with_capacity(3 * d * d);
            let mut qkv_b: Vec<u8> = Vec::with_capacity(3 * d * 4);
            for p in ["attn_q", "attn_k", "attn_v"] {
                qkv.extend(f16_bytes(raw(&n(&format!("{p}.weight")))?));
                qkv_b.extend_from_slice(raw(&n(&format!("{p}.bias")))?);
            }
            layers.push(VitLayer {
                ln1_w: f32b(&n("ln1.weight"))?,
                ln1_b: f32b(&n("ln1.bias"))?,
                qkv_w: gpu.upload(as_bytes(&qkv))?,
                qkv_b: gpu.upload(&qkv_b)?,
                o_w: f16b(&n("attn_out.weight"))?,
                o_b: f32b(&n("attn_out.bias"))?,
                ls1: f32b(&n("ls1.weight"))?,
                ln2_w: f32b(&n("ln2.weight"))?,
                ln2_b: f32b(&n("ln2.bias"))?,
                up_w: f16b(&n("ffn_up.weight"))?,
                up_b: f32b(&n("ffn_up.bias"))?,
                down_w: f16b(&n("ffn_down.weight"))?,
                down_b: f32b(&n("ffn_down.bias"))?,
                ls2: f32b(&n("ls2.weight"))?,
            });
        }
        // patch embedding: [D][3 * ps * ps] padded to a multiple of 32
        let k = 3 * patch * patch;
        let kp = k.div_ceil(32) * 32;
        let pw = f16_bytes(raw("v.patch_embd.weight")?);
        ensure!(pw.len() == d * k, "patch embedding shape");
        let mut pwp = vec![0u16; d * kp];
        for r in 0..d {
            pwp[r * kp..r * kp + k].copy_from_slice(&pw[r * k..(r + 1) * k]);
        }
        let pos_t = g.require("v.position_embd.weight")?;
        let n_pos = pos_t.n_elements() as usize / d;
        let grid = (n_pos as f64).sqrt() as usize;
        ensure!(grid * grid == n_pos, "position grid is not square");
        let mm2_t = g.require("mm.2.weight")?;
        let n_out = mm2_t.n_elements() as usize / (4 * d);
        ensure!(n_out % 128 == 0, "projector width");
        Ok(Self {
            layers,
            patch_w: gpu.upload(as_bytes(&pwp))?,
            pos: f32b("v.position_embd.weight")?,
            cls: f32b("v.class_embd")?,
            cls_pos: f32b("v.class_position_embd.weight")?,
            pre_w: f32b("v.pre_ln.weight")?,
            pre_b: f32b("v.pre_ln.bias")?,
            mm1: f16b("mm.1.weight")?,
            mm2: f16b("mm.2.weight")?,
            patch,
            merge,
            d,
            ff,
            grid,
            kp,
            eps,
            n_out,
            mean: arr3("clip.vision.image_mean")?,
            std: arr3("clip.vision.image_std")?,
        })
    }

    pub fn grid_for(&self, w: usize, h: usize) -> (usize, usize, usize, usize) {
        image_grid(self.patch, self.merge, w, h)
    }

    /// Number of LM tokens an image of this size becomes.
    pub fn n_tokens(&self, w: usize, h: usize) -> usize {
        image_tokens(self.patch, self.merge, w, h)
    }
}

/// (scaled width, scaled height, patch columns, patch rows) for a `w` x `h` image: padded to a multiple of
/// patch * merge, downscaled (aspect kept) only when the image-token cap would be exceeded.
pub fn image_grid(patch: usize, merge: usize, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let align = patch * merge;
    let al = |v: usize| v.div_ceil(align) * align;
    let max_px = MAX_IMAGE_TOKENS * align * align;
    let (mut sw, mut sh) = (w.max(1), h.max(1));
    if al(sw) * al(sh) > max_px {
        let s = (max_px as f64 / (sw * sh) as f64).sqrt();
        sw = (((sw as f64 * s) as usize) / align * align).max(align);
        sh = (((sh as f64 * s) as usize) / align * align).max(align);
        while al(sw) * al(sh) > max_px {
            if sw >= sh { sw -= align } else { sh -= align }
        }
    }
    (sw, sh, al(sw) / patch, al(sh) / patch)
}

/// LM tokens of a `w` x `h` image.
pub fn image_tokens(patch: usize, merge: usize, w: usize, h: usize) -> usize {
    let (_, _, wp, hp) = image_grid(patch, merge, w, h);
    (wp / merge) * (hp / merge)
}

/// Bilinear resize (half-pixel centers) of an RGB8 image.
fn resize(img: &Image, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let sx = img.width as f32 / w as f32;
    let sy = img.height as f32 / h as f32;
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy as usize).min(img.height - 1);
        let y1 = (y0 + 1).min(img.height - 1);
        let dy = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx as usize).min(img.width - 1);
            let x1 = (x0 + 1).min(img.width - 1);
            let dx = fx - x0 as f32;
            for c in 0..3 {
                let p = |xx: usize, yy: usize| img.rgb[(yy * img.width + xx) * 3 + c] as f32;
                let v = p(x0, y0) * (1.0 - dx) * (1.0 - dy) + p(x1, y0) * dx * (1.0 - dy) + p(x0, y1) * (1.0 - dx) * dy + p(x1, y1) * dx * dy;
                out[(y * w + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

impl Engine {
    /// Load the vision encoder (mmproj GGUF) next to the LM.
    pub fn load_vision(&mut self, mmproj: &str) -> Result<()> {
        // f16 weights (half the F32 file) + scratch for the largest image (~410 MiB at the 4,096-token cap)
        let need = std::fs::metadata(mmproj).with_context(|| format!("opening {mmproj}"))?.len() as usize / 2 + (480 << 20);
        let free = self.gpu.free_vram()?;
        crate::check_vram(free, need, "the vision encoder")?;
        let v = Vision::load(&self.gpu, mmproj)?;
        ensure!(v.n_out == self.cfg.n_embd, "mmproj projects to {} but the LM width is {}", v.n_out, self.cfg.n_embd);
        self.vision = Some(v);
        Ok(())
    }

    pub fn vision(&self) -> Option<&Vision> {
        self.vision.as_ref()
    }

    /// Encode one image into LM embeddings (row-major [n][n_embd]).
    pub fn encode_image(&self, img: &Image) -> Result<ImageEmbed> {
        let v = self.vision.as_ref().context("no vision encoder loaded (pass --mmproj)")?;
        ensure!(img.width > 0 && img.height > 0 && img.rgb.len() == img.width * img.height * 3, "bad image buffer");
        let (sw, sh, wp, hp) = v.grid_for(img.width, img.height);
        let resized;
        let (src, iw) = if (sw, sh) != (img.width, img.height) {
            resized = resize(img, sw, sh);
            (&resized[..], sw)
        } else {
            (&img.rgb[..], img.width)
        };
        // normalized pixels, padded bottom-right with -1 before normalization (as the reference does)
        let norm = |c: usize, val: f32| (val - v.mean[c] * 255.0) / (v.std[c] * 255.0);
        let pad = [norm(0, -1.0), norm(1, -1.0), norm(2, -1.0)];
        let pix: Vec<f32> = (0..sw * sh * 3).map(|i| norm(i % 3, src[(i / 3 / sw * iw + i / 3 % sw) * 3 + i % 3] as f32)).collect();
        let mut e = self.encode_pixels(sw, sh, wp, hp, &pix, pad)?;
        e.hash = img.hash();
        Ok(e)
    }

    /// Encode already-normalized pixels (HWC f32, `w` x `h`) on a `wp` x `hp` patch grid (pixels outside the
    /// image take `pad`).
    pub fn encode_pixels(&self, w: usize, h: usize, wp: usize, hp: usize, pix: &[f32], pad: [f32; 3]) -> Result<ImageEmbed> {
        let v = self.vision.as_ref().context("no vision encoder loaded (pass --mmproj)")?;
        ensure!(pix.len() == w * h * 3 && wp % 2 == 0 && hp % 2 == 0 && wp * v.patch >= w && hp * v.patch >= h, "bad pixel grid");
        let (d, ps, kp) = (v.d, v.patch, v.kp);
        let np = wp * hp;
        let n = np + 1;
        let m = np / 4;
        let mut a = vec![0u16; np * kp];
        for py in 0..hp {
            for px in 0..wp {
                let row = &mut a[(py * wp + px) * kp..][..kp];
                for c in 0..3 {
                    for ky in 0..ps {
                        let y = py * ps + ky;
                        for kx in 0..ps {
                            let x = px * ps + kx;
                            let val = if x < w && y < h { pix[(y * w + x) * 3 + c] } else { pad[c] };
                            row[c * ps * ps + ky * ps + kx] = half::f16::from_f32(val).to_bits();
                        }
                    }
                }
            }
        }
        let gpu = &self.gpu;
        let a_buf = gpu.upload(as_bytes(&a))?;
        let res = gpu.alloc(n * d * 4)?;
        let xn = gpu.alloc(n * d * 2)?;
        let qkv = gpu.alloc(n * 3 * d * 2)?;
        let hid = gpu.alloc(n * v.ff.max(4 * d) * 2)?;
        let out = gpu.alloc(m * v.n_out * 4)?;
        let gemm = |x: u64, ldx: usize, w: &DevBuf, bias: u64, scale: u64, o: u64, ldo: usize, rows: usize, n_out: usize, k: usize, mode: i32| {
            gpu.launch("ink_vit_gemm", ((n_out / 128) as u32, rows.div_ceil(128) as u32), 256, &[
                Arg::Ptr(x), Arg::I32(ldx as i32), Arg::Ptr(w.ptr), Arg::Ptr(bias), Arg::Ptr(scale), Arg::Ptr(o), Arg::I32(ldo as i32),
                Arg::I32(rows as i32), Arg::I32(k as i32), Arg::I32(mode),
            ])
        };
        let ln = |w: &DevBuf, b: &DevBuf, o16: u64, o32: u64| {
            gpu.launch("ink_vit_ln", (n as u32, 1), 256, &[Arg::Ptr(res.ptr), Arg::Ptr(w.ptr), Arg::Ptr(b.ptr), Arg::Ptr(o16), Arg::Ptr(o32), Arg::I32(d as i32), Arg::F32(v.eps)])
        };
        // patches -> rows 1.., then position / CLS embeddings, pre-LN (in place)
        gemm(a_buf.ptr, kp, &v.patch_w, 0, 0, res.at(d * 4), d, np, d, kp, 3)?;
        gpu.launch("ink_vit_embed", (n as u32, 1), 256, &[
            Arg::Ptr(res.ptr), Arg::Ptr(v.pos.ptr), Arg::I32(v.grid as i32), Arg::Ptr(v.cls.ptr), Arg::Ptr(v.cls_pos.ptr), Arg::I32(wp as i32), Arg::I32(hp as i32), Arg::I32(d as i32),
        ])?;
        ln(&v.pre_w, &v.pre_b, 0, res.ptr)?;
        for l in &v.layers {
            ln(&l.ln1_w, &l.ln1_b, xn.ptr, 0)?;
            gemm(xn.ptr, d, &l.qkv_w, l.qkv_b.ptr, 0, qkv.ptr, 3 * d, n, 3 * d, d, 0)?;
            gpu.launch("ink_vit_rope", (n as u32, 1), 256, &[Arg::Ptr(qkv.ptr), Arg::I32(wp as i32), Arg::F32(10000.0)])?;
            gpu.launch("ink_vit_attn", (n.div_ceil(64) as u32, 16), 128, &[Arg::Ptr(qkv.ptr), Arg::Ptr(xn.ptr), Arg::I32(n as i32)])?;
            gemm(xn.ptr, d, &l.o_w, l.o_b.ptr, l.ls1.ptr, res.ptr, d, n, d, d, 2)?;
            ln(&l.ln2_w, &l.ln2_b, xn.ptr, 0)?;
            gemm(xn.ptr, d, &l.up_w, l.up_b.ptr, 0, hid.ptr, v.ff, n, v.ff, d, 1)?;
            gemm(hid.ptr, v.ff, &l.down_w, l.down_b.ptr, l.ls2.ptr, res.ptr, d, n, d, v.ff, 2)?;
        }
        // drop CLS, 2x2 merge, projector MLP
        gpu.launch("ink_vit_merge", (m as u32, 1), 256, &[Arg::Ptr(res.ptr), Arg::Ptr(hid.ptr), Arg::I32(wp as i32), Arg::I32(d as i32)])?;
        gemm(hid.ptr, 4 * d, &v.mm1, 0, 0, qkv.ptr, 4 * d, m, 4 * d, 4 * d, 1)?;
        gemm(qkv.ptr, 4 * d, &v.mm2, 0, 0, out.ptr, v.n_out, m, v.n_out, 4 * d, 3)?;
        let data = gpu.read_f32(out.ptr, m * v.n_out)?;
        Ok(ImageEmbed { n: m, data: Arc::new(data), hash: 0 })
    }
}
