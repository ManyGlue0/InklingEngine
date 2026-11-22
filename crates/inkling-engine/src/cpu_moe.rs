//! CPU expert FFN for MoE layers offloaded to system RAM (decode path).
//!
//! The expert weights live in pinned host memory in the GPU's repacked layout (repack.rs), so the same copy also
//! streams to the GPU for batched prompt processing:
//!   NVFP4: qs [E][rows][K/2] + sc [E][rows][K/16]. Per 64-value block: 32 qs bytes, sub-block s uses bytes
//!          8s..8s+7, byte j holding value 16s + j (low nibble) and 16s + j + 8 (high nibble); 4 UE4M3 scales (one
//!          per 16 values). Plus an F32 scale per expert.
//!   Q8_0:  qs int8 [E][rows][K] + d f16 [E][rows][K/32].
//! The math matches the GPU kernels (experts.cu): q8_1 activations, E2M1 values doubled to integers with a halved
//! block scale, int8 dot products, SiTU gating, the hidden vector quantized to int8 per 16 features, and the
//! weighted sum over the selected experts. Work runs on a persistent thread pool in two phases (gate_up + hidden,
//! then down), AVX2 when available with a scalar fallback.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::weights::ExpertFmt;

/// One offloaded expert tensor [E][rows][K] (repacked planes in host memory).
#[derive(Clone, Copy)]
pub struct CpuExp {
    pub qs: *const u8,
    pub sc: *const u8,
    pub fmt: ExpertFmt,
    pub rows: usize,
    pub k: usize,
}

unsafe impl Send for CpuExp {}
unsafe impl Sync for CpuExp {}

impl CpuExp {
    /// bytes per row of the (qs, sc) planes
    pub fn row_bytes(&self) -> (usize, usize) {
        match self.fmt {
            ExpertFmt::Nvfp4 => (self.k / 2, self.k / 16),
            ExpertFmt::Q8_0 => (self.k, self.k / 16),
            ExpertFmt::Native(id) => (crate::codecs::row_bytes(id, self.k), 0),
        }
    }
    /// (qs, sc) of row r of expert e
    fn row(&self, e: usize, r: usize) -> (*const u8, *const u8) {
        let (q, s) = self.row_bytes();
        let i = e * self.rows + r;
        unsafe { (self.qs.add(i * q), self.sc.add(i * s)) }
    }
}

/// An offloaded layer: gate_up [E][2F][D], down [E][D][F], optional per-expert scales.
pub struct CpuLayer {
    pub gate_up: CpuExp,
    pub down: CpuExp,
    pub gu_scale: Option<Vec<f32>>,
    pub down_scale: Option<Vec<f32>>,
}

/// Activations prepared for one weight format: int8 values in the order the kernel consumes them and one f32
/// scale per int32 lane of each block (8 lanes per block).
struct Acts {
    q: Vec<i8>,
    s8: Vec<f32>,
    /// native formats: activation scale and sum of the int8 values per 16
    s16: Vec<f32>,
    sum16: Vec<i32>,
}

impl Acts {
    fn with_capacity(k: usize) -> Self {
        Acts { q: Vec::with_capacity(k), s8: Vec::with_capacity(k / 4), s16: Vec::with_capacity(k / 16), sum16: Vec::with_capacity(k / 16) }
    }
}

/// `x` (int8, quantized in groups of `g` values with scales `xs`) arranged for rows of format `fmt` and length k.
/// AVX2 NVFP4 path: a UE4M3 byte shifted left by 20 is an f32 of value 2^(e-127) (1 + m/8), 2^-119 times the
/// kernels' (halved) scale for normal numbers (load time guarantees no subnormal, zero or negative scales); the
/// activation lane scales carry the 2^119.
const FP4_FAST_MUL: f32 = 6.646139978924579e35; // 2^119

/// `x` (int8, quantized in groups of `g` values with scales `xs`) arranged for rows of format `fmt` and length k;
/// lane scales multiplied by `mul`.
fn prep_acts_mul(x: &[i8], xs: &[f32], g: usize, fmt: ExpertFmt, k: usize, mul: f32, out: &mut Acts) {
    prep_acts(x, xs, g, fmt, k, out);
    if mul != 1.0 {
        out.s8.iter_mut().for_each(|v| *v *= mul);
    }
}

fn prep_acts(x: &[i8], xs: &[f32], g: usize, fmt: ExpertFmt, k: usize, out: &mut Acts) {
    out.q.clear();
    out.s8.clear();
    out.s16.clear();
    out.sum16.clear();
    match fmt {
        ExpertFmt::Native(_) => {
            out.q.extend_from_slice(&x[..k]);
            for i in 0..k / 16 {
                out.s16.push(xs[16 * i / g]);
                out.sum16.push(x[16 * i..16 * i + 16].iter().map(|&v| v as i32).sum());
            }
        }
        ExpertFmt::Nvfp4 => {
            for b in 0..k / 64 {
                let xb = &x[64 * b..64 * b + 64];
                // low nibbles: byte p -> value 16 (p / 8) + p % 8; high nibbles: + 8
                for half in 0..2 {
                    for p in 0..32 {
                        out.q.push(xb[16 * (p / 8) + p % 8 + 8 * half]);
                    }
                }
                // int32 lanes 2s, 2s+1 belong to sub-block s
                for lane in 0..8 {
                    out.s8.push(xs[(64 * b + 16 * (lane / 2)) / g]);
                }
            }
        }
        ExpertFmt::Q8_0 => {
            out.q.extend_from_slice(&x[..k]);
            for b in 0..k / 32 {
                for lane in 0..8 {
                    out.s8.push(xs[(32 * b + 4 * lane) / g]);
                }
            }
        }
    }
}

#[inline]
fn ue4m3(x: u8) -> f32 {
    // as ue4m3_fast on the GPU: 2^(e-8) (1 + m/8) (half the real value: E2M1 values are doubled), subnormals m/1024
    let (e, m) = ((x >> 3) & 15, x & 7);
    if x == 0 || x == 0x7F {
        0.0
    } else if e == 0 {
        m as f32 / 1024.0
    } else {
        f32::from_bits(((e as u32 + 119) << 23) | ((m as u32) << 20))
    }
}

const E2M1: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Native formats: decode each group of 32 (codecs) and dot it with the natural-order activations.
fn dot_native(id: u32, w: CpuExp, row: *const u8, a: &Acts) -> f32 {
    let rb = crate::codecs::row_bytes(id, w.k);
    let row = unsafe { std::slice::from_raw_parts(row, rb) };
    let mut acc = 0f32;
    for g in 0..w.k / 32 {
        let q = crate::codecs::g32(id, row, g);
        let x = &a.q[32 * g..32 * g + 32];
        let (mut lo, mut hi) = (0i32, 0i32);
        for i in 0..16 {
            lo += q.q[i] as i32 * x[i] as i32;
            hi += q.q[16 + i] as i32 * x[16 + i] as i32;
        }
        acc += a.s16[2 * g] * (q.s0 * lo as f32 - q.m * a.sum16[2 * g] as f32) + a.s16[2 * g + 1] * (q.s1 * hi as f32 - q.m * a.sum16[2 * g + 1] as f32);
    }
    acc
}

fn dot_scalar(w: CpuExp, row: (*const u8, *const u8), a: &Acts) -> f32 {
    let mut acc = 0f32;
    match w.fmt {
        ExpertFmt::Native(id) => return dot_native(id, w, row.0, a),
        ExpertFmt::Nvfp4 => {
            for b in 0..w.k / 64 {
                let qs = unsafe { std::slice::from_raw_parts(row.0.add(32 * b), 32) };
                let sc = unsafe { std::slice::from_raw_parts(row.1.add(4 * b), 4) };
                let xq = &a.q[64 * b..64 * b + 64];
                let mut lanes = [0i32; 8];
                for p in 0..32 {
                    let q = qs[p];
                    lanes[p / 4] += E2M1[(q & 15) as usize] as i32 * xq[p] as i32 + E2M1[(q >> 4) as usize] as i32 * xq[32 + p] as i32;
                }
                for (lane, &v) in lanes.iter().enumerate() {
                    acc += v as f32 * ue4m3(sc[lane / 2]) * a.s8[8 * b + lane];
                }
            }
        }
        ExpertFmt::Q8_0 => {
            for b in 0..w.k / 32 {
                let qs = unsafe { std::slice::from_raw_parts(row.0.add(32 * b), 32) };
                let d = half::f16::from_bits(unsafe { (row.1.add(2 * b) as *const u16).read_unaligned() }).to_f32();
                let xq = &a.q[32 * b..32 * b + 32];
                for lane in 0..8 {
                    let mut s = 0i32;
                    for i in 4 * lane..4 * lane + 4 {
                        s += qs[i] as i8 as i32 * xq[i] as i32;
                    }
                    acc += s as f32 * d * a.s8[8 * b + lane];
                }
            }
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2 {
    use super::{Acts, CpuExp, ExpertFmt};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn f16(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((p as *const u16).read_unaligned() as i32)))
    }

    /// Group g of a native row as 32 int8 values + (s0, s1, m); K-quants vectorized, other formats via the
    /// scalar decoder.
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn g32(id: u32, row: *const u8, g: usize, k: usize) -> (__m256i, f32, f32, f32) {
        use crate::codec_tables::*;
        use crate::codecs::{IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K};
        let m4 = _mm256_set1_epi8(0x0F);
        let j = g % 8;
        match id {
            Q4_K | Q5_K => {
                let bs = if id == Q4_K { 144 } else { 176 };
                let b = row.add(g / 8 * bs);
                let qs = _mm256_loadu_si256(b.add(if id == Q4_K { 16 } else { 48 } + 32 * (j / 2)) as *const __m256i);
                let mut q = if j & 1 == 1 { _mm256_and_si256(_mm256_srli_epi16(qs, 4), m4) } else { _mm256_and_si256(qs, m4) };
                if id == Q5_K {
                    let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
                    let bit = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(j as i32)), _mm256_set1_epi8(1));
                    q = _mm256_or_si256(q, _mm256_slli_epi16(bit, 4));
                }
                let sc = std::slice::from_raw_parts(b.add(4), 12);
                let (d, mn) = if j < 4 {
                    (sc[j] & 63, sc[j + 4] & 63)
                } else {
                    ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))
                };
                let s = f16(b) * d as f32;
                (q, s, s, f16(b.add(2)) * mn as f32)
            }
            Q6_K => {
                let b = row.add(g / 8 * 210);
                let (h, jj) = (j / 4, j % 4);
                let ql = _mm256_loadu_si256(b.add(64 * h + 32 * (jj & 1)) as *const __m256i);
                let ql = _mm256_and_si256(_mm256_srl_epi16(ql, _mm_cvtsi32_si128(4 * (jj as i32 >> 1))), m4);
                let qh = _mm256_loadu_si256(b.add(128 + 32 * h) as *const __m256i);
                let qh = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2 * jj as i32)), _mm256_set1_epi8(3));
                let q = _mm256_sub_epi8(_mm256_or_si256(ql, _mm256_slli_epi16(qh, 4)), _mm256_set1_epi8(32));
                let d = f16(b.add(208));
                let sc = b.add(192 + 8 * h + 2 * jj) as *const i8;
                (q, d * *sc as f32, d * *sc.add(1) as f32, 0.0)
            }
            Q3_K => {
                let b = row.add(g / 8 * 110);
                let (n, jj) = (j / 4, j % 4);
                let qs = _mm256_loadu_si256(b.add(32 + 32 * n) as *const __m256i);
                let hm = _mm256_loadu_si256(b as *const __m256i);
                let v = _mm256_and_si256(_mm256_srl_epi16(qs, _mm_cvtsi32_si128(2 * jj as i32)), _mm256_set1_epi8(3));
                let h = _mm256_and_si256(_mm256_srl_epi16(hm, _mm_cvtsi32_si128((4 * n + jj) as i32)), _mm256_set1_epi8(1));
                let q = _mm256_sub_epi8(v, _mm256_slli_epi16(_mm256_xor_si256(h, _mm256_set1_epi8(1)), 2));
                let rd = |o: usize| (b.add(o) as *const u32).read_unaligned();
                let (a0, a1, tmp) = (rd(96), rd(100), rd(104));
                let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
                let aux = [(a0 & k2) | ((tmp & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4), ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)];
                let sc = |i: usize| ((aux[i / 4] >> (8 * (i % 4))) & 0xff) as i32 - 32;
                let d = f16(b.add(108));
                (q, d * sc(8 * n + 2 * jj) as f32, d * sc(8 * n + 2 * jj + 1) as f32, 0.0)
            }
            IQ2_XXS => {
                let b = row.add(g / 8 * 66);
                let a0 = (b.add(2 + 8 * j) as *const u32).read_unaligned();
                let a1 = (b.add(6 + 8 * j) as *const u32).read_unaligned();
                let gr = |l: u32| IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                let sg = |l: u32| KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                let s = f16(b) * (0.5 + (a1 >> 28) as f32) * 0.25;
                (signed32([gr(0), gr(1), gr(2), gr(3)], [sg(0), sg(1), sg(2), sg(3)]), s, s, 0.0)
            }
            IQ2_XS => {
                let b = row.add(g / 8 * 74);
                let v = |l: usize| (b.add(2 + 8 * j + 2 * l) as *const u16).read_unaligned();
                let gr = |l: usize| IQ2XS_GRID[(v(l) & 511) as usize];
                let sg = |l: usize| KSIGNS_IQ2XS[(v(l) >> 9) as usize];
                let (d, sc) = (f16(b), *b.add(66 + j));
                (signed32([gr(0), gr(1), gr(2), gr(3)], [sg(0), sg(1), sg(2), sg(3)]), d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25, 0.0)
            }
            IQ2_S => {
                let b = row.add(g / 8 * 82);
                let qh = *b.add(66 + j) as usize;
                let gr = |l: usize| IQ2S_GRID[*b.add(2 + 4 * j + l) as usize | ((qh << (8 - 2 * l)) & 0x300)];
                let sg = |l: usize| *b.add(34 + 4 * j + l);
                let (d, sc) = (f16(b), *b.add(74 + j));
                (signed32([gr(0), gr(1), gr(2), gr(3)], [sg(0), sg(1), sg(2), sg(3)]), d * (0.5 + (sc & 15) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25, 0.0)
            }
            IQ3_XXS => {
                let b = row.add(g / 8 * 98);
                let aux = (b.add(66 + 4 * j) as *const u32).read_unaligned();
                let gr = |l: usize| IQ3XXS_GRID[*b.add(2 + 8 * j + 2 * l) as usize] as u64 | ((IQ3XXS_GRID[*b.add(3 + 8 * j + 2 * l) as usize] as u64) << 32);
                let sg = |l: u32| KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let s = f16(b) * (0.5 + (aux >> 28) as f32) * 0.5;
                (signed32([gr(0), gr(1), gr(2), gr(3)], [sg(0), sg(1), sg(2), sg(3)]), s, s, 0.0)
            }
            IQ3_S => {
                let b = row.add(g / 8 * 110);
                let qh = *b.add(66 + j) as usize;
                let gr = |l: usize| {
                    let i1 = *b.add(2 + 8 * j + 2 * l) as usize | ((qh << (8 - 2 * l)) & 256);
                    let i2 = *b.add(3 + 8 * j + 2 * l) as usize | ((qh << (7 - 2 * l)) & 256);
                    IQ3S_GRID[i1] as u64 | ((IQ3S_GRID[i2] as u64) << 32)
                };
                let sg = |l: usize| *b.add(74 + 4 * j + l);
                let sc = (*b.add(106 + j / 2) >> (4 * (j & 1))) & 15;
                let s = f16(b) * (1 + 2 * sc as i32) as f32;
                (signed32([gr(0), gr(1), gr(2), gr(3)], [sg(0), sg(1), sg(2), sg(3)]), s, s, 0.0)
            }
            IQ1_S => {
                let b = row.add(g / 8 * 50);
                let qh = (b.add(34 + 2 * j) as *const u16).read_unaligned() as usize;
                let gr = |l: usize| IQ1S_GRID[*b.add(2 + 4 * j + l) as usize | (((qh >> (3 * l)) & 7) << 8)] as i64;
                let q = _mm256_set_epi64x(gr(3), gr(2), gr(1), gr(0));
                let dl = f16(b) * (2 * ((qh >> 12) & 7) + 1) as f32;
                (q, dl, dl, -dl * if qh & 0x8000 != 0 { -0.125 } else { 0.125 })
            }
            IQ4_NL | MXFP4 | IQ4_XS => {
                let (qs, s) = match id {
                    IQ4_NL => {
                        let b = row.add(g * 18);
                        (b.add(2), f16(b))
                    }
                    MXFP4 => {
                        let b = row.add(g * 17);
                        let e = *b as u32;
                        (b.add(1), f32::from_bits(if e < 2 { 0x0020_0000u32 << e } else { (e - 1) << 23 }))
                    }
                    _ => {
                        let b = row.add(g / 8 * 136);
                        let sh = (b.add(2) as *const u16).read_unaligned() as u32;
                        let ls = ((*b.add(4 + j / 2) as u32 >> (4 * (j % 2))) & 15) | (((sh >> (2 * j)) & 3) << 4);
                        (b.add(8 + 16 * j), f16(b) * (ls as i32 - 32) as f32)
                    }
                };
                let t = if id == MXFP4 { &KVALUES_FP4 } else { &KVALUES_IQ4NL };
                let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(t.as_ptr() as *const __m128i));
                let v = _mm_loadu_si128(qs as *const __m128i);
                let m4 = _mm_set1_epi8(0x0F);
                let idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(v, m4));
                (_mm256_shuffle_epi8(lut, idx), s, s, 0.0)
            }
            Q2_0 => {
                let b = row.add(g / 2 * 18);
                let qs = b.add(2 + 8 * (g % 2));
                let idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs as *const __m128i));
                let q = _mm256_i32gather_epi32::<4>(Q2_0_LUT.as_ptr() as *const i32, idx);
                let s = f16(b);
                (q, s, s, 0.0)
            }
            _ => {
                let r = crate::codecs::g32(id, std::slice::from_raw_parts(row, crate::codecs::row_bytes(id, k)), g);
                (_mm256_loadu_si256(r.q.as_ptr() as *const __m256i), r.s0, r.s1, r.m)
            }
        }
    }

    /// 4 x 8 grid bytes, negated where the matching sign byte has a bit set (bit i -> value i of the 8)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signed32(grid: [u64; 4], signs: [u8; 4]) -> __m256i {
        let gv = _mm256_set_epi64x(grid[3] as i64, grid[2] as i64, grid[1] as i64, grid[0] as i64);
        let mk = |s: u8| SIGN_MASK64[s as usize] as i64;
        let m = _mm256_set_epi64x(mk(signs[3]), mk(signs[2]), mk(signs[1]), mk(signs[0]));
        _mm256_sub_epi8(_mm256_xor_si256(gv, m), m)
    }

    /// sign byte -> 8 byte masks (0xFF where the bit is set)
    static SIGN_MASK64: [u64; 256] = {
        let mut t = [0u64; 256];
        let mut i = 0;
        while i < 256 {
            let mut k = 0;
            while k < 8 {
                if i & (1 << k) != 0 {
                    t[i] |= 0xFFu64 << (8 * k);
                }
                k += 1;
            }
            i += 1;
        }
        t
    };

    /// Q2_0: byte -> its four 2-bit values minus 1, as one little-endian u32 of int8
    static Q2_0_LUT: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut w = 0u32;
            let mut k = 0;
            while k < 4 {
                let v = ((i >> (2 * k)) & 3) as i32 - 1;
                w |= ((v as u8) as u32) << (8 * k);
                k += 1;
            }
            t[i] = w;
            i += 1;
        }
        t
    };

    /// Native formats: per group, int8 dot of the decoded values with the activations (|q| x sign-adjusted x),
    /// scales per 16, offsets through the per-16 activation sums.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_native(id: u32, w: CpuExp, row: *const u8, a: &Acts) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        let mut moff = 0f32;
        for g in 0..w.k / 32 {
            let (q, s0, s1, m) = g32(id, row, g, w.k);
            let x = _mm256_loadu_si256(a.q.as_ptr().add(32 * g) as *const __m256i);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(q), _mm256_sign_epi8(x, q)), ones);
            let (xa, xb) = (a.s16[2 * g], a.s16[2 * g + 1]);
            let sc = _mm256_set_m128(_mm_set1_ps(s1 * xb), _mm_set1_ps(s0 * xa));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sc, acc);
            moff += m * (xa * a.sum16[2 * g] as f32 + xb * a.sum16[2 * g + 1] as f32);
        }
        hsum(acc) - moff
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot(w: CpuExp, row: (*const u8, *const u8), a: &Acts) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        match w.fmt {
            ExpertFmt::Native(id) => return dot_native(id, w, row.0, a),
            ExpertFmt::Nvfp4 => {
                let lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
                let m4 = _mm256_set1_epi8(0x0F);
                let dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0);
                for b in 0..w.k / 64 {
                    let q = _mm256_loadu_si256(row.0.add(32 * b) as *const __m256i);
                    let wl = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
                    let wh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
                    let xl = _mm256_loadu_si256(a.q.as_ptr().add(64 * b) as *const __m256i);
                    let xh = _mm256_loadu_si256(a.q.as_ptr().add(64 * b + 32) as *const __m256i);
                    // |w| (<= 12) x sign-adjusted x: int16 pair sums stay below 2 * 12 * 128
                    let pl = _mm256_maddubs_epi16(_mm256_abs_epi8(wl), _mm256_sign_epi8(xl, wl));
                    let ph = _mm256_maddubs_epi16(_mm256_abs_epi8(wh), _mm256_sign_epi8(xh, wh));
                    let s = _mm256_madd_epi16(_mm256_add_epi16(pl, ph), ones);
                    // [d0 d0 d1 d1 d2 d2 d3 d3] << 20 as f32 (x 2^119 in the lane scales)
                    let d4 = _mm_cvtsi32_si128((row.1.add(4 * b) as *const i32).read_unaligned());
                    let d8 = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(d4, dup));
                    let sc = _mm256_mul_ps(_mm256_castsi256_ps(_mm256_slli_epi32(d8, 20)), _mm256_loadu_ps(a.s8.as_ptr().add(8 * b)));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc);
                }
            }
            ExpertFmt::Q8_0 => {
                for b in 0..w.k / 32 {
                    let d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((row.1.add(2 * b) as *const u16).read_unaligned() as i32)));
                    let wv = _mm256_loadu_si256(row.0.add(32 * b) as *const __m256i);
                    let xv = _mm256_loadu_si256(a.q.as_ptr().add(32 * b) as *const __m256i);
                    // int8 x int8 via |w| and sign-adjusted x; |w| <= 127 can reach 2 * 127 * 128 > i16::MAX only
                    // for w = -128, which Q8_0 never stores (symmetric quantization)
                    let p = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
                    let s = _mm256_madd_epi16(p, ones);
                    let sc = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_loadu_ps(a.s8.as_ptr().add(8 * b)));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc);
                }
            }
        }
        hsum(acc)
    }
}

/// A persistent pool: `run(f)` calls f(i) for every worker i in 0..n (the caller is worker 0) and returns when all
/// are done. Idle workers spin briefly (back-to-back offloaded layers arrive ~100 us apart), then sleep.
pub struct Pool {
    n: usize,
    shared: Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

struct PoolShared {
    epoch: AtomicUsize,
    pending: AtomicUsize,
    job: Mutex<Option<JobPtr>>,
    wake: Condvar,
    stop: AtomicBool,
    pin: bool,
}

#[derive(Clone, Copy)]
struct JobPtr(*const (dyn Fn(usize) + Sync));
unsafe impl Send for JobPtr {}
unsafe impl Sync for JobPtr {}

impl Pool {
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        // one worker per physical core when there are enough (the caller, worker 0, runs wherever CUDA calls it)
        let logical = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
        let pin = 2 * n <= logical && std::env::var("INKLING_CPU_PIN").as_deref() == Ok("1");
        let shared = Arc::new(PoolShared { epoch: AtomicUsize::new(0), pending: AtomicUsize::new(0), job: Mutex::new(None), wake: Condvar::new(), stop: AtomicBool::new(false), pin });
        let threads = (1..n)
            .map(|i| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("inkling-cpu-{i}"))
                    .spawn(move || worker(sh, i))
                    .expect("spawn CPU worker")
            })
            .collect();
        Self { n, shared, threads }
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.n == 1 {
            f(0);
            return;
        }
        // SAFETY: the job pointer is only used until `pending` reaches 0, which happens before this returns
        let ptr: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        *self.shared.job.lock().unwrap() = Some(JobPtr(ptr));
        self.shared.pending.store(self.n - 1, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::AcqRel);
        self.shared.wake.notify_all();
        f(0);
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

/// Pin the calling thread to logical CPU `cpu` (Windows; SMT siblings are adjacent there, so 2i = one thread per
/// physical core). Opt-in with INKLING_CPU_PIN=1: it speeds up the isolated CPU benchmark but slowed real decode
/// (the CUDA driver and callback threads then compete with pinned workers).
fn pin_to(cpu: usize) {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        }
        if cpu < usize::BITS as usize {
            unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) };
        }
    }
    #[cfg(not(windows))]
    let _ = cpu;
}

fn worker(sh: Arc<PoolShared>, i: usize) {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if sh.pin {
        pin_to(if 2 * i < logical { 2 * i } else { i % logical });
    }
    let mut seen = 0usize;
    loop {
        // spin, then sleep until the generation changes
        let t0 = Instant::now();
        let mut g = sh.epoch.load(Ordering::Acquire);
        while g == seen && t0.elapsed() < Duration::from_micros(400) {
            std::hint::spin_loop();
            g = sh.epoch.load(Ordering::Acquire);
        }
        if g == seen {
            let mut lock = sh.job.lock().unwrap();
            while sh.epoch.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Acquire) {
                lock = sh.wake.wait(lock).unwrap();
            }
            drop(lock);
            g = sh.epoch.load(Ordering::Acquire);
        }
        if sh.stop.load(Ordering::Acquire) {
            return;
        }
        seen = g;
        let job = sh.job.lock().unwrap().expect("job");
        unsafe { (*job.0)(i) };
        sh.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        {
            let _l = self.shared.job.lock().unwrap();
            self.shared.wake.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Inputs exported by the GPU for one offloaded layer visit (host-mapped memory, written by ink_moe_export).
#[repr(C)]
pub struct MoeIo {
    pub xq: [i8; 2304],
    pub xd: [f32; 2304 / 32],
    pub sel: [i32; 8],
    pub wts: [f32; 8],
    /// output: weighted expert sum (read by the GPU post conv)
    pub y: [f32; 2304],
}

pub static PHASE_DEBUG: AtomicBool = AtomicBool::new(false);

/// CPU expert executor: layers, pool and scratch.
pub struct CpuMoe {
    pub layers: Vec<Option<CpuLayer>>,
    pool: Pool,
    avx2: bool,
    d: usize,
    f: usize,
    beta: f32,
    lb: f32,
    scratch: Mutex<Scratch>,
    /// set when a host callback panicked (its output was zeroed); checked by the engine after each request
    pub failed: AtomicBool,
    /// layer visits run and their total time (ns)
    pub calls: AtomicU64,
    pub nanos: AtomicU64,
}

/// User data of the host-function node that runs one offloaded layer inside the decode graph.
pub struct HostCb {
    pub moe: *const CpuMoe,
    pub io: *mut MoeIo,
    pub layer: usize,
}

unsafe impl Send for HostCb {}
unsafe impl Sync for HostCb {}

/// Default worker count: the physical cores (half the logical CPUs on SMT machines), at most 16. The expert dot
/// products are bound by memory bandwidth, which that many cores saturate.
pub fn default_threads() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (logical / 2).clamp(1, 16)
}

/// CUDA host function: run the experts of `HostCb::layer` on the exchange block. Never unwinds into the driver.
///
/// # Safety
/// `p` must point to a live HostCb whose pointers are valid, and no other code may touch the exchange block while
/// the callback runs.
pub unsafe extern "C" fn host_cb(p: *mut std::ffi::c_void) {
    let cb = unsafe { &*(p as *const HostCb) };
    let moe = unsafe { &*cb.moe };
    let io = unsafe { &mut *cb.io };
    let t0 = Instant::now();
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| moe.run(cb.layer, io))).is_err() {
        io.y.fill(0.0);
        moe.failed.store(true, Ordering::Release);
    }
    moe.nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    moe.calls.fetch_add(1, Ordering::Relaxed);
}

#[derive(Default)]
struct Scratch {
    hq: Vec<i8>,
    hd: Vec<f32>,
}

/// Shared mutable pointer for disjoint parallel writes.
#[derive(Clone, Copy)]
struct SPtr<T>(*mut T);
unsafe impl<T> Send for SPtr<T> {}
unsafe impl<T> Sync for SPtr<T> {}

impl CpuMoe {
    pub fn new(layers: Vec<Option<CpuLayer>>, d: usize, f: usize, beta: f32, lb: f32, threads: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Self {
            layers,
            pool: Pool::new(threads),
            avx2,
            d,
            f,
            beta,
            lb,
            scratch: Mutex::new(Scratch::default()),
            failed: AtomicBool::new(false),
            calls: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn simd(&self) -> &'static str {
        if self.avx2 { "AVX2" } else { "scalar" }
    }

    /// lane-scale multiplier the dot product of format `fmt` expects
    fn fp4_mul(&self, fmt: ExpertFmt) -> f32 {
        if self.avx2 && fmt == ExpertFmt::Nvfp4 { FP4_FAST_MUL } else { 1.0 }
    }

    #[inline]
    fn dot(&self, w: CpuExp, row: (*const u8, *const u8), a: &Acts) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            return unsafe { avx2::dot(w, row, a) };
        }
        dot_scalar(w, row, a)
    }

    /// Microbenchmark: (us per run of layer `li` with random routing, GB/s of expert weights read, GB/s of a
    /// plain parallel read of the same layer's planes = the memory ceiling).
    pub fn bench(&self, li: usize, iters: usize) -> (f64, f64, f64) {
        let layer = self.layers[li].as_ref().expect("layer not offloaded");
        let mut io: Box<MoeIo> = Box::new(unsafe { std::mem::zeroed() });
        for (i, v) in io.xq.iter_mut().enumerate() {
            *v = ((i * 37) % 255) as i8;
        }
        io.xd.fill(0.01);
        io.wts.fill(0.1);
        let (gq, gs) = layer.gate_up.row_bytes();
        let (dq, ds) = layer.down.row_bytes();
        let per_e = (gq + gs) * layer.gate_up.rows + (dq + ds) * layer.down.rows;
        let mut seed = 12345u64;
        let t0 = Instant::now();
        for it in 0..iters {
            for t in 0..8 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                io.sel[t] = ((t * 16 + (seed >> 33) as usize % 16 + it) % 128) as i32;
            }
            self.run(li, &mut io);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        if std::env::var_os("INKLING_CPU_PHASES").is_some() {
            PHASE_DEBUG.store(true, Ordering::Relaxed);
            for _ in 0..5 {
                self.run(li, &mut io);
            }
            PHASE_DEBUG.store(false, Ordering::Relaxed);
        }
        let gbs = 8.0 * per_e as f64 / us / 1e3;
        // ceiling: every thread sums a disjoint slice of the gate_up qs plane (8 experts' worth per iteration)
        let ne = 8;
        let bytes = gq * layer.gate_up.rows * ne;
        let nt = self.pool.threads();
        let base = SPtr(layer.gate_up.qs as *mut u8);
        let sink = AtomicU64::new(0);
        let t1 = Instant::now();
        for it in 0..iters {
            let off = (it % 16) * bytes; // a different 8-expert window each time (beyond the L3)
            self.pool.run(&|ti| {
                let base = base;
                let (lo, hi) = (ti * bytes / nt / 64 * 64, (ti + 1) * bytes / nt / 64 * 64);
                let mut acc = 0u64;
                let mut p = lo;
                while p < hi {
                    acc = acc.wrapping_add(unsafe { (base.0.add(off + p) as *const u64).read_volatile() });
                    p += 64;
                }
                sink.fetch_add(acc, Ordering::Relaxed);
            });
        }
        let raw = bytes as f64 * iters as f64 / t1.elapsed().as_secs_f64() / 1e9;
        (us, gbs, raw)
    }

    /// Expert FFN of offloaded layer `li` for the exported inputs; writes io.y.
    pub fn run(&self, li: usize, io: &mut MoeIo) {
        let layer = self.layers[li].as_ref().expect("layer not offloaded");
        let (d, f) = (self.d, self.f);
        let k = io.sel.len();
        let mut sc = self.scratch.lock().unwrap();
        sc.hq.resize(k * f, 0);
        sc.hd.resize(k * f / 16, 0.0);
        let mut ax = Acts::with_capacity(d);
        prep_acts_mul(&io.xq, &io.xd, 32, layer.gate_up.fmt, d, self.fp4_mul(layer.gate_up.fmt), &mut ax);
        let (hq, hd) = (SPtr(sc.hq.as_mut_ptr()), SPtr(sc.hd.as_mut_ptr()));
        let sel: Vec<usize> = io.sel.iter().map(|&e| e as usize).collect();
        let nt = self.pool.threads();
        // phase 1: units of (expert slot, 16 features): 16 gate + 16 up rows -> 16 quantized hidden values
        let units = k * (f / 16);
        let tp = Instant::now();
        self.pool.run(&|ti| {
            let (hq, hd) = (hq, hd);
            for u in (ti * units / nt)..((ti + 1) * units / nt) {
                let (t, fg) = (u / (f / 16), u % (f / 16));
                let e = sel[t];
                let s = layer.gu_scale.as_ref().map_or(1.0, |v| v[e]);
                let mut h = [0f32; 16];
                for (i, hv) in h.iter_mut().enumerate() {
                    let fi = 16 * fg + i;
                    let g = self.dot(layer.gate_up, layer.gate_up.row(e, fi), &ax) * s;
                    let up = self.dot(layer.gate_up, layer.gate_up.row(e, f + fi), &ax) * s;
                    let a = self.beta * (g / self.beta).tanh() / (1.0 + (-g).exp());
                    *hv = a * (self.lb * (up / self.lb).tanh());
                }
                // as the GPU: v * (127 / amax) rounded half to even
                let amax = h.iter().fold(0f32, |m, v| m.max(v.abs()));
                let id = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                unsafe {
                    for (i, &v) in h.iter().enumerate() {
                        *hq.0.add(t * f + 16 * fg + i) = (v * id).round_ties_even() as i8;
                    }
                    *hd.0.add(t * (f / 16) + fg) = amax / 127.0;
                }
            }
        });
        let t_p1 = tp.elapsed();
        // phase 2: output rows n: y_n = sum_t wts_t s_e (W_e[n] . h_t)
        let hacts: Vec<Acts> = (0..k)
            .map(|t| {
                let mut a = Acts::with_capacity(f);
                prep_acts_mul(&sc.hq[t * f..(t + 1) * f], &sc.hd[t * (f / 16)..(t + 1) * (f / 16)], 16, layer.down.fmt, f, self.fp4_mul(layer.down.fmt), &mut a);
                a
            })
            .collect();
        let y = SPtr(io.y.as_mut_ptr());
        let wts = io.wts;
        self.pool.run(&|ti| {
            let y = y;
            // one expert at a time: each thread streams a contiguous run of rows per expert (fewer concurrent
            // streams for the hardware prefetchers than interleaving the 8 experts row by row)
            let (n0, n1) = (ti * d / nt, (ti + 1) * d / nt);
            let mut acc = vec![0f32; n1 - n0];
            for t in 0..k {
                let e = sel[t];
                let sw = wts[t] * layer.down_scale.as_ref().map_or(1.0, |v| v[e]);
                for (i, n) in (n0..n1).enumerate() {
                    acc[i] += sw * self.dot(layer.down, layer.down.row(e, n), &hacts[t]);
                }
            }
            for (i, v) in acc.into_iter().enumerate() {
                unsafe { *y.0.add(n0 + i) = v };
            }
        });
        if PHASE_DEBUG.load(Ordering::Relaxed) {
            eprintln!("cpu moe layer {li}: phase 1 {:.0} us, total {:.0} us", t_p1.as_secs_f64() * 1e6, tp.elapsed().as_secs_f64() * 1e6);
        }
    }
}
