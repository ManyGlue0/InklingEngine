//! CPU decoders for the natively stored quant formats (mirror of kernels/codecs.cuh). A group of 32 values decodes
//! to int8 q with two scales and an offset: w_i = s_i q_i - m, s_i = s0 for i < 16, s1 for i >= 16.
//! Formats are ggml type ids; layouts are the GGUF blocks (ggml-common.h).

use inkling_core::GgmlType;

use crate::codec_tables::*;

pub const Q5_0: u32 = 6;
pub const Q5_1: u32 = 7;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ2_XXS: u32 = 16;
pub const IQ2_XS: u32 = 17;
pub const IQ3_XXS: u32 = 18;
pub const IQ1_S: u32 = 19;
pub const IQ4_NL: u32 = 20;
pub const IQ3_S: u32 = 21;
pub const IQ2_S: u32 = 22;
pub const IQ4_XS: u32 = 23;
pub const IQ1_M: u32 = 29;
pub const MXFP4: u32 = 39;
pub const Q2_0: u32 = 42;

#[derive(Clone, Copy, Debug, Default)]
pub struct G32 {
    pub q: [i8; 32],
    pub s0: f32,
    pub s1: f32,
    pub m: f32,
}

/// Formats the generic kernels decode.
pub fn supported(fmt: u32) -> bool {
    matches!(fmt, Q5_0 | Q5_1 | Q3_K | Q4_K | Q5_K | Q6_K | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ1_S | IQ1_M | IQ4_NL | IQ4_XS | MXFP4 | Q2_0)
}

/// Bytes of one row of `k` values.
pub fn row_bytes(fmt: u32, k: usize) -> usize {
    let (e, b) = GgmlType::Other(fmt).block().expect("known quant format");
    k / e * b
}

#[inline]
fn f16(b: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

#[inline]
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// 8 grid values (bytes of `grid`), negated where `signs` has a bit set.
#[inline]
fn signed8(grid: u64, signs: u8, out: &mut [i8]) {
    for (j, o) in out.iter_mut().enumerate().take(8) {
        let v = ((grid >> (8 * j)) & 0xff) as i8;
        *o = if signs & (1 << j) != 0 { -v } else { v };
    }
}

/// 2^(e-128): the E8M0 scale with the FP4 0.5 factor (ggml_e8m0_to_fp32_half)
#[inline]
fn e8m0_half(e: u8) -> f32 {
    f32::from_bits(if e < 2 { 0x0020_0000u32 << e } else { (e as u32 - 1) << 23 })
}

/// 6-bit scale and min j of a Q4_K / Q5_K super-block.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Decode group `g` (values 32g..32g+32) of a row.
pub fn g32(fmt: u32, row: &[u8], g: usize) -> G32 {
    let mut r = G32::default();
    let j = g % 8;
    match fmt {
        Q4_K | Q5_K => {
            let bs = if fmt == Q4_K { 144 } else { 176 };
            let b = &row[g / 8 * bs..];
            let (p, sh) = (j / 2, 4 * (j & 1));
            let (sc, mn) = scale_min_k4(j, &b[4..16]);
            let qs = &b[if fmt == Q4_K { 16 } else { 48 } + 32 * p..];
            for l in 0..32 {
                let mut v = (qs[l] >> sh) & 15;
                if fmt == Q5_K {
                    v |= ((b[16 + l] >> j) & 1) << 4;
                }
                r.q[l] = v as i8;
            }
            r.s0 = f16(&b[0..2]) * sc as f32;
            r.s1 = r.s0;
            r.m = f16(&b[2..4]) * mn as f32;
        }
        Q6_K => {
            let b = &row[g / 8 * 210..];
            let (h, jj) = (j / 4, j % 4);
            let ql = &b[64 * h + 32 * (jj & 1)..];
            let qh = &b[128 + 32 * h..];
            for l in 0..32 {
                let v = ((ql[l] >> (4 * (jj >> 1))) & 15) | (((qh[l] >> (2 * jj)) & 3) << 4);
                r.q[l] = v as i8 - 32;
            }
            let d = f16(&b[208..210]);
            r.s0 = d * (b[192 + 8 * h + 2 * jj] as i8) as f32;
            r.s1 = d * (b[192 + 8 * h + 2 * jj + 1] as i8) as f32;
        }
        Q3_K => {
            // hmask 32 | qs 64 | scales 12 | d; group G = 4n + j: qs[32n + l] >> 2j, minus 4 unless hmask bit 4n + j
            let b = &row[g / 8 * 110..];
            let (n, jj) = (j / 4, j % 4);
            let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            let (a0, a1, tmp) = (u32le(&b[96..]), u32le(&b[100..]), u32le(&b[104..]));
            let aux = [(a0 & k2) | ((tmp & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4), ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)];
            let sc = |i: usize| ((aux[i / 4] >> (8 * (i % 4))) & 0xff) as i8 as i32 - 32;
            for l in 0..32 {
                let v = (b[32 + 32 * n + l] >> (2 * jj)) & 3;
                r.q[l] = v as i8 - if b[l] & (1 << (4 * n + jj)) != 0 { 0 } else { 4 };
            }
            let d = f16(&b[108..110]);
            r.s0 = d * sc(8 * n + 2 * jj) as f32;
            r.s1 = d * sc(8 * n + 2 * jj + 1) as f32;
        }
        IQ2_XXS => {
            let b = &row[g / 8 * 66..];
            let (a0, a1) = (u32le(&b[2 + 8 * j..]), u32le(&b[6 + 8 * j..]));
            for l in 0..4 {
                signed8(IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize], KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize], &mut r.q[8 * l..]);
            }
            r.s0 = f16(b) * (0.5 + (a1 >> 28) as f32) * 0.25;
            r.s1 = r.s0;
        }
        IQ2_XS => {
            let b = &row[g / 8 * 74..];
            for l in 0..4 {
                let v = u16::from_le_bytes([b[2 + 8 * j + 2 * l], b[3 + 8 * j + 2 * l]]);
                signed8(IQ2XS_GRID[(v & 511) as usize], KSIGNS_IQ2XS[(v >> 9) as usize], &mut r.q[8 * l..]);
            }
            let (d, sc) = (f16(b), b[66 + j]);
            r.s0 = d * (0.5 + (sc & 15) as f32) * 0.25;
            r.s1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
        }
        IQ2_S => {
            let b = &row[g / 8 * 82..];
            let qh = b[66 + j] as usize;
            for l in 0..4 {
                let idx = b[2 + 4 * j + l] as usize | ((qh << (8 - 2 * l)) & 0x300);
                signed8(IQ2S_GRID[idx], b[34 + 4 * j + l], &mut r.q[8 * l..]);
            }
            let (d, sc) = (f16(b), b[74 + j]);
            r.s0 = d * (0.5 + (sc & 15) as f32) * 0.25;
            r.s1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
        }
        IQ3_XXS => {
            let b = &row[g / 8 * 98..];
            let aux = u32le(&b[66 + 4 * j..]);
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[b[2 + 8 * j + 2 * l] as usize] as u64;
                let g2 = IQ3XXS_GRID[b[3 + 8 * j + 2 * l] as usize] as u64;
                signed8(g1 | (g2 << 32), signs, &mut r.q[8 * l..]);
            }
            r.s0 = f16(b) * (0.5 + (aux >> 28) as f32) * 0.5;
            r.s1 = r.s0;
        }
        IQ3_S => {
            let b = &row[g / 8 * 110..];
            let qh = b[66 + j] as usize;
            for l in 0..4 {
                let i1 = b[2 + 8 * j + 2 * l] as usize | ((qh << (8 - 2 * l)) & 256);
                let i2 = b[3 + 8 * j + 2 * l] as usize | ((qh << (7 - 2 * l)) & 256);
                signed8(IQ3S_GRID[i1] as u64 | ((IQ3S_GRID[i2] as u64) << 32), b[74 + 4 * j + l], &mut r.q[8 * l..]);
            }
            let sc = (b[106 + j / 2] >> (4 * (j & 1))) & 15;
            r.s0 = f16(b) * (1 + 2 * sc as i32) as f32;
            r.s1 = r.s0;
        }
        IQ1_S => {
            let b = &row[g / 8 * 50..];
            let qh = u16::from_le_bytes([b[34 + 2 * j], b[35 + 2 * j]]) as usize;
            for l in 0..4 {
                let idx = b[2 + 4 * j + l] as usize | (((qh >> (3 * l)) & 7) << 8);
                let gv = IQ1S_GRID[idx];
                for jj in 0..8 {
                    r.q[8 * l + jj] = ((gv >> (8 * jj)) & 0xff) as u8 as i8;
                }
            }
            let dl = f16(b) * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
            r.s0 = dl;
            r.s1 = dl;
            r.m = -dl * delta;
        }
        IQ4_NL | MXFP4 => {
            let (bs, lut) = if fmt == IQ4_NL { (18, &KVALUES_IQ4NL) } else { (17, &KVALUES_FP4) };
            let b = &row[g * bs..];
            let qs = &b[bs - 16..];
            for l in 0..16 {
                r.q[l] = lut[(qs[l] & 15) as usize];
                r.q[l + 16] = lut[(qs[l] >> 4) as usize];
            }
            r.s0 = if fmt == IQ4_NL { f16(b) } else { e8m0_half(b[0]) };
            r.s1 = r.s0;
        }
        IQ4_XS => {
            let b = &row[g / 8 * 136..];
            let sh = u16::from_le_bytes([b[2], b[3]]) as u32;
            let ls = ((b[4 + j / 2] as u32 >> (4 * (j % 2))) & 15) | (((sh >> (2 * j)) & 3) << 4);
            let qs = &b[8 + 16 * j..];
            for l in 0..16 {
                r.q[l] = KVALUES_IQ4NL[(qs[l] & 15) as usize];
                r.q[l + 16] = KVALUES_IQ4NL[(qs[l] >> 4) as usize];
            }
            r.s0 = f16(b) * (ls as i32 - 32) as f32;
            r.s1 = r.s0;
        }
        Q5_0 => {
            // d | qh u32 | qs 16: value l = (low nibble of qs[l] | bit l of qh << 4) - 16, value 16 + l: high nibble, bit 16 + l
            let b = &row[g * 22..];
            let qh = u32le(&b[2..]);
            for l in 0..16 {
                r.q[l] = ((b[6 + l] & 15) | (((qh >> l) & 1) << 4) as u8) as i8 - 16;
                r.q[l + 16] = ((b[6 + l] >> 4) | (((qh >> (l + 16)) & 1) << 4) as u8) as i8 - 16;
            }
            r.s0 = f16(b);
            r.s1 = r.s0;
        }
        Q5_1 => {
            // d, m | qh u32 | qs 16: like Q5_0 without the -16, plus the offset m (w = d q + m)
            let b = &row[g * 24..];
            let qh = u32le(&b[4..]);
            for l in 0..16 {
                r.q[l] = ((b[8 + l] & 15) | (((qh >> l) & 1) << 4) as u8) as i8;
                r.q[l + 16] = ((b[8 + l] >> 4) | (((qh >> (l + 16)) & 1) << 4) as u8) as i8;
            }
            r.s0 = f16(b);
            r.s1 = r.s0;
            r.m = -f16(&b[2..]);
        }
        IQ1_M => {
            // qs 32 | qh 16 | scales 8. Group j: 4 grid indices (8 bits in qs, 3 in a qh nibble), a +-1/8 delta per
            // 8 values (bit 3 of the nibble), 3-bit scales for each half. 8 (grid + delta) is an integer in [-9, 9],
            // so the scales carry the 1/8 (exact: a power of two).
            let b = &row[g / 8 * 56..];
            let sc = |i: usize| u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]) as u32;
            let bits = (sc(0) >> 12) | ((sc(1) >> 8) & 0x00f0) | ((sc(2) >> 4) & 0x0f00) | (sc(3) & 0xf000);
            let d = half::f16::from_bits(bits as u16).to_f32();
            for l in 0..4 {
                let h = (b[32 + 2 * j + l / 2] >> (4 * (l % 2))) as usize;
                let gv = IQ1S_GRID[b[4 * j + l] as usize | ((h & 7) << 8)];
                let delta = if h & 8 != 0 { -1 } else { 1 };
                for jj in 0..8 {
                    r.q[8 * l + jj] = 8 * ((gv >> (8 * jj)) & 0xff) as u8 as i8 + delta;
                }
            }
            let s = sc(j / 2) >> (6 * (j % 2));
            r.s0 = d * (2 * (s & 7) + 1) as f32 * 0.125;
            r.s1 = d * (2 * ((s >> 3) & 7) + 1) as f32 * 0.125;
        }
        Q2_0 => {
            // 64 values per block: d | qs 16, value i = ((qs[i/4] >> 2 (i%4)) & 3) - 1
            let b = &row[g / 2 * 18..];
            let h = g % 2;
            for l in 0..32 {
                r.q[l] = ((b[2 + 8 * h + l / 4] >> (2 * (l % 4))) & 3) as i8 - 1;
            }
            r.s0 = f16(b);
            r.s1 = r.s0;
        }
        _ => panic!("codecs: unsupported format {fmt}"),
    }
    r
}

/// Dequantize a whole row to f32 (tests and tools).
pub fn dequant_row(fmt: u32, row: &[u8], k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    for g in 0..k / 32 {
        let q = g32(fmt, row, g);
        for (i, &v) in q.q.iter().enumerate() {
            out.push(if i < 16 { q.s0 } else { q.s1 } * v as f32 - q.m);
        }
    }
    out
}
