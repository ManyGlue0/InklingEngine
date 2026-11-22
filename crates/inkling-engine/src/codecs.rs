//! CPU decoders for the natively stored quant formats (mirror of kernels/codecs.cuh). A group of 32 values decodes
//! to int8 q with two scales and an offset: w_i = s_i q_i - m, s_i = s0 for i < 16, s1 for i >= 16.
//! Formats are ggml type ids; layouts are the GGUF blocks (ggml-common.h).

use inkling_core::GgmlType;

pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;

#[derive(Clone, Copy, Debug, Default)]
pub struct G32 {
    pub q: [i8; 32],
    pub s0: f32,
    pub s1: f32,
    pub m: f32,
}

/// Formats the generic kernels decode.
pub fn supported(fmt: u32) -> bool {
    matches!(fmt, Q4_K | Q5_K | Q6_K)
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
