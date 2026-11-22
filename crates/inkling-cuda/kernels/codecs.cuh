#pragma once
// Native GGUF block formats decoded 32 values ("one group") at a time for the generic kernels (experts_gen.cu,
// dense_gen.cu, the generic branch of moe_gemm). A group decodes to
//     w_i = s_i * q_i - m,   q_i int8,   s_i = s0 for i < 16, s1 for i >= 16,
// so a dot product with int8 activations x (scale xa for the first 16, xb for the last 16) is
//     xa * (s0 * sum_lo(q x) - m * sum_lo(x)) + xb * (s1 * sum_hi(q x) - m * sum_hi(x)).
// Formats are identified by their ggml type id. Weights stay in the file's block layout (no repack); buffers holding
// them are padded so the unaligned 32-bit loads below may read up to 4 bytes past the last block.
#include "common.cuh"
#include "codec_tables.cuh"

namespace cdc {

enum : int { Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
             IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, MXFP4 = 39, Q2_0 = 42 };

struct G32 {
    int q[8];  // 32 int8, little-endian in 8 words
    float s0, s1, m;
};

__host__ __device__ constexpr int blk_elems(int f) { return f == 20 || f == 39 ? 32 : f == 42 ? 64 : 256; }
__host__ __device__ constexpr int blk_bytes(int f) {
    return f == Q3_K ? 110 : f == Q4_K ? 144 : f == Q5_K ? 176 : f == Q6_K ? 210 : f == 16 ? 66 : f == 17 ? 74 : f == 18 ? 98
         : f == 19 ? 50 : f == 20 ? 18 : f == 21 ? 110 : f == 22 ? 82 : f == 23 ? 136 : f == 39 ? 17 : f == 42 ? 18 : 0;
}
__host__ __device__ inline size_t row_bytes(int f, int k) { return (size_t)(k / blk_elems(f)) * blk_bytes(f); }

__device__ __forceinline__ float h2f(uint16_t x) { return __half2float(__ushort_as_half(x)); }
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned short*>(p)); }
// 32-bit load from any byte address (two aligned loads + funnel shift)
__device__ __forceinline__ uint32_t ldu32(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const unsigned int* w = reinterpret_cast<const unsigned int*>(a & ~uintptr_t(3));
    const unsigned int sh = (unsigned int)(a & 3) * 8;
    const unsigned int lo = __ldg(w);
    return sh ? __funnelshift_r(lo, __ldg(w + 1), sh) : lo;
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t byte_of(const uint32_t (&w)[3], int k) { return (w[k >> 2] >> (8 * (k & 3))) & 0xff; }

__device__ __forceinline__ uint32_t ld8(const uint8_t* p) { return __ldg(p); }

// 4 bytes of v negated where the low 4 bits of s are set (bit i -> byte i)
__device__ __forceinline__ int signw(uint32_t v, uint32_t s) {
    const uint32_t m = (((s & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (int)__vsub4(v ^ m, m);
}
// 8 grid bytes (u64 table entry) with 8 sign bits -> two words
__device__ __forceinline__ void grid8(const uint64_t gv, uint32_t s, int& w0, int& w1) {
    w0 = signw((uint32_t)gv, s);
    w1 = signw((uint32_t)(gv >> 32), s >> 4);
}
// 4 nibbles (low or high halves of 4 bytes) through a 16-entry int8 table -> one word
__device__ __forceinline__ int lut4(const int8_t* t, uint32_t v, int sh) {
    return (int)((uint32_t)(uint8_t)t[(v >> sh) & 15] | ((uint32_t)(uint8_t)t[(v >> (sh + 8)) & 15] << 8) |
                 ((uint32_t)(uint8_t)t[(v >> (sh + 16)) & 15] << 16) | ((uint32_t)(uint8_t)t[(v >> (sh + 24)) & 15] << 24));
}

// 6-bit scale and min j of a Q4_K / Q5_K super-block (scales[12] in three words)
__device__ __forceinline__ void scale_min_k4(const uint32_t (&w)[3], int j, int& d, int& m) {
    if (j < 4) {
        d = byte_of(w, j) & 63;
        m = byte_of(w, j + 4) & 63;
    } else {
        d = (byte_of(w, j + 4) & 0xF) | ((byte_of(w, j - 4) >> 6) << 4);
        m = (byte_of(w, j + 4) >> 4) | ((byte_of(w, j) >> 6) << 4);
    }
}

// Q4_K: 256 values = 8 groups; group j: 64-value pair p = j/2, low nibbles (j even) or high nibbles of qs[32p..]
__device__ __forceinline__ G32 g32_q4_k(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 144;
    const int j = g & 7, p = j >> 1, sh = 4 * (j & 1);
    const uint32_t dm = ld32(b);
    const uint32_t sw[3] = {ld32(b + 4), ld32(b + 8), ld32(b + 12)};
    int sc, mn;
    scale_min_k4(sw, j, sc, mn);
    G32 r;
    const uint8_t* qs = b + 16 + 32 * p;
#pragma unroll
    for (int w = 0; w < 8; ++w) r.q[w] = (ld32(qs + 4 * w) >> sh) & 0x0F0F0F0F;
    r.s0 = r.s1 = h2f(dm & 0xffff) * sc;
    r.m = h2f(dm >> 16) * mn;
    return r;
}

// Q5_K: as Q4_K plus a 5th bit: qh[l] bit (2p + j%2) for value l of the group
__device__ __forceinline__ G32 g32_q5_k(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 176;
    const int j = g & 7, p = j >> 1, sh = 4 * (j & 1), hb = j;
    const uint32_t dm = ld32(b);
    const uint32_t sw[3] = {ld32(b + 4), ld32(b + 8), ld32(b + 12)};
    int sc, mn;
    scale_min_k4(sw, j, sc, mn);
    G32 r;
    const uint8_t* qh = b + 16;
    const uint8_t* qs = b + 48 + 32 * p;
#pragma unroll
    for (int w = 0; w < 8; ++w) r.q[w] = ((ld32(qs + 4 * w) >> sh) & 0x0F0F0F0F) | (((ld32(qh + 4 * w) >> hb) & 0x01010101) << 4);
    r.s0 = r.s1 = h2f(dm & 0xffff) * sc;
    r.m = h2f(dm >> 16) * mn;
    return r;
}

// Q6_K (210-byte blocks, 2-byte aligned): group j: half h = j/4 (128 values), jj = j%4 selects the ql bytes
// (+32 for odd jj), the nibble (high for jj >= 2) and the 2 qh bits (2 jj); values q - 32; a scale per 16
__device__ __forceinline__ G32 g32_q6_k(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 210;
    const int j = g & 7, h = j >> 2, jj = j & 3;
    const uint8_t* ql = b + 64 * h + 32 * (jj & 1);
    const uint8_t* qh = b + 128 + 32 * h;
    const int nsh = 4 * (jj >> 1), hsh = 2 * jj;
    G32 r;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t v = ((ldu32(ql + 4 * w) >> nsh) & 0x0F0F0F0F) | (((ldu32(qh + 4 * w) >> hsh) & 0x03030303) << 4);
        r.q[w] = (int)__vsub4(v, 0x20202020u);
    }
    const float d = h2f(ld16(b + 208));
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192 + 8 * h + 2 * jj);
    r.s0 = d * (float)sc[0];
    r.s1 = d * (float)sc[1];
    r.m = 0.0f;
    return r;
}

// Q3_K: hmask 32 | qs 64 | scales 12 | d (110-byte blocks); group G = 4n + j: (qs[32n + l] >> 2j) & 3, minus 4
// unless hmask[l] bit 4n + j; 16 six-bit scales (minus 32), one per 16 values
__device__ __forceinline__ G32 g32_q3_k(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 110;
    const int j = g & 7, n = j >> 2, jj = j & 3;
    G32 r;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t v = (ldu32(b + 32 + 32 * n + 4 * w) >> (2 * jj)) & 0x03030303u;
        const uint32_t h = (ldu32(b + 4 * w) >> (4 * n + jj)) & 0x01010101u;
        r.q[w] = (int)__vsub4(v, (~h & 0x01010101u) << 2);
    }
    const uint32_t a0 = ldu32(b + 96), a1 = ldu32(b + 100), tmp = ldu32(b + 104);
    const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    const uint32_t aux[4] = {(a0 & k2) | ((tmp & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4), ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
                             ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)};
    const int i0 = 8 * n + 2 * jj;
    const float d = h2f(ld16(b + 108));
    r.s0 = d * (float)((int)((aux[i0 >> 2] >> (8 * (i0 & 3))) & 0xff) - 32);
    r.s1 = d * (float)((int)((aux[(i0 + 1) >> 2] >> (8 * ((i0 + 1) & 3))) & 0xff) - 32);
    r.m = 0.0f;
    return r;
}

// IQ2_XXS (66 B): d | per group 8 bytes: 4 grid indices, then 4 x 7 sign bits + a 4-bit scale
__device__ __forceinline__ G32 g32_iq2_xxs(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 66;
    const int j = g & 7;
    const uint32_t a0 = ldu32(b + 2 + 8 * j), a1 = ldu32(b + 6 + 8 * j);
    G32 r;
#pragma unroll
    for (int l = 0; l < 4; ++l) grid8(iq2xxs_grid[(a0 >> (8 * l)) & 255], ksigns_iq2xs[(a1 >> (7 * l)) & 127], r.q[2 * l], r.q[2 * l + 1]);
    r.s0 = r.s1 = h2f(ld16(b)) * (0.5f + (float)(a1 >> 28)) * 0.25f;
    r.m = 0.0f;
    return r;
}

// IQ2_XS (74 B): d | 32 x u16 (9-bit grid index, 7 sign bits) | 8 scale bytes (two 4-bit scales per group)
__device__ __forceinline__ G32 g32_iq2_xs(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 74;
    const int j = g & 7;
    G32 r;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t v = ld16(b + 2 + 8 * j + 2 * l);
        grid8(iq2xs_grid[v & 511], ksigns_iq2xs[v >> 9], r.q[2 * l], r.q[2 * l + 1]);
    }
    const float d = h2f(ld16(b));
    const uint32_t sc = ld8(b + 66 + j);
    r.s0 = d * (0.5f + (float)(sc & 15)) * 0.25f;
    r.s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
    r.m = 0.0f;
    return r;
}

// IQ2_S (82 B): d | 32 grid index bytes | 32 sign bytes | 8 qh (2 high index bits per 8 values) | 8 scale bytes
__device__ __forceinline__ G32 g32_iq2_s(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 82;
    const int j = g & 7;
    const uint32_t qh = ld8(b + 66 + j);
    G32 r;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t idx = ld8(b + 2 + 4 * j + l) | ((qh << (8 - 2 * l)) & 0x300);
        grid8(iq2s_grid[idx], ld8(b + 34 + 4 * j + l), r.q[2 * l], r.q[2 * l + 1]);
    }
    const float d = h2f(ld16(b));
    const uint32_t sc = ld8(b + 74 + j);
    r.s0 = d * (0.5f + (float)(sc & 15)) * 0.25f;
    r.s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
    r.m = 0.0f;
    return r;
}

// IQ3_XXS (98 B): d | 64 grid index bytes (8 per group, 4 values each) | per group 4 x 7 sign bits + 4-bit scale
__device__ __forceinline__ G32 g32_iq3_xxs(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 98;
    const int j = g & 7;
    const uint32_t aux = ldu32(b + 66 + 4 * j);
    G32 r;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        r.q[2 * l] = signw(iq3xxs_grid[ld8(b + 2 + 8 * j + 2 * l)], s);
        r.q[2 * l + 1] = signw(iq3xxs_grid[ld8(b + 3 + 8 * j + 2 * l)], s >> 4);
    }
    r.s0 = r.s1 = h2f(ld16(b)) * (0.5f + (float)(aux >> 28)) * 0.5f;
    r.m = 0.0f;
    return r;
}

// IQ3_S (110 B): d | 64 index bytes | 8 qh (9th index bit) | 32 sign bytes | 4 scale bytes (4 bits per group)
__device__ __forceinline__ G32 g32_iq3_s(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 110;
    const int j = g & 7;
    const uint32_t qh = ld8(b + 66 + j);
    G32 r;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t i1 = ld8(b + 2 + 8 * j + 2 * l) | ((qh << (8 - 2 * l)) & 256);
        const uint32_t i2 = ld8(b + 3 + 8 * j + 2 * l) | ((qh << (7 - 2 * l)) & 256);
        const uint32_t s = ld8(b + 74 + 4 * j + l);
        r.q[2 * l] = signw(iq3s_grid[i1], s);
        r.q[2 * l + 1] = signw(iq3s_grid[i2], s >> 4);
    }
    const uint32_t sc = (ld8(b + 106 + (j >> 1)) >> (4 * (j & 1))) & 15;
    r.s0 = r.s1 = h2f(ld16(b)) * (float)(1 + 2 * (int)sc);
    r.m = 0.0f;
    return r;
}

// IQ1_S (50 B): d | 32 index bytes | 8 x u16 qh (3 high index bits per 8 values, 3-bit scale, delta sign);
// values (grid + delta) with grid in {-1, 0, 1}, delta = +-1/8 -> offset m = -scale * delta
__device__ __forceinline__ G32 g32_iq1_s(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 50;
    const int j = g & 7;
    const uint32_t qh = ld16(b + 34 + 2 * j);
    G32 r;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint64_t gv = iq1s_grid[ld8(b + 2 + 4 * j + l) | (((qh >> (3 * l)) & 7) << 8)];
        r.q[2 * l] = (int)(uint32_t)gv;
        r.q[2 * l + 1] = (int)(uint32_t)(gv >> 32);
    }
    const float dl = h2f(ld16(b)) * (float)(2 * ((qh >> 12) & 7) + 1);
    r.s0 = r.s1 = dl;
    r.m = -dl * ((qh & 0x8000) ? -0.125f : 0.125f);
    return r;
}

// IQ4_NL (18 B per 32): d | 16 bytes, low nibbles -> values 0..15, high -> 16..31 (non-linear int8 table);
// MXFP4 (17 B per 32): E8M0 scale | 16 bytes, same nibble order, doubled E2M1 values
__device__ __forceinline__ G32 g32_nl(const uint8_t* row, int g, bool mx) {
    const uint8_t* b = row + (size_t)g * (mx ? 17 : 18);
    const uint8_t* qs = b + (mx ? 1 : 2);
    const int8_t* t = mx ? kvalues_fp4 : kvalues_iq4nl;
    G32 r;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = ldu32(qs + 4 * w);
        r.q[w] = lut4(t, v, 0);
        r.q[w + 4] = lut4(t, v, 4);
    }
    if (mx) {
        const uint32_t e = ld8(b);
        r.s0 = r.s1 = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
    } else {
        r.s0 = r.s1 = h2f(ld16(b));
    }
    r.m = 0.0f;
    return r;
}

// IQ4_XS (136 B): d | scales_h u16 | scales_l 4 | 128 nibble bytes (16 per group); 6-bit scales minus 32
__device__ __forceinline__ G32 g32_iq4_xs(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 3) * 136;
    const int j = g & 7;
    G32 r;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = ld32(b + 8 + 16 * j + 4 * w);
        r.q[w] = lut4(kvalues_iq4nl, v, 0);
        r.q[w + 4] = lut4(kvalues_iq4nl, v, 4);
    }
    const uint32_t sh = ld16(b + 2);
    const int ls = (int)(((ld8(b + 4 + (j >> 1)) >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4));
    r.s0 = r.s1 = h2f(ld16(b)) * (float)(ls - 32);
    r.m = 0.0f;
    return r;
}

// Q2_0 (Unsloth, 18 B per 64): d | 16 bytes of 2-bit values q, value = q - 1
__device__ __forceinline__ G32 g32_q2_0(const uint8_t* row, int g) {
    const uint8_t* b = row + (size_t)(g >> 1) * 18;
    const uint32_t lo = ldu32(b + 2 + 8 * (g & 1)), hi = ldu32(b + 6 + 8 * (g & 1));
    G32 r;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t v = ((w < 4 ? lo : hi) >> (8 * (w & 3))) & 0xff;
        const uint32_t x = (v & 3) | ((v >> 2) & 3) << 8 | ((v >> 4) & 3) << 16 | ((v >> 6) & 3) << 24;
        r.q[w] = (int)__vsub4(x, 0x01010101u);
    }
    r.s0 = r.s1 = h2f(ld16(b));
    r.m = 0.0f;
    return r;
}

__device__ __forceinline__ G32 g32(int f, const uint8_t* row, int g) {
    switch (f) {
        case Q3_K: return g32_q3_k(row, g);
        case Q4_K: return g32_q4_k(row, g);
        case Q5_K: return g32_q5_k(row, g);
        case Q6_K: return g32_q6_k(row, g);
        case IQ2_XXS: return g32_iq2_xxs(row, g);
        case IQ2_XS: return g32_iq2_xs(row, g);
        case IQ2_S: return g32_iq2_s(row, g);
        case IQ3_XXS: return g32_iq3_xxs(row, g);
        case IQ3_S: return g32_iq3_s(row, g);
        case IQ1_S: return g32_iq1_s(row, g);
        case IQ4_NL: return g32_nl(row, g, false);
        case MXFP4: return g32_nl(row, g, true);
        case IQ4_XS: return g32_iq4_xs(row, g);
        case Q2_0: return g32_q2_0(row, g);
        default: {
            G32 r = {};
            return r;
        }
    }
}

// dot of a decoded group with 32 int8 activations x (8 words), activation scales xa (first 16) / xb (last 16)
__device__ __forceinline__ float g32_dot(const G32& g, const int (&x)[8], float xa, float xb) {
    int lo = 0, hi = 0, slo = 0, shi = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        lo = __dp4a(g.q[w], x[w], lo);
        hi = __dp4a(g.q[w + 4], x[w + 4], hi);
        slo = __dp4a(0x01010101, x[w], slo);
        shi = __dp4a(0x01010101, x[w + 4], shi);
    }
    return xa * (g.s0 * (float)lo - g.m * (float)slo) + xb * (g.s1 * (float)hi - g.m * (float)shi);
}

}  // namespace cdc
