#pragma once
// Native GGUF block formats decoded 32 values ("one group") at a time for the generic kernels (experts_gen.cu,
// dense_gen.cu, the generic branch of moe_gemm). A group decodes to
//     w_i = s_i * q_i - m,   q_i int8,   s_i = s0 for i < 16, s1 for i >= 16,
// so a dot product with int8 activations x (scale xa for the first 16, xb for the last 16) is
//     xa * (s0 * sum_lo(q x) - m * sum_lo(x)) + xb * (s1 * sum_hi(q x) - m * sum_hi(x)).
// Formats are identified by their ggml type id. Weights stay in the file's block layout (no repack); buffers holding
// them are padded so the unaligned 32-bit loads below may read up to 4 bytes past the last block.
#include "common.cuh"

namespace cdc {

enum : int { Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14 };

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

__device__ __forceinline__ G32 g32(int f, const uint8_t* row, int g) {
    switch (f) {
        case Q4_K: return g32_q4_k(row, g);
        case Q5_K: return g32_q5_k(row, g);
        case Q6_K: return g32_q6_k(row, g);
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
