// Smoke-test and roofline kernels.
#include "common.cuh"

// y[i] = a*x[i] + y[i]
INK_EXTERN void ink_axpy_f32(float a, const float* __restrict__ x, float* __restrict__ y, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Streaming read of n uint4 words; one partial sum per block. Measures achievable DRAM read bandwidth,
// the ceiling for memory-bound decode.
INK_EXTERN void ink_read_bw(const uint4* __restrict__ src, uint64_t n, uint32_t* __restrict__ out) {
    uint32_t acc = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(src + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;  // practically never taken; keeps the loads alive
}

// Row-gather read pattern: rows of 72 uint4 (1152 B, one FP4 expert row of K = 2304); LPR lanes per row read
// uint4 l, l + LPR, ... (72 / LPR each, all issued before use). Tasks (32 / LPR rows) are grid-strided over warps.
template <int LPR>
__device__ void pattern_bw(const uint4* __restrict__ src, int rows, uint32_t* __restrict__ out) {
    constexpr int PER = 72 / LPR, RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, g = lane / LPR, l = lane % LPR;
    const int nw = gridDim.x * (blockDim.x >> 5), gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint32_t acc = 0;
    for (int task = gw; task * RPW < rows; task += nw) {
        const uint4* r = src + (size_t)(task * RPW + g) * 72;
        uint4 v[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) v[i] = __ldg(r + l + LPR * i);
#pragma unroll
        for (int i = 0; i < PER; ++i) acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;
}
INK_EXTERN void ink_pattern_bw8(const uint4* __restrict__ src, int rows, uint32_t* __restrict__ out) { pattern_bw<8>(src, rows, out); }
INK_EXTERN void ink_pattern_bw4(const uint4* __restrict__ src, int rows, uint32_t* __restrict__ out) { pattern_bw<4>(src, rows, out); }
INK_EXTERN void ink_pattern_bw2(const uint4* __restrict__ src, int rows, uint32_t* __restrict__ out) { pattern_bw<2>(src, rows, out); }

// As pattern_bw<8>, plus the FP4 scale stream: one ushort per uint4 from a separate region (rows of 144 B) when
// sc_mode = 1, or packed right after each row (row stride 1296 B) when sc_mode = 2.
INK_EXTERN void ink_pattern_bw8s(const uint4* __restrict__ src, int rows, uint32_t* __restrict__ out, int sc_mode) {
    const int lane = threadIdx.x & 31, g = lane / 8, l = lane % 8;
    const int nw = gridDim.x * (blockDim.x >> 5), gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int stride = sc_mode == 2 ? 81 : 72;  // uint4 per row
    const unsigned short* sc = reinterpret_cast<const unsigned short*>(src + (size_t)rows * 72);
    uint32_t acc = 0;
    for (int task = gw; task * 4 < rows; task += nw) {
        const size_t row = (size_t)(task * 4 + g);
        const uint4* r = src + row * stride;
        const unsigned short* sr = sc_mode == 2 ? reinterpret_cast<const unsigned short*>(r + 72) : sc + row * 72;
        uint4 v[9];
        unsigned short w[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            v[i] = __ldg(r + l + 8 * i);
            w[i] = __ldg(sr + l + 8 * i);
        }
#pragma unroll
        for (int i = 0; i < 9; ++i) acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w ^ w[i];
    }
    if (acc == 0x9e3779b9u) out[blockIdx.x] = acc;
}
