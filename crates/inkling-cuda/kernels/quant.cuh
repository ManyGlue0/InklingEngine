#pragma once
// Weight block formats (bit-identical to GGUF) and dequantization helpers.
#include "common.cuh"

struct __align__(2) block_q8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block");

struct block_nvfp4 {
    uint8_t d[4];   // UE4M3 scale per 16 values
    uint8_t qs[32]; // packed E2M1: sub-block s, byte j: low -> 16s+j, high -> 16s+j+8
};
static_assert(sizeof(block_nvfp4) == 36, "nvfp4 block");

// E2M1 values doubled; the UE4M3 decode below carries the matching 0.5 factor.
__constant__ float k_e2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float ue4m3_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    const float raw = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m * 0.125f, e - 7);
    return raw * 0.5f;
}

// Dot of one Q8_0 row (K elements) with x, computed by one warp. Returns the full sum on every lane.
__device__ __forceinline__ float warp_dot_q8(const block_q8_0* __restrict__ row, const float* __restrict__ x, int K) {
    const int lane = threadIdx.x & 31;
    const int nb = K / 32;
    float acc = 0.0f;
    // each lane takes one 4-value chunk of a block per step: 8 lanes per block, 4 blocks per warp step
    for (int b = lane / 8; b < nb; b += 4) {
        const block_q8_0& blk = row[b];
        const int j = (lane % 8) * 4;
        const float d = __half2float(blk.d);
        const float* xb = x + b * 32 + j;
        acc += d * (blk.qs[j] * xb[0] + blk.qs[j + 1] * xb[1] + blk.qs[j + 2] * xb[2] + blk.qs[j + 3] * xb[3]);
    }
    return warp_sum(acc);
}

// Dot of one NVFP4 row with x by one warp. Each lane handles one byte (2 values) of a block per step.
__device__ __forceinline__ float warp_dot_nvfp4(const block_nvfp4* __restrict__ row, const float* __restrict__ x, int K) {
    const int lane = threadIdx.x & 31;
    const int nb = K / 64;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_nvfp4& blk = row[b];
        const int s = lane / 8, j = lane % 8;
        const uint8_t q = blk.qs[lane];
        const float d = ue4m3_to_f32(blk.d[s]);
        const float* xb = x + b * 64 + s * 16;
        acc += d * (k_e2m1x2[q & 0xF] * xb[j] + k_e2m1x2[q >> 4] * xb[j + 8]);
    }
    return warp_sum(acc);
}
