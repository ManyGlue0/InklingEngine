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
