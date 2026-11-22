// Generic decode expert kernels for natively stored formats (codecs.cuh), same interface as ink_moe_gate_up7 /
// ink_moe_down7: gate_up writes the int8 hidden hq [k][F] with scales per 16 hd [k][F/16]; down consumes it and
// applies the expert sum, post conv and residual. A layer may mix one of these with the v7 kernel for the other
// tensor (e.g. IQ gate_up + NVFP4 down). ExpW.qs points at the tensor's blocks, ExpW.fmt is the ggml type id.
#include "codecs.cuh"

// grid (F/16, k), 256 threads; warp w: features f, f+1 (f = 16 blockIdx.x + 2w); lane group g = lane/8 owns row
// {gate f, up f, gate f+1, up f+1}[g]; lane l8 takes groups u = l8 + 8i of the 72 groups of 32 in a row.
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_gate_up_gen(ExpW w, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                                         const int* __restrict__ sel, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                                         float beta, float lb) {
    using namespace ex7;
    __shared__ __align__(16) int8_t xs[D];
    __shared__ float xds[D / 32];
    __shared__ float hv[16];
    const int t = blockIdx.y, e = sel[t];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 3, l8 = lane & 7;
    const int f = blockIdx.x * 16 + 2 * warp + (g >> 1);
    const size_t row = (size_t)e * 2 * F + (g & 1) * F + f;
    const uint8_t* rp = w.qs + row * cdc::row_bytes(w.fmt, D);
    stage16(xs, xq, D, false);
    for (int i = threadIdx.x; i < D / 32; i += blockDim.x) xds[i] = xd[i];
    __syncthreads();
    float acc = 0.0f;
#pragma unroll 3
    for (int i = 0; i < D / 32 / 8; ++i) {
        const int u = l8 + 8 * i;
        const cdc::G32 q = cdc::g32(w.fmt, rp, u);
        const int4 a = *reinterpret_cast<const int4*>(xs + 32 * u), b = *reinterpret_cast<const int4*>(xs + 32 * u + 16);
        const int x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        acc += cdc::g32_dot(q, x, xds[u], xds[u]);
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    const float sc = w.scale ? w.scale[e] : 1.0f;
    const float up = __shfl_sync(0xffffffffu, acc, (lane & 16) | 8) * sc;  // lane 0 <- lane 8, lane 16 <- lane 24
    if ((lane & 15) == 0) {
        const float gt = acc * sc;
        const float a = beta * tanhf(gt / beta) * (1.0f / (1.0f + __expf(-gt)));
        hv[2 * warp + (lane >> 4)] = a * (lb * tanhf(up / lb));
    }
    __syncthreads();
    if (warp == 0) {
        const float v = hv[lane & 15];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        if (lane < 16) hq[t * F + blockIdx.x * 16 + lane] = amax == 0.0f ? 0 : (int8_t)roundf(v / d);
        if (lane == 0) hd[t * (F / 16) + blockIdx.x] = d;
    }
}

// grid D/8, 256 threads; warp w: output channel n = 8 blockIdx.x + w; lane group t = lane/4 is the t-th selected
// expert, lane l4 takes groups u = l4 + 4i of the 36 groups of 32 in a row.
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_down_gen(ExpW w, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                                                                      const int* __restrict__ sel, const float* __restrict__ wts,
                                                                      const float* __restrict__ cw, float* __restrict__ state,
                                                                      float* __restrict__ x, float* __restrict__ dbg) {
    using namespace ex7;
    constexpr int K = 8, HS = F + 64;
    __shared__ __align__(16) int8_t hs[K * HS];
    __shared__ float hds[K][F / 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane >> 2, l4 = lane & 3;
    const int n = blockIdx.x * 8 + warp;
    const int e = sel[t];
    const uint8_t* rp = w.qs + ((size_t)e * D + n) * cdc::row_bytes(w.fmt, F);
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0, st0 = 0, st1 = 0, st2 = 0, xr = 0;
    if (lane == 0) {
        const float4 cv = *reinterpret_cast<const float4*>(cw + (size_t)n * 4);
        c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
        st0 = state[n]; st1 = state[D + n]; st2 = state[2 * D + n];
        xr = x[n];
    }
    const float sw = wts[t] * (w.scale ? w.scale[e] : 1.0f);
    stage_h(hs, hq, false);
    for (int i = threadIdx.x; i < K * (F / 16); i += blockDim.x) hds[i / (F / 16)][i % (F / 16)] = hd[i];
    __syncthreads();
    const int8_t* h = hs + t * HS;
    float acc = 0.0f;
#pragma unroll 3
    for (int i = 0; i < F / 32 / 4; ++i) {
        const int u = l4 + 4 * i;
        const cdc::G32 q = cdc::g32(w.fmt, rp, u);
        const int4 a = *reinterpret_cast<const int4*>(h + 32 * u), b = *reinterpret_cast<const int4*>(h + 32 * u + 16);
        const int xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        acc += cdc::g32_dot(q, xv, hds[t][2 * u], hds[t][2 * u + 1]);
    }
    acc *= sw;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        const float y = st0 * c0 + st1 * c1 + st2 * c2 + acc * c3;
        const float o = (acc + y) * rsqrtf(5.0f);
        state[n] = st1;
        state[D + n] = st2;
        state[2 * D + n] = acc;
        if (dbg) dbg[n] = o;
        x[n] = xr + o;
    }
}

// Test: dequantize `rows` rows of k values (format fmt, rows rb bytes apart) to f32, one block per row.
extern "C" __global__ void ink_dequant_rows(const uint8_t* __restrict__ w, int fmt, int k, float* __restrict__ out) {
    const uint8_t* rp = w + (size_t)blockIdx.x * cdc::row_bytes(fmt, k);
    for (int g = threadIdx.x; g < k / 32; g += blockDim.x) {
        const cdc::G32 q = cdc::g32(fmt, rp, g);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int v = (int)(int8_t)((q.q[i >> 2] >> (8 * (i & 3))) & 0xff);
            out[(size_t)blockIdx.x * k + 32 * g + i] = (i < 16 ? q.s0 : q.s1) * (float)v - q.m;
        }
    }
}
