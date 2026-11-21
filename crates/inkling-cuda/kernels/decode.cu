// M4 decode kernels on repacked weights.
//
// Repacked layouts (built on load, see inkling-engine/src/repack.rs):
//   Q8 "SoA":   qs int8 [rows][K] (16B aligned rows), d half [rows][K/32]
//   NVFP4 SoA:  qs u8 [rows][K/2] (sub-block s = bytes 8s..8s+7: low nibble -> 16s+j, high -> 16s+j+8),
//               sc u8 [rows][K/16] (UE4M3, one per 16 values)
#include "quant.cuh"

// ---------------------------------------------------------------------------------------------------------------
// Multi-segment Q8 matvec: several matrices with the same K, each with its own input and output, in one launch.
struct MVSeg {
    const int8_t* qs;
    const __half* d;
    const float* x;
    float* y;
    int n;
};
struct MV4 {
    MVSeg s[4];
    int nseg;
    int K;
};

__device__ __forceinline__ float warp_dot_q8s(const int8_t* __restrict__ qs, const __half* __restrict__ d,
                                              const float* __restrict__ x, int K) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    const int nchunk = K / 16;
    for (int c = lane; c < nchunk; c += 32) {
        const int4 q = *reinterpret_cast<const int4*>(qs + c * 16);
        const float sc = __half2float(d[c >> 1]);
        const float4* xv = reinterpret_cast<const float4*>(x + c * 16);
        const int8_t* b = reinterpret_cast<const int8_t*>(&q);
        float s = 0.0f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float4 xx = xv[t];
            s += b[4 * t] * xx.x + b[4 * t + 1] * xx.y + b[4 * t + 2] * xx.z + b[4 * t + 3] * xx.w;
        }
        acc += sc * s;
    }
    return warp_sum(acc);
}

INK_EXTERN void ink_mv_q8(MV4 a) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    int seg = 0;
    while (seg < a.nseg && row >= a.s[seg].n) { row -= a.s[seg].n; ++seg; }
    if (seg >= a.nseg) return;
    const MVSeg& s = a.s[seg];
    const float v = warp_dot_q8s(s.qs + (size_t)row * a.K, s.d + (size_t)row * (a.K / 32), s.x, a.K);
    if ((threadIdx.x & 31) == 0) s.y[row] = v;
}

// Multi-segment BF16 matvec (small projections: sink, dynamic conv taps).
struct MVSegB {
    const __nv_bfloat16* w;
    const float* x;
    float* y;
    int n;
};
struct MVB4 {
    MVSegB s[4];
    int nseg;
    int K;
};

INK_EXTERN void ink_mv_bf16(MVB4 a) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    int seg = 0;
    while (seg < a.nseg && row >= a.s[seg].n) { row -= a.s[seg].n; ++seg; }
    if (seg >= a.nseg) return;
    const MVSegB& s = a.s[seg];
    const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(s.w + (size_t)row * a.K);
    float acc = 0.0f;
    for (int i = threadIdx.x & 31; i < a.K / 2; i += 32) {
        const float2 w = __bfloat1622float2(r[i]);
        acc += w.x * s.x[2 * i] + w.y * s.x[2 * i + 1];
    }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) s.y[row] = acc;
}

// ---------------------------------------------------------------------------------------------------------------
// NVFP4 SoA row dot, one warp. lut: shared E2M1x2 table.
__device__ __forceinline__ float warp_dot_fp4s(const uint8_t* __restrict__ qs, const uint8_t* __restrict__ sc,
                                               const float* __restrict__ x, int K, const float* lut) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    const int nchunk = K / 32;  // 16 bytes = 32 values = 2 sub-blocks
    for (int c = lane; c < nchunk; c += 32) {
        const uint4 q = *reinterpret_cast<const uint4*>(qs + c * 16);
        const uint16_t s2 = *reinterpret_cast<const uint16_t*>(sc + c * 2);
        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {  // sub-block 2c + h: bytes 8h..8h+7
            const float d = ue4m3_to_f32((uint8_t)(s2 >> (8 * h)));
            const float* xb = x + c * 32 + h * 16;
            const float4 xl0 = *reinterpret_cast<const float4*>(xb), xl1 = *reinterpret_cast<const float4*>(xb + 4);
            const float4 xh0 = *reinterpret_cast<const float4*>(xb + 8), xh1 = *reinterpret_cast<const float4*>(xb + 12);
            const float xl[8] = {xl0.x, xl0.y, xl0.z, xl0.w, xl1.x, xl1.y, xl1.z, xl1.w};
            const float xh[8] = {xh0.x, xh0.y, xh0.z, xh0.w, xh1.x, xh1.y, xh1.z, xh1.w};
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t byte = (w[2 * h + (j >> 2)] >> (8 * (j & 3))) & 0xFF;
                s += lut[byte & 0xF] * xl[j] + lut[byte >> 4] * xh[j];
            }
            acc += d * s;
        }
    }
    return warp_sum(acc);
}

struct ExpW {
    const uint8_t* qs;  // fp4: [E][rows][K/2]; q8: int8 [E][rows][K]
    const uint8_t* sc;  // fp4: [E][rows][K/16]; q8: half [E][rows][K/32]
    const float* scale; // per-expert scale or null
    int fmt;            // 0 = q8, 1 = nvfp4
};

__device__ __forceinline__ float exp_row_dot(const ExpW& w, size_t row, const float* x, int K, const float* lut) {
    if (w.fmt == 1) return warp_dot_fp4s(w.qs + row * (K / 2), w.sc + row * (K / 16), x, K, lut);
    return warp_dot_q8s(reinterpret_cast<const int8_t*>(w.qs) + row * K, reinterpret_cast<const __half*>(w.sc) + row * (K / 32), x, K);
}

__device__ __forceinline__ void load_lut(float* lut) {
    if (threadIdx.x < 16) {
        const float v[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        lut[threadIdx.x] = v[threadIdx.x];
    }
    __syncthreads();
}

// h[t][i] = situ(s*g_i, s*u_i); grid (F/8, k). Rows i (gate) and F+i (up) of expert sel[t].
INK_EXTERN void ink_moe_gate_up2(ExpW w, const float* __restrict__ x, const int* __restrict__ sel, float* __restrict__ h,
                                 int K, int F, float beta, float lb) {
    __shared__ float lut[16];
    load_lut(lut);
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (i >= F) return;
    const int t = blockIdx.y, e = sel[t];
    const float s = w.scale ? w.scale[e] : 1.0f;
    const float g = s * exp_row_dot(w, (size_t)e * 2 * F + i, x, K, lut);
    const float u = s * exp_row_dot(w, (size_t)e * 2 * F + F + i, x, K, lut);
    if ((threadIdx.x & 31) == 0) {
        const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
        h[t * F + i] = a * (lb * tanhf(u / lb));
    }
}

// y[n] = sum_t w_t s_e (W_e[n] · h_t); grid D/8.
INK_EXTERN void ink_moe_down2(ExpW w, const float* __restrict__ h, const int* __restrict__ sel, const float* __restrict__ wts,
                              float* __restrict__ y, int F, int D, int k) {
    __shared__ float lut[16];
    load_lut(lut);
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < k; ++t) {
        const int e = sel[t];
        const float s = w.scale ? w.scale[e] : 1.0f;
        acc += wts[t] * s * exp_row_dot(w, (size_t)e * D + n, h + t * F, F, lut);
    }
    if ((threadIdx.x & 31) == 0) y[n] = acc;
}

// ---------------------------------------------------------------------------------------------------------------
// v3 expert kernels: shared LUTs (byte -> two E2M1 values, UE4M3 byte -> scale), activations staged in shared
// memory, all loads of a lane issued before the math.
struct FP4Lut {
    float2 pair[256];  // byte -> (low nibble value, high nibble value), doubled E2M1
    float sc[256];     // UE4M3 -> float (includes the 0.5 factor)
};

__device__ __forceinline__ void fp4lut_init(FP4Lut* L) {
    const float v[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        L->pair[i] = make_float2(v[i & 15], v[i >> 4]);
        L->sc[i] = ue4m3_to_f32((uint8_t)i);
    }
}

// Dot of one 16-byte chunk (2 sub-blocks, 32 values) with x (shared), x offset = chunk * 32.
__device__ __forceinline__ float fp4_chunk_dot(uint4 q, uint16_t s2, const float* __restrict__ xs, const FP4Lut* L) {
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float* xl = xs + h * 16;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 p = L->pair[(w[2 * h + (j >> 2)] >> (8 * (j & 3))) & 0xFF];
            s = fmaf(p.x, xl[j], s);
            s = fmaf(p.y, xl[j + 8], s);
        }
        acc = fmaf(L->sc[(s2 >> (8 * h)) & 0xFF], s, acc);
    }
    return acc;
}

template <int K>
__device__ __forceinline__ float warp_dot_fp4_v3(const uint8_t* __restrict__ qs, const uint8_t* __restrict__ sc,
                                                 const float* __restrict__ xs, const FP4Lut* L) {
    constexpr int NC = K / 32;            // chunks per row
    constexpr int PER = (NC + 31) / 32;   // chunks per lane (ceil)
    const int lane = threadIdx.x & 31;
    uint4 q[PER];
    uint16_t s[PER];
#pragma unroll
    for (int t = 0; t < PER; ++t) {
        const int c = lane + 32 * t;
        if (c < NC) {
            q[t] = __ldg(reinterpret_cast<const uint4*>(qs) + c);
            s[t] = __ldg(reinterpret_cast<const unsigned short*>(sc) + c);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < PER; ++t) {
        const int c = lane + 32 * t;
        if (c < NC) acc += fp4_chunk_dot(q[t], s[t], xs + c * 32, L);
    }
    return warp_sum(acc);
}

template <int K>
__device__ __forceinline__ float warp_dot_q8_v3(const int8_t* __restrict__ qs, const __half* __restrict__ d, const float* __restrict__ xs) {
    constexpr int NC = K / 16;
    constexpr int PER = (NC + 31) / 32;
    const int lane = threadIdx.x & 31;
    int4 q[PER];
    float sc[PER];
#pragma unroll
    for (int t = 0; t < PER; ++t) {
        const int c = lane + 32 * t;
        if (c < NC) {
            q[t] = __ldg(reinterpret_cast<const int4*>(qs) + c);
            sc[t] = __half2float(d[c >> 1]);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < PER; ++t) {
        const int c = lane + 32 * t;
        if (c < NC) {
            const int8_t* b = reinterpret_cast<const int8_t*>(&q[t]);
            const float* xv = xs + c * 16;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) s = fmaf((float)b[j], xv[j], s);
            acc = fmaf(sc[t], s, acc);
        }
    }
    return warp_sum(acc);
}

template <int K>
__device__ __forceinline__ float exp_row_dot_v3(const ExpW& w, size_t row, const float* xs, const FP4Lut* L) {
    if (w.fmt == 1) return warp_dot_fp4_v3<K>(w.qs + row * (K / 2), w.sc + row * (K / 16), xs, L);
    return warp_dot_q8_v3<K>(reinterpret_cast<const int8_t*>(w.qs) + row * K, reinterpret_cast<const __half*>(w.sc) + row * (K / 32), xs);
}

// gate_up for D = 2304 inputs, F = 1152 features: grid (F/8, k), 8 warps, warp per feature (gate + up rows).
INK_EXTERN void ink_moe_gate_up3(ExpW w, const float* __restrict__ x, const int* __restrict__ sel, float* __restrict__ h,
                                 int F, float beta, float lb) {
    constexpr int K = 2304;
    __shared__ FP4Lut L;
    __shared__ __align__(16) float xs[K];
    fp4lut_init(&L);
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x) reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    __syncthreads();
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (i >= F) return;
    const int t = blockIdx.y, e = sel[t];
    const float s = w.scale ? w.scale[e] : 1.0f;
    const float g = s * exp_row_dot_v3<K>(w, (size_t)e * 2 * F + i, xs, &L);
    const float u = s * exp_row_dot_v3<K>(w, (size_t)e * 2 * F + F + i, xs, &L);
    if ((threadIdx.x & 31) == 0) {
        const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
        h[t * F + i] = a * (lb * tanhf(u / lb));
    }
}

// down for F = 1152 inputs per expert, D outputs: grid D/8, warp per output row, all k experts.
INK_EXTERN void ink_moe_down3(ExpW w, const float* __restrict__ h, const int* __restrict__ sel, const float* __restrict__ wts,
                              float* __restrict__ y, int D, int k) {
    constexpr int F = 1152;
    __shared__ FP4Lut L;
    __shared__ __align__(16) float hs[8 * F];
    fp4lut_init(&L);
    for (int i = threadIdx.x; i < k * F / 4; i += blockDim.x) reinterpret_cast<float4*>(hs)[i] = reinterpret_cast<const float4*>(h)[i];
    __syncthreads();
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < k; ++t) {
        const int e = sel[t];
        const float s = w.scale ? w.scale[e] : 1.0f;
        acc += wts[t] * s * exp_row_dot_v3<F>(w, (size_t)e * D + n, hs + t * F, &L);
    }
    if ((threadIdx.x & 31) == 0) y[n] = acc;
}

// v4: more work per block. gate_up: grid (F/32, k), each warp computes 4 features (8 row dots).
INK_EXTERN void ink_moe_gate_up4(ExpW w, const float* __restrict__ x, const int* __restrict__ sel, float* __restrict__ h,
                                 int F, float beta, float lb) {
    constexpr int K = 2304, FPW = 4;
    __shared__ FP4Lut L;
    __shared__ __align__(16) float xs[K];
    fp4lut_init(&L);
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x) reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    __syncthreads();
    const int t = blockIdx.y, e = sel[t];
    const float s = w.scale ? w.scale[e] : 1.0f;
    const int f0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * FPW;
#pragma unroll 1
    for (int j = 0; j < FPW; ++j) {
        const int i = f0 + j;
        if (i >= F) break;
        const float g = s * exp_row_dot_v3<K>(w, (size_t)e * 2 * F + i, xs, &L);
        const float u = s * exp_row_dot_v3<K>(w, (size_t)e * 2 * F + F + i, xs, &L);
        if ((threadIdx.x & 31) == 0) {
            const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
            h[t * F + i] = a * (lb * tanhf(u / lb));
        }
    }
}

// down: grid (D/32, k); block = one selected expert, 32 output rows (4 per warp). Writes weighted partials
// part[t][n] = w_t * s_e * (W_e[n] · h_t); the k partials are summed (in order) by ink_conv_res.
INK_EXTERN void ink_moe_down4(ExpW w, const float* __restrict__ h, const int* __restrict__ sel, const float* __restrict__ wts,
                              float* __restrict__ part, int D) {
    constexpr int F = 1152, RPW = 4;
    __shared__ FP4Lut L;
    __shared__ __align__(16) float hs[F];
    const int t = blockIdx.y, e = sel[t];
    fp4lut_init(&L);
    for (int i = threadIdx.x; i < F / 4; i += blockDim.x) reinterpret_cast<float4*>(hs)[i] = reinterpret_cast<const float4*>(h + t * F)[i];
    __syncthreads();
    const float sw = wts[t] * (w.scale ? w.scale[e] : 1.0f);
    const int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RPW;
#pragma unroll 1
    for (int j = 0; j < RPW; ++j) {
        const int n = n0 + j;
        if (n >= D) break;
        const float v = exp_row_dot_v3<F>(w, (size_t)e * D + n, hs, &L);
        if ((threadIdx.x & 31) == 0) part[t * D + n] = sw * v;
    }
}

// Short conv + residual: xin = sum of nparts inputs (stride apart), o = (xin + conv(xin)) / sqrt(5);
// mode 0: resid = o, mode 1: resid += o. Optional copy of o to dbg.
INK_EXTERN void ink_conv_res(const float* __restrict__ xin, int nparts, int stride, const float* __restrict__ w,
                             float* __restrict__ state, float* __restrict__ resid, float* __restrict__ dbg, int mode, int ch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float xv = 0.0f;
    for (int p = 0; p < nparts; ++p) xv += xin[p * stride + c];
    const float s0 = state[c], s1 = state[ch + c], s2 = state[2 * ch + c];
    const float* wc = w + (size_t)c * 4;
    const float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + xv * wc[3];
    const float o = (xv + y) * rsqrtf(5.0f);
    state[c] = s1;
    state[ch + c] = s2;
    state[2 * ch + c] = xv;
    if (dbg) dbg[c] = o;
    resid[c] = mode ? resid[c] + o : o;
}

// ---------------------------------------------------------------------------------------------------------------
// v5 FP4 path: bank-conflict-free. Lane = one 4-byte word of a row (8 values of one 16-value sub-block); the
// activations are staged in shared memory permuted per sub-block as [0..3, 8..11, 4..7, 12..15] so word w reads
// the 8 contiguous floats xp[8w .. 8w+8). E2M1 decoding uses __byte_perm on a register table (no shared LUT).

// position in the permuted sub-block layout for original index o (involution: swaps the middle quarters)
__device__ __forceinline__ int perm16(int o) {
    const int q = (o >> 2) & 3;
    return (o & ~15) | (((q == 1) ? 2 : (q == 2) ? 1 : q) << 2) | (o & 3);
}

// UE4M3 (with the 0.5 NVFP4 factor) via bit math
__device__ __forceinline__ float ue4m3_fast(uint32_t x) {
    const uint32_t e = (x >> 3) & 15, m = x & 7;
    float f = __uint_as_float(((e + 119u) << 23) | (m << 20));  // 2^(e-8) (1 + m/8)
    f = e == 0 ? (float)m * (1.0f / 1024.0f) : f;
    return (x == 0 || x == 0x7F) ? 0.0f : f;
}

// 8 nibble indices (4 bytes) -> .x: int8 values of the low nibbles, .y: of the high nibbles (byte order kept)
__device__ __forceinline__ int2 e2m1x2_gather(uint32_t q) {
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t0, t1, q >> sh);
        const uint32_t hi = __byte_perm(t2, t3, q >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

template <int K>
__device__ __forceinline__ float warp_dot_fp4_v5(const uint8_t* __restrict__ qs, const uint8_t* __restrict__ sc,
                                                 const float* __restrict__ xp) {
    constexpr int NW = K / 8, IT = (NW + 31) / 32;
    const int lane = threadIdx.x & 31;
    uint32_t q[IT], s[IT];
#pragma unroll
    for (int t = 0; t < IT; ++t) {
        const int w = lane + 32 * t;
        if (w < NW) {
            q[t] = __ldg(reinterpret_cast<const unsigned int*>(qs) + w);
            s[t] = __ldg(sc + (w >> 1));
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < IT; ++t) {
        const int w = lane + 32 * t;
        if (w < NW) {
            const int2 v = e2m1x2_gather(q[t]);
            const float4 xl = *reinterpret_cast<const float4*>(xp + 8 * w);
            const float4 xh = *reinterpret_cast<const float4*>(xp + 8 * w + 4);
            float d = (float)(int8_t)(v.x) * xl.x + (float)(int8_t)(v.x >> 8) * xl.y + (float)(int8_t)(v.x >> 16) * xl.z +
                      (float)(int8_t)(v.x >> 24) * xl.w;
            d += (float)(int8_t)(v.y) * xh.x + (float)(int8_t)(v.y >> 8) * xh.y + (float)(int8_t)(v.y >> 16) * xh.z +
                 (float)(int8_t)(v.y >> 24) * xh.w;
            acc = fmaf(ue4m3_fast(s[t]), d, acc);
        }
    }
    return warp_sum(acc);
}

template <int K>
__device__ __forceinline__ float exp_row_dot_v5(const ExpW& w, size_t row, const float* xs) {
    if (w.fmt == 1) return warp_dot_fp4_v5<K>(w.qs + row * (K / 2), w.sc + row * (K / 16), xs);
    return warp_dot_q8_v3<K>(reinterpret_cast<const int8_t*>(w.qs) + row * K, reinterpret_cast<const __half*>(w.sc) + row * (K / 32), xs);
}

// stage n floats from global into shared, permuted for the FP4 path (plain copy for Q8)
__device__ __forceinline__ void stage_x(float* dst, const float* __restrict__ src, int n, bool perm) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[perm ? perm16(i) : i] = src[i];
}

INK_EXTERN void ink_moe_gate_up5(ExpW w, const float* __restrict__ x, const int* __restrict__ sel, float* __restrict__ h,
                                 int F, float beta, float lb) {
    constexpr int K = 2304, FPW = 4;
    __shared__ __align__(16) float xs[K];
    stage_x(xs, x, K, w.fmt == 1);
    __syncthreads();
    const int t = blockIdx.y, e = sel[t];
    const float s = w.scale ? w.scale[e] : 1.0f;
    const int f0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * FPW;
#pragma unroll 1
    for (int j = 0; j < FPW; ++j) {
        const int i = f0 + j;
        if (i >= F) break;
        const float g = s * exp_row_dot_v5<K>(w, (size_t)e * 2 * F + i, xs);
        const float u = s * exp_row_dot_v5<K>(w, (size_t)e * 2 * F + F + i, xs);
        if ((threadIdx.x & 31) == 0) {
            const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
            h[t * F + i] = a * (lb * tanhf(u / lb));
        }
    }
}

INK_EXTERN void ink_moe_down5(ExpW w, const float* __restrict__ h, const int* __restrict__ sel, const float* __restrict__ wts,
                              float* __restrict__ part, int D) {
    constexpr int F = 1152, RPW = 4;
    __shared__ __align__(16) float hs[F];
    const int t = blockIdx.y, e = sel[t];
    stage_x(hs, h + t * F, F, w.fmt == 1);
    __syncthreads();
    const float sw = wts[t] * (w.scale ? w.scale[e] : 1.0f);
    const int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RPW;
#pragma unroll 1
    for (int j = 0; j < RPW; ++j) {
        const int n = n0 + j;
        if (n >= D) break;
        const float v = exp_row_dot_v5<F>(w, (size_t)e * D + n, hs);
        if ((threadIdx.x & 31) == 0) part[t * D + n] = sw * v;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Loop control, two phases over up to 4 controls at once.
struct Ctrl4 {
    const float* c[4];
    const float* m[4];
    const float* wd[4];  // [R][2D]
    const float* wg[4];  // [D][R]
    const float* wf[4];
    float* g[4];         // [R] scratch
    float* out[4];
    int n;
};

__device__ float block_sum2(float v) {
    __shared__ float s[32];
    __shared__ float total;
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) s[w] = v;
    __syncthreads();
    if (w == 0) {
        float r = l < (int)(blockDim.x >> 5) ? s[l] : 0.0f;
        r = warp_sum(r);
        if (l == 0) total = r;
    }
    __syncthreads();
    return total;
}

// phase 1: g_j = Wd_j [rms(c_j); rms(m_j)]; grid (R/8, n) with 8 warps per block
INK_EXTERN void ink_ctrl_down(Ctrl4 a, int D, int R) {
    const int j = blockIdx.y;
    const float* c = a.c[j];
    const float* m = a.m[j];
    float sc = 0.0f, sm = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { sc += c[i] * c[i]; sm += m[i] * m[i]; }
    const float ic = rsqrtf(block_sum2(sc) / D + 1e-6f);
    const float im = rsqrtf(block_sum2(sm) / D + 1e-6f);
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= R) return;
    const float4* row = reinterpret_cast<const float4*>(a.wd[j] + (size_t)r * 2 * D);
    const float4* c4 = reinterpret_cast<const float4*>(c);
    const float4* m4 = reinterpret_cast<const float4*>(m);
    float acc = 0.0f;
    for (int i = lane; i < D / 4; i += 32) {
        const float4 w0 = row[i], w1 = row[D / 4 + i], cv = c4[i], mv = m4[i];
        acc += (w0.x * cv.x + w0.y * cv.y + w0.z * cv.z + w0.w * cv.w) * ic + (w1.x * mv.x + w1.y * mv.y + w1.z * mv.z + w1.w * mv.w) * im;
    }
    acc = warp_sum(acc);
    if (lane == 0) a.g[j][r] = acc;
}

// phase 2: out_j = (1 - tanh(Wf g)) c + tanh(Wg g) m; grid (D/8, n), warp per output row, R = 64
INK_EXTERN void ink_ctrl_up(Ctrl4 a, int D, int R) {
    const int j = blockIdx.y;
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= D) return;
    const float* g = a.g[j];
    const float* wg = a.wg[j] + (size_t)i * R;
    const float* wf = a.wf[j] + (size_t)i * R;
    float ga = 0.0f, fa = 0.0f;
    for (int r = lane; r < R; r += 32) { ga += wg[r] * g[r]; fa += wf[r] * g[r]; }
    ga = warp_sum(ga);
    fa = warp_sum(fa);
    if (lane == 0) a.out[j][i] = (1.0f - tanhf(fa)) * a.c[j][i] + tanhf(ga) * a.m[j][i];
}

// ---------------------------------------------------------------------------------------------------------------
// Depth attention in one pass: all states' norms and query dots reduced together.
// Optionally writes rmsnorm(out_q) * nw[q] (nw[q] != null) to outn + q*D.
struct MixArgs {
    const float* st[8];
    const float* q;     // [nq][D]
    float* out;         // [nq][D]
    float* outn;        // [nq][D] or null
    const float* nw[4]; // post-norm weights per query or null
    int n;
    int nq;
    int D;
    float eps;
};

INK_EXTERN void ink_mix2(MixArgs a) {
    // partial sums: [n] squared norms, [nq*n] dots
    __shared__ float red[32][40];
    __shared__ float tot[40];
    __shared__ float prob[4][8];
    __shared__ float inv[8];
    __shared__ float onorm[4];
    const int n = a.n, nq = a.nq, D = a.D, nv = n + nq * n;
    float part[40];
#pragma unroll
    for (int t = 0; t < 40; ++t) part[t] = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float qv[4];
        for (int q = 0; q < nq; ++q) qv[q] = a.q[q * D + i];
        for (int s = 0; s < n; ++s) {
            const float v = a.st[s][i];
            part[s] += v * v;
            for (int q = 0; q < nq; ++q) part[n + q * n + s] += qv[q] * v;
        }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int t = 0; t < nv; ++t) {
        const float v = warp_sum(part[t]);
        if (l == 0) red[w][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < nv) {
        float s = 0.0f;
        for (int ww = 0; ww < (int)(blockDim.x >> 5); ++ww) s += red[ww][threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
    if (threadIdx.x < n) inv[threadIdx.x] = rsqrtf(tot[threadIdx.x] / D + 1e-6f);
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        const float scale = rsqrtf((float)D);
        float mx = -INFINITY;
        for (int s = 0; s < n; ++s) { prob[q][s] = tot[n + q * n + s] * inv[s] * scale; mx = fmaxf(mx, prob[q][s]); }
        float sum = 0.0f;
        for (int s = 0; s < n; ++s) { prob[q][s] = __expf(prob[q][s] - mx); sum += prob[q][s]; }
        for (int s = 0; s < n; ++s) prob[q][s] = prob[q][s] / sum * inv[s];
    }
    __syncthreads();
    float osq[4] = {0, 0, 0, 0};
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v[8];
        for (int s = 0; s < n; ++s) v[s] = a.st[s][i];
        for (int q = 0; q < nq; ++q) {
            float acc = 0.0f;
            for (int s = 0; s < n; ++s) acc += prob[q][s] * v[s];
            a.out[q * D + i] = acc;
            osq[q] += acc * acc;
        }
    }
    if (!a.outn) return;
    for (int q = 0; q < nq; ++q) {
        const float t = block_sum2(osq[q]);
        if (threadIdx.x == 0) onorm[q] = rsqrtf(t / D + a.eps);
    }
    __syncthreads();
    for (int q = 0; q < nq; ++q) {
        if (!a.nw[q]) continue;
        for (int i = threadIdx.x; i < D; i += blockDim.x) a.outn[q * D + i] = a.out[q * D + i] * onorm[q] * a.nw[q][i];
    }
}

// v3 depth attention: compile-time maxima (6 states, 4 queries) so all partials stay in registers.
INK_EXTERN void ink_mix3(MixArgs a) {
    constexpr int MS = 6, MQ = 4, NV = MS + MQ * MS;
    __shared__ float red[16][NV];
    __shared__ float prob[MQ][MS];
    __shared__ float onorm[MQ];
    const int n = a.n, nq = a.nq, D = a.D;
    float part[NV];
#pragma unroll
    for (int t = 0; t < NV; ++t) part[t] = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v[MS], qv[MQ];
#pragma unroll
        for (int s = 0; s < MS; ++s) v[s] = s < n ? a.st[s][i] : 0.0f;
#pragma unroll
        for (int q = 0; q < MQ; ++q) qv[q] = q < nq ? a.q[q * D + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < MS; ++s) {
            part[s] = fmaf(v[s], v[s], part[s]);
#pragma unroll
            for (int q = 0; q < MQ; ++q) part[MS + q * MS + s] = fmaf(qv[q], v[s], part[MS + q * MS + s]);
        }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < NV; ++t) {
        const float v = warp_sum(part[t]);
        if (l == 0) red[w][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x, nw = blockDim.x >> 5;
        float inv[MS], sc[MS], mx = -INFINITY;
        for (int s = 0; s < n; ++s) {
            float ss = 0.0f, dd = 0.0f;
            for (int ww = 0; ww < nw; ++ww) { ss += red[ww][s]; dd += red[ww][MS + q * MS + s]; }
            inv[s] = rsqrtf(ss / D + 1e-6f);
            sc[s] = dd * inv[s] * rsqrtf((float)D);
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < n; ++s) { sc[s] = __expf(sc[s] - mx); sum += sc[s]; }
        for (int s = 0; s < MS; ++s) prob[q][s] = s < n ? sc[s] / sum * inv[s] : 0.0f;
    }
    __syncthreads();
    float osq[MQ] = {0, 0, 0, 0};
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v[MS];
#pragma unroll
        for (int s = 0; s < MS; ++s) v[s] = s < n ? a.st[s][i] : 0.0f;
#pragma unroll
        for (int q = 0; q < MQ; ++q) {
            if (q < nq) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < MS; ++s) acc = fmaf(prob[q][s], v[s], acc);
                a.out[q * D + i] = acc;
                osq[q] = fmaf(acc, acc, osq[q]);
            }
        }
    }
    if (!a.outn) return;
    __syncthreads();
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
        const float v = warp_sum(osq[q]);
        if (l == 0) red[w][q] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        float ss = 0.0f;
        for (int ww = 0; ww < (int)(blockDim.x >> 5); ++ww) ss += red[ww][threadIdx.x];
        onorm[threadIdx.x] = rsqrtf(ss / D + a.eps);
    }
    __syncthreads();
    for (int q = 0; q < nq; ++q) {
        if (!a.nw[q]) continue;
        for (int i = threadIdx.x; i < D; i += blockDim.x) a.outn[q * D + i] = a.out[q * D + i] * onorm[q] * a.nw[q][i];
    }
}

// Router top-k on precomputed logits (one warp): p = sigmoid, select by p + bias, w = p/sum * scale.
INK_EXTERN void ink_topk(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ sel,
                         float* __restrict__ wts, int E, int k, float wscale) {
    const int lane = threadIdx.x;
    float p[4], s[4];  // E <= 128: 4 per lane
    for (int t = 0; t < 4; ++t) {
        const int e = lane + 32 * t;
        p[t] = e < E ? 1.0f / (1.0f + __expf(-logits[e])) : 0.0f;
        s[t] = e < E ? p[t] + (bias ? bias[e] : 0.0f) : -INFINITY;
    }
    float sum = 0.0f, mine = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = s[0];
        int bi = lane;
        for (int t = 1; t < 4; ++t) if (s[t] > bv) { bv = s[t]; bi = lane + 32 * t; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const float pb = __shfl_sync(0xffffffffu, p[bi >> 5], bi & 31);
        if ((bi & 31) == lane) s[bi >> 5] = -INFINITY;
        sum += pb;
        if (lane == r) mine = pb;
        if (lane == 0) sel[r] = bi;
    }
    if (lane < k) wts[lane] = mine / (sum + 1e-20f) * wscale;
}
