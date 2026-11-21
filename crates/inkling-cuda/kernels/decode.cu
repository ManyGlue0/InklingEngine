// M4 decode kernels on repacked weights.
//
// Repacked layouts (built on load, see inkling-engine/src/repack.rs):
//   Q8 "SoA":   qs int8 [rows][K] (16B aligned rows), d half [rows][K/32]
//   NVFP4 SoA:  qs u8 [rows][K/2] (sub-block s = bytes 8s..8s+7: low nibble -> 16s+j, high -> 16s+j+8),
//               sc u8 [rows][K/16] (UE4M3, one per 16 values)
#include "quant.cuh"

// Depth-attention arguments (shared by the mix kernels).
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

extern "C" __global__ void __launch_bounds__(256, 3) ink_moe_gate_up5(ExpW w, const float* __restrict__ x, const int* __restrict__ sel, float* __restrict__ h,
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

extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_down5(ExpW w, const float* __restrict__ h, const int* __restrict__ sel, const float* __restrict__ wts,
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
// v6: int8 activations (q8_1-style blocks of 32: d = amax/127, q = round(x/d), as llama.cpp's mmvq) + dp4a.

// One warp per 32-value block: xq[i] = round(x[i]/d), xd[b] = d.
INK_EXTERN void ink_quant_q8_1(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xd, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.0f;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    if (i < n) {
        xq[i] = amax == 0.0f ? 0 : (int8_t)roundf(v / d);
        if ((threadIdx.x & 31) == 0) xd[i >> 5] = d;
    }
}

// stage int8 activations into shared memory, permuted per 16-value sub-block for the FP4 path
__device__ __forceinline__ void stage_q(int8_t* dst, const int8_t* __restrict__ src, int n, bool perm) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[perm ? perm16(i) : i] = src[i];
}

// FP4 row · int8 activations. xq permuted (word w -> 8 bytes at 8w: 4 "low" then 4 "high"), xd per 32 values.
template <int K>
__device__ __forceinline__ float warp_dot_fp4_q8(const uint8_t* __restrict__ qs, const uint8_t* __restrict__ sc,
                                                 const int8_t* __restrict__ xq, const float* __restrict__ xd) {
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
            const int2 x = *reinterpret_cast<const int2*>(xq + 8 * w);
            const int isum = __dp4a(v.y, x.y, __dp4a(v.x, x.x, 0));
            acc = fmaf((float)isum, ue4m3_fast(s[t]) * xd[w >> 2], acc);  // 4 words per 32-value x block
        }
    }
    return warp_sum(acc);
}

// Q8 row (SoA) · int8 activations (unpermuted). 16 values per lane step.
template <int K>
__device__ __forceinline__ float warp_dot_q8_q8(const int8_t* __restrict__ qs, const __half* __restrict__ d,
                                                const int8_t* __restrict__ xq, const float* __restrict__ xd) {
    constexpr int NC = K / 16, IT = (NC + 31) / 32;
    const int lane = threadIdx.x & 31;
    int4 q[IT];
    float sc[IT];
#pragma unroll
    for (int t = 0; t < IT; ++t) {
        const int c = lane + 32 * t;
        if (c < NC) {
            q[t] = __ldg(reinterpret_cast<const int4*>(qs) + c);
            sc[t] = __half2float(d[c >> 1]);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < IT; ++t) {
        const int c = lane + 32 * t;
        if (c < NC) {
            const int4 x = *reinterpret_cast<const int4*>(xq + 16 * c);
            int isum = __dp4a(q[t].x, x.x, 0);
            isum = __dp4a(q[t].y, x.y, isum);
            isum = __dp4a(q[t].z, x.z, isum);
            isum = __dp4a(q[t].w, x.w, isum);
            acc = fmaf((float)isum, sc[t] * xd[c >> 1], acc);
        }
    }
    return warp_sum(acc);
}

template <int K>
__device__ __forceinline__ float exp_row_dot_q(const ExpW& w, size_t row, const int8_t* xq, const float* xd) {
    if (w.fmt == 1) return warp_dot_fp4_q8<K>(w.qs + row * (K / 2), w.sc + row * (K / 16), xq, xd);
    return warp_dot_q8_q8<K>(reinterpret_cast<const int8_t*>(w.qs) + row * K, reinterpret_cast<const __half*>(w.sc) + row * (K / 32), xq, xd);
}

// gate_up: grid (F/32, k); block = 32 consecutive features of one expert (4 per warp) = one q8_1 block of h,
// so the block quantizes its own output: hq int8 [k][F], hd [k][F/32].
extern "C" __global__ void __launch_bounds__(256, 3) ink_moe_gate_up6(ExpW w, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                                      const int* __restrict__ sel, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                                      int F, float beta, float lb) {
    constexpr int K = 2304, FPW = 4;
    __shared__ __align__(16) int8_t xs[K];
    __shared__ float xds[K / 32];
    __shared__ float hv[32];
    stage_q(xs, xq, K, w.fmt == 1);
    for (int i = threadIdx.x; i < K / 32; i += blockDim.x) xds[i] = xd[i];
    __syncthreads();
    const int t = blockIdx.y, e = sel[t];
    const float s = w.scale ? w.scale[e] : 1.0f;
    const int warp = threadIdx.x >> 5;
#pragma unroll 1
    for (int j = 0; j < FPW; ++j) {
        const int i = (blockIdx.x * 8 + warp) * FPW + j;
        const float g = s * exp_row_dot_q<K>(w, (size_t)e * 2 * F + i, xs, xds);
        const float u = s * exp_row_dot_q<K>(w, (size_t)e * 2 * F + F + i, xs, xds);
        if ((threadIdx.x & 31) == 0) {
            const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
            hv[warp * FPW + j] = a * (lb * tanhf(u / lb));
        }
    }
    __syncthreads();
    if (warp == 0) {
        const int lane = threadIdx.x;
        const float v = hv[lane];
        const float amax = warp_max(fabsf(v));
        const float d = amax / 127.0f;
        hq[t * F + blockIdx.x * 32 + lane] = amax == 0.0f ? 0 : (int8_t)roundf(v / d);
        if (lane == 0) hd[t * (F / 32) + blockIdx.x] = d;
    }
}

// down: grid (D/32, k); block = one selected expert, 32 output rows. part[t][n] = w_t s_e (W_e[n] · h_t).
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_down6(ExpW w, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                                                                   const int* __restrict__ sel, const float* __restrict__ wts,
                                                                   float* __restrict__ part, int D) {
    constexpr int F = 1152, RPW = 4;
    __shared__ __align__(16) int8_t hs[F];
    __shared__ float hds[F / 32];
    const int t = blockIdx.y, e = sel[t];
    stage_q(hs, hq + t * F, F, w.fmt == 1);
    for (int i = threadIdx.x; i < F / 32; i += blockDim.x) hds[i] = hd[t * (F / 32) + i];
    __syncthreads();
    const float sw = wts[t] * (w.scale ? w.scale[e] : 1.0f);
    const int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RPW;
#pragma unroll 1
    for (int j = 0; j < RPW; ++j) {
        const int n = n0 + j;
        const float v = exp_row_dot_q<F>(w, (size_t)e * D + n, hs, hds);
        if ((threadIdx.x & 31) == 0) part[t * D + n] = sw * v;
    }
}

// Greedy argmax over n logits, one block of 1024 threads; ties -> lowest index. Writes out[0] = index.
INK_EXTERN void ink_argmax(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float t = x[i];
        if (t > v) { v = t; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x < 32) {
        v = bv[threadIdx.x];
        idx = bi[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (threadIdx.x == 0) out[0] = idx;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Fused decode kernels (pass 7).

__device__ float block_sum2(float v);

// q8_1-quantize 32 consecutive values held one per lane of a warp; lane 0 writes the scale.
__device__ __forceinline__ void warp_quant32(float v, int8_t* q, float* d) {
    const float amax = warp_max(fabsf(v));
    const float s = amax / 127.0f;
    q[threadIdx.x & 31] = amax == 0.0f ? 0 : (int8_t)roundf(v / s);
    if ((threadIdx.x & 31) == 0) *d = s;
}

// Multi-segment Q8 matvec on int8 activations (dp4a).
struct MVSegQ {
    const int8_t* qs;
    const __half* d;
    const int8_t* xq;
    const float* xd;
    float* y;
    int n;
};
struct MVQ4 {
    MVSegQ s[4];
    int nseg;
    int K;
};

INK_EXTERN void ink_mv_q8q(MVQ4 a) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    int seg = 0;
    while (seg < a.nseg && row >= a.s[seg].n) { row -= a.s[seg].n; ++seg; }
    if (seg >= a.nseg) return;
    const MVSegQ& s = a.s[seg];
    const int K = a.K, lane = threadIdx.x & 31;
    const int8_t* qs = s.qs + (size_t)row * K;
    const __half* d = s.d + (size_t)row * (K / 32);
    float acc = 0.0f;
    for (int c = lane; c < K / 16; c += 32) {
        const int4 q = __ldg(reinterpret_cast<const int4*>(qs) + c);
        const int4 x = *reinterpret_cast<const int4*>(s.xq + 16 * c);
        int is = __dp4a(q.x, x.x, 0);
        is = __dp4a(q.y, x.y, is);
        is = __dp4a(q.z, x.z, is);
        is = __dp4a(q.w, x.w, is);
        acc = fmaf((float)is, __half2float(d[c >> 1]) * s.xd[c >> 1], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) s.y[row] = acc;
}

// Depth attention (as ink_mix3) that can also emit the q8_1-quantized normalized outputs.
struct MixArgsQ {
    MixArgs m;
    int8_t* xq;  // [nq][D] or null
    float* xd;   // [nq][D/32]
};

INK_EXTERN void ink_mix4(MixArgsQ A) {
    const MixArgs& a = A.m;
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
        // D is a multiple of blockDim.x? not required: each warp quantizes whole 32-blocks
        for (int base = w * 32; base < D; base += blockDim.x) {
            const int i = base + l;
            const float v = a.out[q * D + i] * onorm[q] * a.nw[q][i];
            a.outn[q * D + i] = v;
            if (A.xq) warp_quant32(v, A.xq + q * D + base, A.xd + q * (D / 32) + base / 32);
        }
    }
}

// RMS-normalize up to 4 vectors (one block each) with weights, writing y_j and q8_1 (xq_j, xd_j).
struct NormQ4 {
    const float* x[4];
    const float* w[4];
    float* y[4];
    int8_t* xq[4];
    float* xd[4];
};

INK_EXTERN void ink_rmsnorm_q(NormQ4 a, int D, float eps) {
    const int j = blockIdx.x;
    const float* x = a.x[j];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
    const float inv = rsqrtf(block_sum2(ss) / D + eps);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int base = w * 32; base < D; base += blockDim.x) {
        const int i = base + l;
        const float v = x[i] * inv * a.w[j][i];
        if (a.y[j]) a.y[j][i] = v;
        if (a.xq[j]) warp_quant32(v, a.xq[j] + base, a.xd[j] + base / 32);
    }
}

// KV format study: quantize-dequantize v (one per lane) to `bits` (symmetric, f16 scale per 32 lanes); 0 = f16.
__device__ __forceinline__ float kv_fakequant(float v, int bits) {
    if (bits == 0) return v;
    const float amax = warp_max(fabsf(v));
    const float qmax = (float)((1 << (bits - 1)) - 1);
    const float s = __half2float(__float2half(amax / qmax));
    if (s == 0.0f) return 0.0f;
    return fminf(fmaxf(rintf(v / s), -qmax), qmax) * s;
}

// KV cache formats. fmt 0: f16 [row][8 heads x 128]. fmt 1 (q6): per row and kv head 104 bytes =
// ql[64] (low 4 bits, channel c in byte c/2, nibble c&1) | qh[32] (high 2 bits, byte c/4, bits 2(c&3)) |
// 4 f16 scales (one per 32 channels); values are q - 32 with q in 1..63 (symmetric 6-bit).
constexpr int KVQ6_HEAD = 104, KVQ6_ROW = 8 * KVQ6_HEAD;

// Store channel c (= 32 * warp + lane, one quant group per warp) of kv head h at cache row `row`.
__device__ __forceinline__ void kv_store(void* cache, int fmt, size_t row, int h, int c, float v) {
    if (fmt == 0) {
        reinterpret_cast<__half*>(cache)[row * 1024 + h * 128 + c] = __float2half(v);
        return;
    }
    uint8_t* b = reinterpret_cast<uint8_t*>(cache) + row * KVQ6_ROW + h * KVQ6_HEAD;
    const float amax = warp_max(fabsf(v));
    const __half sh = __float2half(amax / 31.0f);
    const float s = __half2float(sh);
    const int q = (s == 0.0f ? 0 : (int)fminf(fmaxf(rintf(v / s), -31.0f), 31.0f)) + 32;
    const int q1 = __shfl_down_sync(0xffffffffu, q, 1), q2 = __shfl_down_sync(0xffffffffu, q, 2), q3 = __shfl_down_sync(0xffffffffu, q, 3);
    const int lane = threadIdx.x & 31;
    if ((c & 1) == 0) b[c >> 1] = (uint8_t)((q & 15) | ((q1 & 15) << 4));
    if ((c & 3) == 0) b[64 + (c >> 2)] = (uint8_t)((q >> 4) | ((q1 >> 4) << 2) | ((q2 >> 4) << 4) | ((q3 >> 4) << 6));
    if (lane == 0) reinterpret_cast<__half*>(b + 96)[c >> 5] = sh;
}

// Dequantize channels 16p .. 16p+15 of kv head h at cache row `row` into dst (16 halves, 16-byte aligned).
__device__ __forceinline__ void kv_load16(const void* cache, int fmt, size_t row, int h, int p, __half* dst) {
    if (fmt == 0) {
        const uint4* s = reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(cache) + row * 1024 + h * 128 + 16 * p);
        reinterpret_cast<uint4*>(dst)[0] = s[0];
        reinterpret_cast<uint4*>(dst)[1] = s[1];
        return;
    }
    const uint8_t* b = reinterpret_cast<const uint8_t*>(cache) + row * KVQ6_ROW + h * KVQ6_HEAD;
    const uint2 lo = *reinterpret_cast<const uint2*>(b + 8 * p);
    const uint32_t hi = *reinterpret_cast<const uint32_t*>(b + 64 + 4 * p);
    const float s = __half2float(reinterpret_cast<const __half*>(b + 96)[p >> 1]);
    __align__(16) __half out[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t lw = i < 8 ? lo.x : lo.y;
        const int low = (lw >> (4 * (i & 7))) & 15;
        const int high = (hi >> (2 * i)) & 3;
        out[i] = __float2half((float)((low | (high << 4)) - 32) * s);
    }
    reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<const uint4*>(out)[0];
    reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<const uint4*>(out)[1];
}

// Attention prep with the K/V short convs fused. Grid 64 blocks x 128 threads:
//   0..23 q heads (clip, tau), 24..31 k heads (conv [+dyn] -> clip -> cache), 32..39 v heads (conv -> rms -> cache),
//   40..63 r heads (clip, tau).
__device__ __forceinline__ float sum128(float v) {
    __shared__ float s[4];
    v = warp_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v;
    __syncthreads();
    return s[0] + s[1] + s[2] + s[3];
}

INK_EXTERN void ink_attn_prep2(float* __restrict__ q, const float* __restrict__ kraw, const float* __restrict__ vraw, float* __restrict__ r,
                               const float* __restrict__ qclip, const float* __restrict__ rclip,
                               const float* __restrict__ kconv, const float* __restrict__ vconv, float* __restrict__ kstate, float* __restrict__ vstate,
                               const float* __restrict__ dynk, const float* __restrict__ dynv, const int* __restrict__ step,
                               void* __restrict__ Kc, void* __restrict__ Vc, int kv_cap, int is_full,
                               float qk_clip, float rel_clip, float log_floor, float log_alpha, float eps, int kbits, int vbits, int kvfmt) {
    const int task = blockIdx.x, tid = threadIdx.x;
    const int pos = step[1];
    const float tau = is_full ? 1.0f + log_alpha * logf(fmaxf(1.0f, (float)(pos + 1) / log_floor)) : 1.0f;
    const int row = pos % kv_cap;
    if (task < 24) {
        const float y = q[task * 128 + tid] * qclip[task * 128 + tid];
        const float f = fminf(1.0f, qk_clip / (sqrtf(sum128(y * y)) + 1e-6f)) * tau;
        q[task * 128 + tid] = y * f;
    } else if (task < 40) {
        const bool isk = task < 32;
        const int h = isk ? task - 24 : task - 32, c = h * 128 + tid, ch = 1024;
        const float* x = isk ? kraw : vraw;
        const float* wc = (isk ? kconv : vconv) + (size_t)c * 4;
        float* st = isk ? kstate : vstate;
        const float* dyn = isk ? dynk : dynv;
        const float s0 = st[c], s1 = st[ch + c], s2 = st[2 * ch + c], xv = x[c];
        float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + xv * wc[3];
        float scale = rsqrtf(5.0f);
        if (dyn) {
            const float* t = dyn + h * 4;
            y += xv * t[0] + s2 * t[1] + s1 * t[2] + s0 * t[3];
            scale = rsqrtf(9.0f);
        }
        st[c] = s1;
        st[ch + c] = s2;
        st[2 * ch + c] = xv;
        const float o = (xv + y) * scale;
        float f;
        if (isk) f = fminf(1.0f, qk_clip / (sqrtf(sum128(o * o)) + 1e-6f));
        else f = rsqrtf(sum128(o * o) / 128.0f + eps);
        kv_store(isk ? Kc : Vc, kvfmt, row, h, tid, kv_fakequant(o * f, isk ? kbits : vbits));
    } else {
        const int h = task - 40;
        const float y = tid < 16 ? r[h * 16 + tid] * rclip[h * 16 + tid] : 0.0f;
        const float f = fminf(1.0f, rel_clip / (sqrtf(sum128(y * y)) + 1e-6f)) * tau;
        if (tid < 16) r[h * 16 + tid] = y * f;
    }
}

// Decode attention (as ink_attn_decode) with the sigmoid output gate fused and q8_1 output for the wo projection.
INK_EXTERN void ink_attn_decode2(const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sink,
                                 const __half* __restrict__ Kc, const __half* __restrict__ Vc, const float* __restrict__ relbias,
                                 const float* __restrict__ gate, float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od,
                                 const int* __restrict__ step, int kv_cap, int window, int extent) {
    __shared__ float rs[16];
    __shared__ float wm[4], wl[4], wacc[4][128];
    const int h = blockIdx.x, kvh = h / 3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = step[1];
    if (threadIdx.x < 16) rs[threadIdx.x] = r[h * 16 + threadIdx.x];
    __syncthreads();
    float qv[4];
    for (int j = 0; j < 4; ++j) qv[j] = q[h * 128 + lane * 4 + j];
    const int j0 = window ? max(0, pos - window + 1) : 0;
    float m = -INFINITY, l = 0.0f, acc[4] = {0, 0, 0, 0};
    const float scale = rsqrtf(128.0f);
    for (int j = j0 + warp; j <= pos; j += 4) {
        const int row = j % kv_cap;
        const uint2 kr = *reinterpret_cast<const uint2*>(Kc + (size_t)row * 1024 + kvh * 128 + lane * 4);
        const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&kr.x));
        const float2 k23 = __half22float2(*reinterpret_cast<const __half2*>(&kr.y));
        float d = warp_sum(qv[0] * k01.x + qv[1] * k01.y + qv[2] * k23.x + qv[3] * k23.y) * scale;
        const int dist = pos - j;
        if (dist < extent) d += warp_sum(lane < 16 ? rs[lane] * relbias[dist * 16 + lane] : 0.0f);
        const float mn = fmaxf(m, d);
        const float corr = __expf(m - mn), p = __expf(d - mn);
        l = l * corr + p;
        const uint2 vr = *reinterpret_cast<const uint2*>(Vc + (size_t)row * 1024 + kvh * 128 + lane * 4);
        const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vr.x));
        const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&vr.y));
        acc[0] = acc[0] * corr + p * v01.x;
        acc[1] = acc[1] * corr + p * v01.y;
        acc[2] = acc[2] * corr + p * v23.x;
        acc[3] = acc[3] * corr + p * v23.y;
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    for (int t = 0; t < 4; ++t) wacc[warp][lane * 4 + t] = acc[t];
    __syncthreads();
    // 128 threads: one output dim each
    const int dim = threadIdx.x;
    const float s = sink[h];
    float M = s;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w]);
    float L = __expf(s - M), o = 0.0f;
    for (int w = 0; w < 4; ++w) {
        if (wm[w] == -INFINITY) continue;
        const float e = __expf(wm[w] - M);
        L += wl[w] * e;
        o += wacc[w][dim] * e;
    }
    const float g = gate[h * 128 + dim];
    o = o / L * (1.0f / (1.0f + __expf(-g)));
    out[h * 128 + dim] = o;
    warp_quant32(o, oq + h * 128 + warp * 32, od + h * 4 + warp);
}

// Router: logits = W x (F32 rows, one warp each, multi-block); the last block to finish does the top-k.
// counter must be 0 before the launch; the last block resets it (graph replay safe).
INK_EXTERN void ink_router2(const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias, float* __restrict__ logits,
                            int* __restrict__ sel, float* __restrict__ wts, unsigned int* __restrict__ counter, int D, int E, int k, float wscale) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * (blockDim.x >> 5) + warp;
    if (e < E) {
        const float4* row = reinterpret_cast<const float4*>(W + (size_t)e * D);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float acc = 0.0f;
        for (int i = lane; i < D / 4; i += 32) {
            const float4 a = row[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        acc = warp_sum(acc);
        if (lane == 0) logits[e] = acc;
    }
    __threadfence();
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last || warp != 0) return;
    __threadfence();
    float p[4], s[4];
    for (int t = 0; t < 4; ++t) {
        const int ee = lane + 32 * t;
        const float lg = ee < E ? *((volatile float*)logits + ee) : 0.0f;
        p[t] = ee < E ? 1.0f / (1.0f + __expf(-lg)) : 0.0f;
        s[t] = ee < E ? p[t] + (bias ? bias[ee] : 0.0f) : -INFINITY;
    }
    float sum = 0.0f, mine = 0.0f;
    for (int rr = 0; rr < k; ++rr) {
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
        if (lane == rr) mine = pb;
        if (lane == 0) sel[rr] = bi;
    }
    if (lane < k) wts[lane] = mine / (sum + 1e-20f) * wscale;
    if (lane == 0) *counter = 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Pass 8: register-cached depth attention, wider router, F16 loop-control weights.

// ink_mix5: as ink_mix4, but every thread keeps its elements of all states in registers (one global read).
// Launch with exactly 768 threads for D = 2304 (3 elements per thread).
INK_EXTERN void __launch_bounds__(768) ink_mix5(MixArgsQ A) {
    const MixArgs& a = A.m;
    constexpr int MS = 6, MQ = 4, NV = MS + MQ * MS, EPT = 3, NWARP = 24;
    __shared__ float red[NWARP][NV];
    __shared__ float prob[MQ][MS];
    __shared__ float onorm[MQ];
    const int n = a.n, nq = a.nq, D = a.D;
    float v[EPT][MS];
    float part[NV];
#pragma unroll
    for (int t = 0; t < NV; ++t) part[t] = 0.0f;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = threadIdx.x + e * 768;
        float qv[MQ];
#pragma unroll
        for (int q = 0; q < MQ; ++q) qv[q] = q < nq ? a.q[q * D + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < MS; ++s) {
            v[e][s] = s < n ? a.st[s][i] : 0.0f;
            part[s] = fmaf(v[e][s], v[e][s], part[s]);
#pragma unroll
            for (int q = 0; q < MQ; ++q) part[MS + q * MS + s] = fmaf(qv[q], v[e][s], part[MS + q * MS + s]);
        }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < NV; ++t) {
        const float r = warp_sum(part[t]);
        if (l == 0) red[w][t] = r;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        float inv[MS], sc[MS], mx = -INFINITY;
        for (int s = 0; s < n; ++s) {
            float ss = 0.0f, dd = 0.0f;
            for (int ww = 0; ww < NWARP; ++ww) { ss += red[ww][s]; dd += red[ww][MS + q * MS + s]; }
            inv[s] = rsqrtf(ss / D + 1e-6f);
            sc[s] = dd * inv[s] * rsqrtf((float)D);
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < n; ++s) { sc[s] = __expf(sc[s] - mx); sum += sc[s]; }
        for (int s = 0; s < MS; ++s) prob[q][s] = s < n ? sc[s] / sum * inv[s] : 0.0f;
    }
    __syncthreads();
    float o[EPT][MQ], osq[MQ] = {0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = threadIdx.x + e * 768;
#pragma unroll
        for (int q = 0; q < MQ; ++q) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < MS; ++s) acc = fmaf(prob[q][s], v[e][s], acc);
            o[e][q] = acc;
            if (q < nq) {
                a.out[q * D + i] = acc;
                osq[q] = fmaf(acc, acc, osq[q]);
            }
        }
    }
    if (!a.outn) return;
    __syncthreads();
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
        const float r = warp_sum(osq[q]);
        if (l == 0) red[w][q] = r;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        float ss = 0.0f;
        for (int ww = 0; ww < NWARP; ++ww) ss += red[ww][threadIdx.x];
        onorm[threadIdx.x] = rsqrtf(ss / D + a.eps);
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
        if (q >= nq || !a.nw[q]) continue;
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            const int i = threadIdx.x + e * 768;  // warp covers 32 consecutive i: one q8_1 block
            const float val = o[e][q] * onorm[q] * a.nw[q][i];
            a.outn[q * D + i] = val;
            if (A.xq) warp_quant32(val, A.xq + q * D + (i & ~31), A.xd + q * (D / 32) + (i >> 5));
        }
    }
}

// Router with 2 warps per expert row (split K), 4 rows per block -> E/4 blocks; last block does the top-k.
INK_EXTERN void ink_router3(const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias, float* __restrict__ logits,
                            int* __restrict__ sel, float* __restrict__ wts, unsigned int* __restrict__ counter, int D, int E, int k, float wscale) {
    __shared__ float halfsum[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 4 + (warp >> 1), half = warp & 1;
    {
        const float4* row = reinterpret_cast<const float4*>(W + (size_t)e * D) + half * (D / 8);
        const float4* x4 = reinterpret_cast<const float4*>(x) + half * (D / 8);
        float acc = 0.0f;
        for (int i = lane; i < D / 8; i += 32) {
            const float4 a = row[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        acc = warp_sum(acc);
        if (lane == 0) halfsum[warp] = acc;
    }
    __syncthreads();
    if (threadIdx.x < 4) logits[blockIdx.x * 4 + threadIdx.x] = halfsum[2 * threadIdx.x] + halfsum[2 * threadIdx.x + 1];
    __threadfence();
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last || warp != 0) return;
    __threadfence();
    float p[4], s[4];
    for (int t = 0; t < 4; ++t) {
        const int ee = lane + 32 * t;
        const float lg = ee < E ? *((volatile float*)logits + ee) : 0.0f;
        p[t] = ee < E ? 1.0f / (1.0f + __expf(-lg)) : 0.0f;
        s[t] = ee < E ? p[t] + (bias ? bias[ee] : 0.0f) : -INFINITY;
    }
    float sum = 0.0f, mine = 0.0f;
    for (int rr = 0; rr < k; ++rr) {
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
        if (lane == rr) mine = pb;
        if (lane == 0) sel[rr] = bi;
    }
    if (lane < k) wts[lane] = mine / (sum + 1e-20f) * wscale;
    if (lane == 0) *counter = 0;
}

// Loop control with F16 weights (same math as ink_ctrl_down / ink_ctrl_up).
struct Ctrl4H {
    const float* c[4];
    const float* m[4];
    const __half* wd[4];  // [R][2D]
    const __half* wg[4];  // [D][R]
    const __half* wf[4];
    float* g[4];
    float* out[4];
    int n;
};

INK_EXTERN void ink_ctrl_down_h(Ctrl4H a, int D, int R) {
    const int j = blockIdx.y;
    const float* c = a.c[j];
    const float* m = a.m[j];
    float sc = 0.0f, sm = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { sc += c[i] * c[i]; sm += m[i] * m[i]; }
    const float ic = rsqrtf(block_sum2(sc) / D + 1e-6f);
    const float im = rsqrtf(block_sum2(sm) / D + 1e-6f);
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= R) return;
    const uint4* row = reinterpret_cast<const uint4*>(a.wd[j] + (size_t)r * 2 * D);  // 8 halves per uint4
    float acc = 0.0f;
    for (int i = lane; i < 2 * D / 8; i += 32) {
        const uint4 u = __ldg(row + i);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
        const int base = i * 8;
        const bool second = base >= D;
        const float* src = second ? m + (base - D) : c + base;
        const float sclf = second ? im : ic;
        float s = 0.0f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 w2 = __half22float2(h[t]);
            s += w2.x * src[2 * t] + w2.y * src[2 * t + 1];
        }
        acc = fmaf(s, sclf, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) a.g[j][r] = acc;
}

INK_EXTERN void ink_ctrl_up_h(Ctrl4H a, int D, int R) {
    const int j = blockIdx.y;
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= D) return;
    const size_t to = (size_t)blockIdx.z * D;  // batched: token = grid.z
    const float* g = a.g[j] + (size_t)blockIdx.z * R;
    const __half2 wg = reinterpret_cast<const __half2*>(a.wg[j] + (size_t)i * R)[lane];  // R = 64: 2 per lane
    const __half2 wf = reinterpret_cast<const __half2*>(a.wf[j] + (size_t)i * R)[lane];
    const float2 g2 = make_float2(g[2 * lane], g[2 * lane + 1]);
    const float2 a2 = __half22float2(wg), f2 = __half22float2(wf);
    const float ga = warp_sum(a2.x * g2.x + a2.y * g2.y);
    const float fa = warp_sum(f2.x * g2.x + f2.y * g2.y);
    if (lane == 0) a.out[j][to + i] = (1.0f - tanhf(fa)) * a.c[j][to + i] + tanhf(ga) * a.m[j][to + i];
}

// ---------------------------------------------------------------------------------------------------------------
// Device-side autoregressive loop. step = [token, pos, n_out].

// Greedy: argmax of the logits becomes the next input token; it is appended to hist and pos advances.
INK_EXTERN void ink_argmax_next(const float* __restrict__ x, int n, int* __restrict__ step, int* __restrict__ hist) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float t = x[i];
        if (t > v) { v = t; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x < 32) {
        v = bv[threadIdx.x];
        idx = bi[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (threadIdx.x == 0) {
            hist[step[2]] = idx;
            step[2] += 1;
            step[0] = idx;
            step[1] += 1;
        }
    }
}

// Prefill: the next prompt token becomes the input and pos advances.
INK_EXTERN void ink_advance_prompt(int* __restrict__ step, const int* __restrict__ prompt) {
    if (threadIdx.x == 0) {
        step[1] += 1;
        step[0] = prompt[step[1]];
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
