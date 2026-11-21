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
