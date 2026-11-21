// Latency-bound decode kernels (included after decode.cu by the umbrella file; uses its structs and helpers).
//
// Small weights (norms, router, loop control) miss L2 every token because ~4 GB of weights stream through it, so a
// loop that waits on one global load per iteration pays the DRAM latency serially. These kernels issue every global
// load up front, and reduce many values at once with warp reduce-scatter instead of one butterfly per value.

// Warp reduce-scatter of N values (N power of two <= 32): returns, in lane l, the warp sum of v[l % N].
template <int N>
__device__ __forceinline__ float warp_rs(float (&v)[N]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int o = N / 2; o >= 1; o >>= 1) {
        const bool up = lane & o;
#pragma unroll
        for (int i = 0; i < o; ++i) {
            const float send = up ? v[i] : v[i + o];
            const float keep = up ? v[i + o] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    float r = v[0];
#pragma unroll
    for (int o = N; o < 32; o <<= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    return r;
}

// Block reduction of N values (N power of two <= 32) over NW warps into tot[0..N) (shared); synchronizes.
template <int N, int NW>
__device__ __forceinline__ void block_rs(float (&v)[N], float (*red)[32], float* tot) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float r = warp_rs<N>(v);
    if (l < N) red[w][l] = r;
    __syncthreads();
    if (threadIdx.x < N) {
        float s = 0.0f;
#pragma unroll
        for (int ww = 0; ww < NW; ++ww) s += red[ww][threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
}

// RMS-normalize up to 4 vectors (one block of 768 threads each, D = 2304) -> y_j and q8_1.
// Batched: grid.y = tokens, every array advances by D (scales by D/32) per token.
INK_EXTERN void __launch_bounds__(768) ink_rmsnorm_q2(NormQ4 a, int D_, float eps) {
    constexpr int D = 2304, NT = 768, EPT = 3;
    __shared__ float red[NT / 32][32];
    __shared__ float tot[1];
    const int j = blockIdx.x;
    const size_t to = (size_t)blockIdx.y * D;
    const float* x = a.x[j] + to;
    const float* wt = a.w[j];
    float xv[EPT], wv[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        xv[e] = x[threadIdx.x + e * NT];
        wv[e] = __ldg(wt + threadIdx.x + e * NT);
    }
    float ss[1] = {0.0f};
#pragma unroll
    for (int e = 0; e < EPT; ++e) ss[0] = fmaf(xv[e], xv[e], ss[0]);
    block_rs<1, NT / 32>(ss, red, tot);
    const float inv = rsqrtf(tot[0] / D + eps);
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = threadIdx.x + e * NT;
        const float v = xv[e] * inv * wv[e];
        if (a.y[j]) a.y[j][to + i] = v;
        if (a.xq[j]) warp_quant32(v, a.xq[j] + to + (i & ~31), a.xd[j] + to / 32 + (i >> 5));
    }
}

// Depth attention for NQ query rows over n <= 6 states, one block of 768 threads (D = 2304, 3 elements/thread).
// Optional rmsnorm(out_q) * nw[q] -> outn (+ q8_1 copy).
template <int NQ>
__device__ __forceinline__ void mix6(const MixArgsQ& A) {
    const MixArgs& a = A.m;
    constexpr int MS = 6, NV = MS + NQ * MS, NVP = NV <= 16 ? 16 : 32, EPT = 3, NT = 768, NW = NT / 32, D = 2304;
    constexpr int NQP = NQ == 1 ? 1 : 4;
    __shared__ float red[NW][32];
    __shared__ float tot[32];
    __shared__ float prob[NQ][MS];
    const int n = a.n;
    const bool want_n = a.outn != nullptr;
    // batched: token t of T (grid.y); states are [T][D], outputs [NQ][T][D]
    const int t = blockIdx.y, T = gridDim.y;
    const size_t so = (size_t)t * D;
    auto oo = [&](int q) { return ((size_t)q * T + t) * D; };
    float v[EPT][MS], nwv[EPT][NQ];
    float part[NVP];
#pragma unroll
    for (int t = 0; t < NVP; ++t) part[t] = 0.0f;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = threadIdx.x + e * NT;
        float qv[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            qv[q] = __ldg(a.q + q * D + i);
            nwv[e][q] = want_n && a.nw[q] ? __ldg(a.nw[q] + i) : 0.0f;
        }
#pragma unroll
        for (int s = 0; s < MS; ++s) v[e][s] = s < n ? a.st[s][so + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < MS; ++s) {
            part[s] = fmaf(v[e][s], v[e][s], part[s]);
#pragma unroll
            for (int q = 0; q < NQ; ++q) part[MS + q * MS + s] = fmaf(qv[q], v[e][s], part[MS + q * MS + s]);
        }
    }
    block_rs<NVP, NW>(part, red, tot);
    if (threadIdx.x < NQ) {
        const int q = threadIdx.x;
        float inv[MS], sc[MS], mx = -INFINITY;
#pragma unroll
        for (int s = 0; s < MS; ++s) {
            inv[s] = rsqrtf(tot[s] / D + 1e-6f);
            sc[s] = s < n ? tot[MS + q * MS + s] * inv[s] * rsqrtf((float)D) : -INFINITY;
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < MS; ++s) { sc[s] = s < n ? __expf(sc[s] - mx) : 0.0f; sum += sc[s]; }
#pragma unroll
        for (int s = 0; s < MS; ++s) prob[q][s] = sc[s] / sum * inv[s];
    }
    __syncthreads();
    float o[EPT][NQ], osq[NQP];
#pragma unroll
    for (int q = 0; q < NQP; ++q) osq[q] = 0.0f;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = threadIdx.x + e * NT;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < MS; ++s) acc = fmaf(prob[q][s], v[e][s], acc);
            o[e][q] = acc;
            a.out[oo(q) + i] = acc;
            osq[q] = fmaf(acc, acc, osq[q]);
        }
    }
    if (!want_n) return;
    block_rs<NQP, NW>(osq, red, tot);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        if (!a.nw[q]) continue;
        const float on = rsqrtf(tot[q] / D + a.eps);
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            const int i = threadIdx.x + e * NT;  // a warp covers 32 consecutive i: one q8_1 block
            const float val = o[e][q] * on * nwv[e][q];
            a.outn[oo(q) + i] = val;
            if (A.xq) warp_quant32(val, A.xq + oo(q) + (i & ~31), A.xd + oo(q) / 32 + (i >> 5));
        }
    }
}
INK_EXTERN void __launch_bounds__(768) ink_mix6_1(MixArgsQ A) { mix6<1>(A); }
INK_EXTERN void __launch_bounds__(768) ink_mix6_3(MixArgsQ A) { mix6<3>(A); }
INK_EXTERN void __launch_bounds__(768) ink_mix6_4(MixArgsQ A) { mix6<4>(A); }

// Router: one block (6 warps) per expert row, all loads in flight; the last block does the top-k.
INK_EXTERN void __launch_bounds__(192) ink_router4(const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias,
                                                   float* __restrict__ logits, int* __restrict__ sel, float* __restrict__ wts,
                                                   unsigned int* __restrict__ counter, int D_, int E, int k, float wscale) {
    constexpr int D = 2304, NT = 192, PT = D / 4 / NT;  // 3 float4 per thread
    __shared__ float red[NT / 32][32];
    __shared__ float tot[1];
    __shared__ bool last;
    const int e = blockIdx.x;
    const float4* row = reinterpret_cast<const float4*>(W + (size_t)e * D);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4 wv[PT], xv[PT];
#pragma unroll
    for (int t = 0; t < PT; ++t) {
        wv[t] = __ldg(row + threadIdx.x + t * NT);
        xv[t] = x4[threadIdx.x + t * NT];
    }
    float acc[1] = {0.0f};
#pragma unroll
    for (int t = 0; t < PT; ++t) acc[0] += wv[t].x * xv[t].x + wv[t].y * xv[t].y + wv[t].z * xv[t].z + wv[t].w * xv[t].w;
    block_rs<1, NT / 32>(acc, red, tot);
    if (threadIdx.x == 0) {
        logits[e] = tot[0];
        __threadfence();
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    const int lane = threadIdx.x;
    float p[4], s[4];
#pragma unroll
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
#pragma unroll
        for (int t = 1; t < 4; ++t)
            if (s[t] > bv) { bv = s[t]; bi = lane + 32 * t; }
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

// Loop control phase 1 (F16 weights): g_j[r] = Wd_j[r] . [rms(c_j); rms(m_j)]. Grid (R, n), 192 threads: the block
// covers the whole 2D-long row once, so the same loads give the norms of c and m.
INK_EXTERN void __launch_bounds__(192) ink_ctrl_down2(Ctrl4H a, int D_, int R) {
    constexpr int D = 2304, NT = 192, PT = 2 * D / 8 / NT;  // 3 uint4 (8 halves) per thread
    __shared__ float red[NT / 32][32];
    __shared__ float tot[4];
    const int j = blockIdx.y, r = blockIdx.x;
    const size_t to = (size_t)blockIdx.z * D;  // batched: token = grid.z (c, m advance by D, g by R)
    const uint4* row = reinterpret_cast<const uint4*>(a.wd[j] + (size_t)r * 2 * D);
    uint4 u[PT];
    float4 x0[PT], x1[PT];
#pragma unroll
    for (int t = 0; t < PT; ++t) {
        const int ui = threadIdx.x + t * NT, base = ui * 8;
        u[t] = __ldg(row + ui);
        const float* src = base >= D ? a.m[j] + to + (base - D) : a.c[j] + to + base;
        x0[t] = *reinterpret_cast<const float4*>(src);
        x1[t] = *reinterpret_cast<const float4*>(src + 4);
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // dot_c, dot_m, ss_c, ss_m
#pragma unroll
    for (int t = 0; t < PT; ++t) {
        const bool second = (threadIdx.x + t * NT) * 8 >= D;
        const __half2* h = reinterpret_cast<const __half2*>(&u[t]);
        const float xs[8] = {x0[t].x, x0[t].y, x0[t].z, x0[t].w, x1[t].x, x1[t].y, x1[t].z, x1[t].w};
        float d = 0.0f, q = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 w2 = __half22float2(h[i]);
            d = fmaf(w2.x, xs[2 * i], fmaf(w2.y, xs[2 * i + 1], d));
            q = fmaf(xs[2 * i], xs[2 * i], fmaf(xs[2 * i + 1], xs[2 * i + 1], q));
        }
        acc[0] += second ? 0.0f : d;
        acc[1] += second ? d : 0.0f;
        acc[2] += second ? 0.0f : q;
        acc[3] += second ? q : 0.0f;
    }
    block_rs<4, NT / 32>(acc, red, tot);
    if (threadIdx.x == 0) a.g[j][(size_t)blockIdx.z * R + r] = tot[0] * rsqrtf(tot[2] / D + 1e-6f) + tot[1] * rsqrtf(tot[3] / D + 1e-6f);
}
