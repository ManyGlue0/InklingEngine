// Inkling 2 decode kernels, M3 "reference" set: one kernel per spec stage (docs/MODEL.md), F32 math.
// Step parameters (token, pos) live in device memory so the launch sequence is identical for every token.
#include "quant.cuh"

struct Ptrs8 {
    const float* p[8];
};

// Sum over the whole block (blockDim.x multiple of 32, <= 1024). All threads get the result.
__device__ float block_sum(float v) {
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

// out = rmsnorm(dequant(emb[token])) * w
INK_EXTERN void ink_embed(const block_q8_0* __restrict__ emb, const int* __restrict__ step, const float* __restrict__ w,
                          float* __restrict__ out, int D, float eps) {
    const block_q8_0* row = emb + (size_t)step[0] * (D / 32);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = __half2float(row[i / 32].d) * row[i / 32].qs[i % 32];
        out[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = out[i] * inv * w[i];
}

// y = rmsnorm(x) (* w if w != null). One block.
INK_EXTERN void ink_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int D, float eps) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    const float inv = rsqrtf(block_sum(ss) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// y[N] = W[N][K] x[K]; one warp per row, 8 warps per block.
INK_EXTERN void ink_matvec_q8(const block_q8_0* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int K, int N) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= N) return;
    const float v = warp_dot_q8(W + (size_t)row * (K / 32), x, K);
    if ((threadIdx.x & 31) == 0) y[row] = v;
}

INK_EXTERN void ink_matvec_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int K, int N) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= N) return;
    const float* r = W + (size_t)row * K;
    float acc = 0.0f;
    for (int i = threadIdx.x & 31; i < K; i += 32) acc += r[i] * x[i];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) y[row] = acc;
}

INK_EXTERN void ink_matvec_bf16(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int K, int N) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= N) return;
    const __nv_bfloat16* r = W + (size_t)row * K;
    float acc = 0.0f;
    for (int i = threadIdx.x & 31; i < K; i += 32) acc += __bfloat162float(r[i]) * x[i];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) y[row] = acc;
}

// Depth attention: states S = ptrs.p[0..n), Ŝ = rms(S); for each of nq query rows: softmax(q·Ŝ/sqrt(D)) weighted Ŝ.
INK_EXTERN void ink_mix(Ptrs8 ptrs, int n, const float* __restrict__ Q, int nq, float* __restrict__ out, int D) {
    __shared__ float inv[8];
    __shared__ float prob[4][8];
    for (int s = 0; s < n; ++s) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) ss += ptrs.p[s][i] * ptrs.p[s][i];
        const float t = block_sum(ss);
        if (threadIdx.x == 0) inv[s] = rsqrtf(t / D + 1e-6f);
    }
    __syncthreads();
    const float scale = rsqrtf((float)D);
    for (int q = 0; q < nq; ++q) {
        for (int s = 0; s < n; ++s) {
            float d = 0.0f;
            for (int i = threadIdx.x; i < D; i += blockDim.x) d += Q[q * D + i] * ptrs.p[s][i];
            const float t = block_sum(d);
            if (threadIdx.x == 0) prob[q][s] = t * inv[s] * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        float m = -INFINITY;
        for (int s = 0; s < n; ++s) m = fmaxf(m, prob[q][s]);
        float l = 0.0f;
        for (int s = 0; s < n; ++s) { prob[q][s] = expf(prob[q][s] - m); l += prob[q][s]; }
        for (int s = 0; s < n; ++s) prob[q][s] /= l;
    }
    __syncthreads();
    for (int q = 0; q < nq; ++q)
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float acc = 0.0f;
            for (int s = 0; s < n; ++s) acc += prob[q][s] * inv[s] * ptrs.p[s][i];
            out[q * D + i] = acc;
        }
}

// Loop control: g = Wd [rms(c); rms(m)]; out = (1 - tanh(Wf g)) c + tanh(Wg g) m.  Wd: [R][2D], Wg/Wf: [D][R].
INK_EXTERN void ink_control(const float* __restrict__ c, const float* __restrict__ m, const float* __restrict__ Wd,
                            const float* __restrict__ Wg, const float* __restrict__ Wf, float* __restrict__ out, int D, int R) {
    __shared__ float g[128];
    float sc = 0.0f, sm = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { sc += c[i] * c[i]; sm += m[i] * m[i]; }
    const float ic = rsqrtf(block_sum(sc) / D + 1e-6f);
    const float im = rsqrtf(block_sum(sm) / D + 1e-6f);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int r = warp; r < R; r += nw) {
        const float* row = Wd + (size_t)r * 2 * D;
        float acc = 0.0f;
        for (int i = lane; i < D; i += 32) acc += row[i] * c[i] * ic + row[D + i] * m[i] * im;
        acc = warp_sum(acc);
        if (lane == 0) g[r] = acc;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float a = 0.0f, f = 0.0f;
        for (int r = 0; r < R; ++r) { a += Wg[(size_t)i * R + r] * g[r]; f += Wf[(size_t)i * R + r] * g[r]; }
        out[i] = (1.0f - tanhf(f)) * c[i] + tanhf(a) * m[i];
    }
}

// Causal short conv (4 taps) with per-slot state [3][ch] (oldest first) and optional per-head dynamic taps.
// out = (x + y) / sqrt(m*4 + 1); state shifts in x.
INK_EXTERN void ink_conv(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ state,
                         const float* __restrict__ dyn, float* __restrict__ out, int ch, int head_ch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float s0 = state[c], s1 = state[ch + c], s2 = state[2 * ch + c], xv = x[c];
    const float* wc = w + (size_t)c * 4;
    float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + xv * wc[3];
    float scale = rsqrtf(5.0f);
    if (dyn) {
        const float* t = dyn + (c / head_ch) * 4;
        y += xv * t[0] + s2 * t[1] + s1 * t[2] + s0 * t[3];
        scale = rsqrtf(9.0f);
    }
    out[c] = (xv + y) * scale;
    state[c] = s1;
    state[ch + c] = s2;
    state[2 * ch + c] = xv;
}

// Post-projection attention prep, one warp per task: 24 q heads, 8 k heads, 8 v heads, 24 r heads.
// q,r: clip (with per-channel scale) and *tau on full layers; k: clip, v: rms; k,v written to the f16 cache.
INK_EXTERN void ink_attn_prep(float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ r,
                              const float* __restrict__ qclip, const float* __restrict__ rclip, const int* __restrict__ step,
                              __half* __restrict__ Kc, __half* __restrict__ Vc, int kv_cap, int is_full,
                              float qk_clip, float rel_clip, float log_floor, float log_alpha, float eps) {
    const int task = blockIdx.x, lane = threadIdx.x;
    const int pos = step[1];
    const float tau = is_full ? 1.0f + log_alpha * logf(fmaxf(1.0f, (float)(pos + 1) / log_floor)) : 1.0f;
    const int row = pos % kv_cap;
    if (task < 24) {  // q head
        float* h = q + task * 128;
        float y[4], ss = 0.0f;
        for (int j = 0; j < 4; ++j) { y[j] = h[lane * 4 + j] * qclip[task * 128 + lane * 4 + j]; ss += y[j] * y[j]; }
        const float f = fminf(1.0f, qk_clip / (sqrtf(warp_sum(ss)) + 1e-6f)) * tau;
        for (int j = 0; j < 4; ++j) h[lane * 4 + j] = y[j] * f;
    } else if (task < 32) {  // k head -> cache
        const int hk = task - 24;
        const float* h = k + hk * 128;
        float y[4], ss = 0.0f;
        for (int j = 0; j < 4; ++j) { y[j] = h[lane * 4 + j]; ss += y[j] * y[j]; }
        const float f = fminf(1.0f, qk_clip / (sqrtf(warp_sum(ss)) + 1e-6f));
        for (int j = 0; j < 4; ++j) Kc[(size_t)row * 1024 + hk * 128 + lane * 4 + j] = __float2half(y[j] * f);
    } else if (task < 40) {  // v head -> cache
        const int hv = task - 32;
        const float* h = v + hv * 128;
        float y[4], ss = 0.0f;
        for (int j = 0; j < 4; ++j) { y[j] = h[lane * 4 + j]; ss += y[j] * y[j]; }
        const float f = rsqrtf(warp_sum(ss) / 128.0f + eps);
        for (int j = 0; j < 4; ++j) Vc[(size_t)row * 1024 + hv * 128 + lane * 4 + j] = __float2half(y[j] * f);
    } else {  // r head (16 dims, lanes 0..15)
        const int hr = task - 40;
        float* h = r + hr * 16;
        const float y = lane < 16 ? h[lane] * rclip[hr * 16 + lane] : 0.0f;
        const float f = fminf(1.0f, rel_clip / (sqrtf(warp_sum(y * y)) + 1e-6f)) * tau;
        if (lane < 16) h[lane] = y * f;
    }
}

// Decode attention for one head per block (4 warps): logits = q·k/sqrt(128) + r·relbias[d] (d < 1024), sink logit
// with zero value, online softmax. window = 0 for full attention, else keys with pos - j < window.
INK_EXTERN void ink_attn_decode(const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sink,
                                const __half* __restrict__ Kc, const __half* __restrict__ Vc, const float* __restrict__ relbias,
                                float* __restrict__ out, const int* __restrict__ step, int kv_cap, int window, int extent) {
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
        const __half* kr = Kc + (size_t)row * 1024 + kvh * 128 + lane * 4;
        float d = 0.0f;
        for (int t = 0; t < 4; ++t) d += qv[t] * __half2float(kr[t]);
        d = warp_sum(d) * scale;
        const int dist = pos - j;
        if (dist < extent) {
            float b = lane < 16 ? rs[lane] * relbias[dist * 16 + lane] : 0.0f;
            d += warp_sum(b);
        }
        const float mn = fmaxf(m, d);
        const float corr = expf(m - mn), p = expf(d - mn);
        l = l * corr + p;
        const __half* vr = Vc + (size_t)row * 1024 + kvh * 128 + lane * 4;
        for (int t = 0; t < 4; ++t) acc[t] = acc[t] * corr + p * __half2float(vr[t]);
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    for (int t = 0; t < 4; ++t) wacc[warp][lane * 4 + t] = acc[t];
    __syncthreads();
    if (warp == 0) {
        const float s = sink[h];
        float M = s;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w]);
        float L = expf(s - M);
        float o[4] = {0, 0, 0, 0};
        for (int w = 0; w < 4; ++w) {
            if (wm[w] == -INFINITY) continue;
            const float e = expf(wm[w] - M);
            L += wl[w] * e;
            for (int t = 0; t < 4; ++t) o[t] += wacc[w][lane * 4 + t] * e;
        }
        for (int t = 0; t < 4; ++t) out[h * 128 + lane * 4 + t] = o[t] / L;
    }
}

INK_EXTERN void ink_gate_sigmoid(float* __restrict__ o, const float* __restrict__ gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

INK_EXTERN void ink_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

INK_EXTERN void ink_copy(const float* __restrict__ a, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}

// Router: logits = W x (F32, W [E][D]); p = sigmoid; top-k by p + bias; w = p_sel / (sum + 1e-20) * wscale.
INK_EXTERN void ink_router(const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias,
                           int* __restrict__ sel, float* __restrict__ wts, float* __restrict__ logits_out, int D, int E, int k, float wscale) {
    __shared__ float lg[256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int e = warp; e < E; e += nw) {
        const float* row = W + (size_t)e * D;
        float acc = 0.0f;
        for (int i = lane; i < D; i += 32) acc += row[i] * x[i];
        acc = warp_sum(acc);
        if (lane == 0) lg[e] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float p[256], s[256];
        for (int e = 0; e < E; ++e) {
            if (logits_out) logits_out[e] = lg[e];
            p[e] = 1.0f / (1.0f + expf(-lg[e]));
            s[e] = p[e] + (bias ? bias[e] : 0.0f);
        }
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            int best = 0;
            for (int e = 1; e < E; ++e) if (s[e] > s[best]) best = e;
            sel[t] = best;
            wts[t] = p[best];
            sum += p[best];
            s[best] = -INFINITY;
        }
        for (int t = 0; t < k; ++t) wts[t] = wts[t] / (sum + 1e-20f) * wscale;
    }
}

__device__ __forceinline__ float expert_row_dot(const uint8_t* base, int wtype, size_t row, const float* x, int K) {
    if (wtype == 1) {
        const size_t rb = (size_t)(K / 64) * sizeof(block_nvfp4);
        return warp_dot_nvfp4((const block_nvfp4*)(base + row * rb), x, K);
    }
    const size_t rb = (size_t)(K / 32) * sizeof(block_q8_0);
    return warp_dot_q8((const block_q8_0*)(base + row * rb), x, K);
}

// h[t][i] = situ(s*g_i, s*u_i) for the t-th selected expert; W [E][2F][K] (gate rows, then up rows). wtype 0=q8_0 1=nvfp4
INK_EXTERN void ink_moe_gate_up(const uint8_t* __restrict__ W, int wtype, const float* __restrict__ scale, const float* __restrict__ x,
                                const int* __restrict__ sel, float* __restrict__ h, int K, int F, float beta, float lb) {
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (i >= F) return;
    const int t = blockIdx.y, e = sel[t];
    const float s = scale ? scale[e] : 1.0f;
    const float g = s * expert_row_dot(W, wtype, (size_t)e * 2 * F + i, x, K);
    const float u = s * expert_row_dot(W, wtype, (size_t)e * 2 * F + F + i, x, K);
    if ((threadIdx.x & 31) == 0) {
        const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
        h[t * F + i] = a * (lb * tanhf(u / lb));
    }
}

// y[n] = sum_t w_t * s_e * (W_e[n] · h_t); W [E][D][F]. Experts summed in order inside one warp (deterministic).
INK_EXTERN void ink_moe_down(const uint8_t* __restrict__ W, int wtype, const float* __restrict__ scale, const float* __restrict__ h,
                             const int* __restrict__ sel, const float* __restrict__ wts, float* __restrict__ y, int F, int D, int k) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < k; ++t) {
        const int e = sel[t];
        const float s = scale ? scale[e] : 1.0f;
        acc += wts[t] * s * expert_row_dot(W, wtype, (size_t)e * D + n, h + t * F, F);
    }
    if ((threadIdx.x & 31) == 0) y[n] = acc;
}

INK_EXTERN void ink_softcap(float* __restrict__ x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
