// DSpark drafter (Inkling 2 Mini's block drafter, llama.cpp `dflash` architecture with the DSpark Markov head).
//
// Drafter: 3 dense layers (D = 2304, 24 q heads / 8 kv heads of 128, NeoX RoPE, SiLU FFN), non-causal attention over
// its own KV cache. Context K/V come from the target's features of every committed token (KV injection); the draft
// block [anchor, mask x k] writes its own K/V at positions pos0.. and attends to the context before the anchor plus
// the whole block. A rank-256 recurrent Markov head then walks the k draft slots, adding a vocabulary bias.

#include "common.cuh"
#include <cuda_bf16.h>

namespace ds {
constexpr int D = 2304, HD = 128, NQ = 24, NKV = 8, R = 256, NJ = 5120;

// sum over a 128-thread block (4 warps)
__device__ __forceinline__ float sum128b(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float s = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return s;
}

// NeoX RoPE of one 128-dim head held one value per thread (tid < 128); returns the rotated value of dim tid
__device__ __forceinline__ float rope_neox(float v, int pos, float base, float* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    const int i = tid & 63;
    const float theta = (float)pos * powf(base, -2.0f * (float)i / (float)HD);
    float s, c;
    sincosf(theta, &s, &c);
    const float x1 = sh[i], x2 = sh[i + 64];
    __syncthreads();
    return tid < 64 ? x1 * c - x2 * s : x2 * c + x1 * s;
}
}  // namespace ds

// q / k head norms + RoPE, K/V cache write. Grid (nq + 8 + 8, T), 128 threads; token t sits at position step[1] + t.
//   blocks 0..nq-1: q head (q_norm, rope) in place in q [T][qs]
//   next 8: k head (k_norm, rope) -> Kc[pos]; last 8: v head -> Vc[pos] (caches f16 [cap][1024], ring index pos % cap)
INK_EXTERN void __launch_bounds__(128) ink_ds_qk_rope(float* __restrict__ q, int qs, const float* __restrict__ kv, int kvs, int nq,
                                                      const float* __restrict__ qn, const float* __restrict__ kn, const int* __restrict__ step, float eps,
                                                      float base, __half* __restrict__ Kc, __half* __restrict__ Vc, int cap) {
    using namespace ds;
    __shared__ float red[4], sh[HD];
    const int b = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, pos = step[1] + t;
    if (b < nq) {
        float v = q[(size_t)t * qs + b * HD + tid];
        v = v * rsqrtf(sum128b(v * v, red) / HD + eps) * qn[tid];
        q[(size_t)t * qs + b * HD + tid] = rope_neox(v, pos, base, sh);
    } else if (gridDim.y - 1 - t >= cap) {
        return;  // a later row of this batch lands on the same ring slot
    } else if (b < nq + NKV) {
        const int h = b - nq;
        float v = kv[(size_t)t * kvs + h * HD + tid];
        v = v * rsqrtf(sum128b(v * v, red) / HD + eps) * kn[tid];
        Kc[(size_t)(pos % cap) * (NKV * HD) + h * HD + tid] = __float2half(rope_neox(v, pos, base, sh));
    } else {
        const int h = b - nq - NKV;
        Vc[(size_t)(pos % cap) * (NKV * HD) + h * HD + tid] = __float2half(kv[(size_t)t * kvs + NKV * HD + h * HD + tid]);
    }
}

// Non-causal block attention (pos0 = step[1]): query t (head blockIdx.x) attends to positions [lo, pos0 + T), lo = max(0, pos0 - win)
// (win = 0: no window). Grid (nq, T), 128 threads = 4 warps splitting the keys; lane l holds dims 4l..4l+3.
// Writes att [T][nq * 128] as q8_1 (values + one scale per 32) for the output projection.
INK_EXTERN void __launch_bounds__(128) ink_ds_attn(const float* __restrict__ q, int qs, int nq, const __half* __restrict__ Kc, const __half* __restrict__ Vc,
                                                   int cap, const int* __restrict__ step, int T, int win, int8_t* __restrict__ oq, float* __restrict__ od) {
    using namespace ds;
    const int pos0 = step[1];
    __shared__ float sm[4], sl[4], so[4][HD];
    const int h = blockIdx.x, t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kvh = h / (nq / NKV);
    const float scale = rsqrtf((float)HD);
    const float4 qv = *reinterpret_cast<const float4*>(q + (size_t)t * qs + h * HD + 4 * lane);
    const int lo = win > 0 ? max(0, pos0 - win) : 0, hi = pos0 + T;
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int p = lo + warp; p < hi; p += 4) {
        const size_t off = (size_t)(p % cap) * (NKV * HD) + kvh * HD + 4 * lane;
        const __half2* kp = reinterpret_cast<const __half2*>(Kc + off);
        const float2 k01 = __half22float2(kp[0]), k23 = __half22float2(kp[1]);
        const float s = warp_sum(qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y) * scale;
        const float mn = fmaxf(m, s);
        const float c = __expf(m - mn), e = __expf(s - mn);
        const __half2* vp = reinterpret_cast<const __half2*>(Vc + off);
        const float2 v01 = __half22float2(vp[0]), v23 = __half22float2(vp[1]);
        l = l * c + e;
        a0 = a0 * c + e * v01.x;
        a1 = a1 * c + e * v01.y;
        a2 = a2 * c + e * v23.x;
        a3 = a3 * c + e * v23.y;
        m = mn;
    }
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    so[warp][4 * lane] = a0;
    so[warp][4 * lane + 1] = a1;
    so[warp][4 * lane + 2] = a2;
    so[warp][4 * lane + 3] = a3;
    __syncthreads();
    const int d = threadIdx.x;
    const float M = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
    float L = 0.0f, o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const float c = sm[w] == -INFINITY ? 0.0f : __expf(sm[w] - M);
        L += sl[w] * c;
        o += so[w][d] * c;
    }
    o = L > 0.0f ? o / L : 0.0f;
    // q8_1: one scale per 32 values (= one warp)
    float amax = fabsf(o);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
    const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
    const size_t e = (size_t)t * (nq * HD) + h * HD + d;
    oq[e] = (int8_t)__float2int_rn(o * id);
    if (lane == 0) od[e / 32] = amax / 127.0f;
}

// x += y (n floats)
INK_EXTERN void ink_ds_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// h = silu(gate) * up from gu [T][2F] (gate | up), quantized q8_1 into hq [T][F] / hd [T][F/32].
// Grid (F / 256, T), 256 threads: warp = one group of 32.
INK_EXTERN void ink_ds_swiglu_q8(const float* __restrict__ gu, int F, int8_t* __restrict__ hq, float* __restrict__ hd) {
    const int t = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
    const float g = gu[(size_t)t * 2 * F + f], u = gu[(size_t)t * 2 * F + F + f];
    const float h = g / (1.0f + __expf(-g)) * u;
    float amax = fabsf(h);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
    const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
    hq[(size_t)t * F + f] = (int8_t)__float2int_rn(h * id);
    if ((threadIdx.x & 31) == 0) hd[((size_t)t * F + f) / 32] = amax / 127.0f;
}

// Feature mix + norm: y[t] = rms_norm(sum_i a[i] feat[i][t]) * w, feat [4][T][D] (tap-major, stride T*D);
// optional q8_1 copy (yq / yd). Grid (T), 768 threads (3 values each).
INK_EXTERN void __launch_bounds__(768) ink_ds_mix4(const float* __restrict__ feat, int T, float a0, float a1, float a2, float a3,
                                                    const float* __restrict__ w, float eps, float* __restrict__ y, int8_t* __restrict__ yq,
                                                    float* __restrict__ yd) {
    using namespace ds;
    __shared__ float red[24];
    const int t = blockIdx.x;
    const size_t st = (size_t)T * D;
    float v[3], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const size_t i = (size_t)t * D + threadIdx.x + j * 768;
        v[j] = a0 * feat[i] + a1 * feat[st + i] + a2 * feat[2 * st + i] + a3 * feat[3 * st + i];
        ss += v[j] * v[j];
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 24; ++k) tot += red[k];
    const float inv = rsqrtf(tot / D + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = threadIdx.x + j * 768;
        const float o = v[j] * inv * w[c];
        y[(size_t)t * D + c] = o;
        if (yq) {
            float amax = fabsf(o);
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
            const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
            yq[(size_t)t * D + c] = (int8_t)__float2int_rn(o * id);
            if ((threadIdx.x & 31) == 0) yd[((size_t)t * D + c) / 32] = amax / 127.0f;
        }
    }
}

// Markov head, step 1: joint = W [3R][NJ] . [state | w1[prev] | hidden | f_rnn] + b (first: state = 0). Grid (3R / 8), 256 threads.
INK_EXTERN void __launch_bounds__(256) ink_ds_joint(const float* __restrict__ state, int first, const __nv_bfloat16* __restrict__ w1, const int* __restrict__ prev,
                                                    const float* __restrict__ hidden, const float* __restrict__ frnn,
                                                    const __nv_bfloat16* __restrict__ W, const float* __restrict__ b, float* __restrict__ joint) {
    using namespace ds;
    __shared__ __align__(16) float x[NJ];
    const int tok = *prev;
    for (int i = threadIdx.x; i < NJ; i += blockDim.x) {
        float v;
        if (i < R) v = first ? 0.0f : state[i];
        else if (i < 2 * R) v = __bfloat162float(w1[(size_t)tok * R + (i - R)]);
        else if (i < 2 * R + D) v = hidden[i - 2 * R];
        else v = frnn[i - 2 * R - D];
        x[i] = v;
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)row * NJ);  // 8 bf16 per uint4
    float acc = 0.0f;
    for (int i = lane; i < NJ / 8; i += 32) {
        const uint4 u = __ldg(wr + i);
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(h2[k]);
            acc = fmaf(f.x, x[8 * i + 2 * k], fmaf(f.y, x[8 * i + 2 * k + 1], acc));
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) joint[row] = acc + b[row];
}

// Markov head, step 2: state = cand + gate (state - cand), markov = tanh(c2). 256 threads.
INK_EXTERN void ink_ds_rnn(float* __restrict__ state, int first, const float* __restrict__ joint, float* __restrict__ markov) {
    using namespace ds;
    const int i = threadIdx.x;
    const float g = 1.0f / (1.0f + __expf(-joint[i])), c = tanhf(joint[R + i]);
    state[i] = c + g * ((first ? 0.0f : state[i]) - c);
    markov[i] = tanhf(joint[2 * R + i]);
}

// Markov head, step 3: col[v] = softcap(base[v] + b2[v] + W2[v] . markov), v < V. Grid (ceil(V / 8)), 256 threads.
INK_EXTERN void __launch_bounds__(256) ink_ds_bias(const __nv_bfloat16* __restrict__ W2, const float* __restrict__ b2, const float* __restrict__ markov,
                                                   const float* __restrict__ base, float cap, int V, float* __restrict__ col) {
    using namespace ds;
    __shared__ float mk[R];
    mk[threadIdx.x] = markov[threadIdx.x];
    __syncthreads();
    const int v = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= V) return;
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(W2 + (size_t)v * R) + lane);  // 8 bf16: dims 8 lane ..
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 f = __bfloat1622float2(h2[k]);
        acc = fmaf(f.x, mk[8 * lane + 2 * k], fmaf(f.y, mk[8 * lane + 2 * k + 1], acc));
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        const float z = base[v] + b2[v] + acc;
        col[v] = cap > 0.0f ? cap * tanhf(z / cap) : z;
    }
}

// Commit deferred short-conv states: state <- the last 3 of (state, x[0..n)) for every (state, x, ch) entry.
// Grid (ceil(maxch / 256), entries), 256 threads. x rows are ch apart.
struct ConvCommit {
    float* state[64];
    const float* x[64];
    int ch[64];
};
INK_EXTERN void ink_conv_commit(ConvCommit a, int n) {
    const int e = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, ch = a.ch[e];
    if (c >= ch || n <= 0) return;
    float* st = a.state[e];
    const float* x = a.x[e];
    float v[3];
#pragma unroll
    for (int m = 3; m >= 1; --m) {
        const int j = n - m;  // input index; negative: from the old state
        v[3 - m] = j >= 0 ? x[(size_t)j * ch + c] : st[(3 + j) * ch + c];
    }
    st[c] = v[0];
    st[ch + c] = v[1];
    st[2 * ch + c] = v[2];
}

// Per-row sampler steps of a verify pass: row r = token at position step[1] + r; ink_sample with steps[4r..] writes
// its token to hist[r]. One thread per row.
INK_EXTERN void ink_spec_steps(const int* __restrict__ step, int* __restrict__ steps, int n) {
    const int r = threadIdx.x;
    if (r < n) {
        steps[4 * r] = 0;
        steps[4 * r + 1] = step[1] + r;
        steps[4 * r + 2] = r;
        steps[4 * r + 3] = 0;
    }
}
