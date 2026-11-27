// DFlash2 drafter (Inco AI, block diffusion with dynamic convolutions and a candidate selector), reference:
// llama.cpp src/models/dflash.cpp + dflash2.cpp, common/speculative.cpp (draft-dflash2). Shares the block attention,
// q/k norm + RoPE and SwiGLU kernels of dspark.cu.

#include "common.cuh"
#include <cuda_bf16.h>

namespace df {
constexpr int D = 2304, TOPK = 16, RANK = 256;
}

// Target features [4][stride][D] (tap-major) -> q8_1 rows [T][4 D] (the fc input: taps concatenated).
// Grid (4, T), 768 threads (3 values each; a warp covers 32 consecutive values of each third).
INK_EXTERN void __launch_bounds__(768) ink_df_feat_q8(const float* __restrict__ feat, int stride, int8_t* __restrict__ xq, float* __restrict__ xd) {
    using namespace df;
    const int j = blockIdx.x, t = blockIdx.y;
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int c = threadIdx.x + s * 768;
        const float v = feat[((size_t)j * stride + t) * D + c];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
        const size_t e = (size_t)t * 4 * D + j * D + c;
        xq[e] = (int8_t)__float2int_rn(v * id);
        if ((threadIdx.x & 31) == 0) xd[e / 32] = amax / 127.0f;
    }
}

// Dynamic two-tap causal convolution over the block (token axis), per channel c of group c / 16:
//   out[t] (+)= w0[t] x[t] + w1[t] x[t-1],  w_tap[t] = dyn[t][side][tap][c / 16] + base[side][tap][c]
// (dyn rows of 576 = 2 sides x 2 taps x 144 groups; x[-1] = 0). Optional q8_1 copy of the output (oq / od).
// Grid (T), 768 threads (3 channels each).
INK_EXTERN void __launch_bounds__(768) ink_df_conv(const float* __restrict__ x, const float* __restrict__ dyn, const float* __restrict__ base, int side,
                                                   float* __restrict__ out, int accumulate, int8_t* __restrict__ oq, float* __restrict__ od) {
    using namespace df;
    constexpr int G = D / 16;
    const int t = blockIdx.x;
    const float* dr = dyn + (size_t)t * (4 * G) + side * 2 * G;
    const float* br = base + (size_t)side * 2 * D;
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int c = threadIdx.x + s * 768, g = c / 16;
        float v = (dr[g] + br[c]) * x[(size_t)t * D + c];
        if (t > 0) v = fmaf(dr[G + g] + br[D + c], x[(size_t)(t - 1) * D + c], v);
        float* o = out + (size_t)t * D + c;
        if (accumulate) v += *o;
        *o = v;
        if (oq) {
            float amax = fabsf(v);
#pragma unroll
            for (int k = 16; k > 0; k >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, k));
            const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
            oq[(size_t)t * D + c] = (int8_t)__float2int_rn(v * id);
            if ((threadIdx.x & 31) == 0) od[((size_t)t * D + c) / 32] = amax / 127.0f;
        }
    }
}

// Top-16 of each logits row (first V entries), descending (ties: lower id first), values soft-capped (cap > 0).
// Grid (rows), 1024 threads: thread-local sorted lists, merged per warp, then across warps.
INK_EXTERN void __launch_bounds__(1024) ink_df_topk16(const float* __restrict__ logits, int ld, int V, float cap, int* __restrict__ cand, float* __restrict__ val) {
    using namespace df;
    __shared__ float wv[32][TOPK];
    __shared__ int wi[32][TOPK];
    const float* row = logits + (size_t)blockIdx.x * ld;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float v[TOPK];
    int ix[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        v[k] = -INFINITY;
        ix[k] = 0x7fffffff;
    }
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float x = row[i];
        if (x > v[TOPK - 1]) {
            // insert, keeping the list sorted (i increases, so equal values stay in id order)
            float cv = x;
            int ci = i;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                if (cv > v[k]) {
                    const float tv = v[k];
                    const int ti = ix[k];
                    v[k] = cv;
                    ix[k] = ci;
                    cv = tv;
                    ci = ti;
                }
            }
        }
    }
    // merge the 32 lists of a warp: 16 rounds of warp argmax over the list heads, the winner pops its head
    auto merge = [&](float* outv, int* outi) {
#pragma unroll 1
        for (int r = 0; r < TOPK; ++r) {
            float bv = v[0];
            int bi = ix[0], bl = lane;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                const int ol = __shfl_xor_sync(0xffffffffu, bl, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                    bl = ol;
                }
            }
            if (lane == 0) {
                outv[r] = bv;
                outi[r] = bi;
            }
            if (lane == bl) {
#pragma unroll
                for (int k = 0; k < TOPK - 1; ++k) {
                    v[k] = v[k + 1];
                    ix[k] = ix[k + 1];
                }
                v[TOPK - 1] = -INFINITY;
                ix[TOPK - 1] = 0x7fffffff;
            }
        }
    };
    merge(wv[warp], wi[warp]);
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            v[k] = wv[lane][k];
            ix[k] = wi[lane][k];
        }
        __shared__ float fv[TOPK];
        __shared__ int fi[TOPK];
        merge(fv, fi);
        __syncwarp();
        if (lane < TOPK) {
            cand[blockIdx.x * TOPK + lane] = fi[lane];
            const float x = fv[lane];
            val[blockIdx.x * TOPK + lane] = cap > 0.0f ? cap * tanhf(x / cap) : x;
        }
    }
}

// Selector gate: gate[t][r] = W[r] . h[t] (W bf16 [RANK][D]). Grid (RANK / 8, T), 256 threads (warp per output).
INK_EXTERN void __launch_bounds__(256) ink_df_gate(const __nv_bfloat16* __restrict__ W, const float* __restrict__ h, float* __restrict__ gate) {
    using namespace df;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)r * D);
    const float* hr = h + (size_t)t * D;
    float acc = 0.0f;
    for (int i = lane; i < D / 8; i += 32) {
        const uint4 u = __ldg(wr + i);
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(p[k]);
            acc = fmaf(f.x, hr[8 * i + 2 * k], fmaf(f.y, hr[8 * i + 2 * k + 1], acc));
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) gate[(size_t)t * RANK + r] = acc;
}

// Greedy walk of the selector lattice: for draft slot p = 1..k, score candidate j of slot p as
//   succ[cand[p][j]] . (pred[tok[p-1]] * gate[p]) + val[p][j]
// and take the best (first on ties) as tok[p]; tok[0] is the anchor. One block of 512 threads (warp j: candidate j).
// cand / val / gate are indexed by block slot (row p).
INK_EXTERN void __launch_bounds__(512) ink_df_select(const int* __restrict__ cand, const float* __restrict__ val, const float* __restrict__ gate,
                                                     const __nv_bfloat16* __restrict__ pred, const __nv_bfloat16* __restrict__ succ, int* __restrict__ tok, int k) {
    using namespace df;
    __shared__ float cond[RANK];
    __shared__ float sc[TOPK];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int p = 1; p <= k; ++p) {
        const int prev = tok[p - 1];
        if (threadIdx.x < RANK) cond[threadIdx.x] = __bfloat162float(pred[(size_t)prev * RANK + threadIdx.x]) * gate[(size_t)p * RANK + threadIdx.x];
        __syncthreads();
        const int c = cand[p * TOPK + warp];
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(succ + (size_t)c * RANK) + lane);  // dims 8 lane ..
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
        float acc = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(h2[q]);
            acc = fmaf(f.x, cond[8 * lane + 2 * q], fmaf(f.y, cond[8 * lane + 2 * q + 1], acc));
        }
        acc = warp_sum(acc);
        if (lane == 0) sc[warp] = acc + val[p * TOPK + warp];
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int j = 1; j < TOPK; ++j)
                if (sc[j] > sc[best]) best = j;
            tok[p] = cand[p * TOPK + best];
        }
        __syncthreads();
    }
}
