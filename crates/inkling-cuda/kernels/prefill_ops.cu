// Prefill (batched) elementwise / small kernels and the grouped MoE GEMM. Included after prefill.cu.
// Layouts: per-token activations are [T][width]; short-conv states are [3][ch] holding x(-3), x(-2), x(-1).

// x[t] = rmsnorm(token_embd[tokens[t]]) * w   (Q8_0 AoS embedding), grid T
INK_EXTERN void ink_embed_b(const block_q8_0* __restrict__ emb, const int* __restrict__ tokens, const float* __restrict__ w,
                            float* __restrict__ out, int D, float eps) {
    const block_q8_0* row = emb + (size_t)tokens[blockIdx.x] * (D / 32);
    out += (size_t)blockIdx.x * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = __half2float(row[i / 32].d) * row[i / 32].qs[i % 32];
        out[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = out[i] * inv * w[i];
}

// BF16 multi-segment matvec per token (grid.y = token): x advances by K, y by ldy.
INK_EXTERN void ink_mv_bf16b(MVB4 a, int ldy) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    int seg = 0;
    while (seg < a.nseg && row >= a.s[seg].n) { row -= a.s[seg].n; ++seg; }
    if (seg >= a.nseg) return;
    const MVSegB& s = a.s[seg];
    const float* x = s.x + (size_t)blockIdx.y * a.K;
    float acc = 0.0f;
    if (s.f32w) {
        const float2* r = reinterpret_cast<const float2*>(reinterpret_cast<const float*>(s.w) + (size_t)row * a.K);
        for (int i = threadIdx.x & 31; i < a.K / 2; i += 32) {
            const float2 w = r[i];
            acc += w.x * x[2 * i] + w.y * x[2 * i + 1];
        }
    } else {
        const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(s.w + (size_t)row * a.K);
        for (int i = threadIdx.x & 31; i < a.K / 2; i += 32) {
            const float2 w = __bfloat1622float2(r[i]);
            acc += w.x * x[2 * i] + w.y * x[2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) s.y[(size_t)blockIdx.y * ldy + row] = acc;
}

// input x(t - m) of a causal short conv along tokens: from the chunk, or from the state for t - m < 0
__device__ __forceinline__ float conv_in(const float* __restrict__ x, const float* __restrict__ st, int t, int m, int c, int ch) {
    const int j = t - m;
    return j >= 0 ? x[(size_t)j * ch + c] : st[(3 + j) * ch + c];
}

// Attention prep for a chunk, grid (64, T), 128 threads (as ink_attn_prep2 per token, position step[1] + t):
//   0..23 q heads (clip, tau), 24..31 k heads (causal conv [+dyn] -> clip -> cache), 32..39 v heads (conv -> rms ->
//   cache), 40..63 r heads (clip, tau). The conv states are read only (ink_conv_state_b advances them afterwards).
// Parallel streams (rt != null): row t is one token of stream s = rt[2t] at position rt[2t + 1]; its conv states sit
// s * cs floats and its caches s * ks / s * vs bytes past the given pointers, and the conv reads only the state.
INK_EXTERN void ink_attn_prep_b(float* __restrict__ qrg, int qs, int roff, const float* __restrict__ kraw, const float* __restrict__ vraw,
                                const float* __restrict__ qclip, const float* __restrict__ rclip, const float* __restrict__ kconv,
                                const float* __restrict__ vconv, const float* __restrict__ kstate, const float* __restrict__ vstate,
                                const float* __restrict__ dyn, int dyns, int has_dyn, const int* __restrict__ step,
                                void* __restrict__ Kc, void* __restrict__ Vc, int kv_cap, int is_full, float qk_clip,
                                float rel_clip, float log_floor, float log_alpha, float eps, int kbits, int vbits, int kvfmt,
                                const int* __restrict__ rt, long long ks, long long vs, int cs) {
    const int task = blockIdx.x, tid = threadIdx.x, t = blockIdx.y;
    int pos = step[1] + t, tc = t;
    if (rt) {
        const int sq = rt[2 * t];
        pos = rt[2 * t + 1];
        tc = 0;
        kstate += (size_t)sq * cs;
        vstate += (size_t)sq * cs;
        Kc = (char*)Kc + sq * ks;
        Vc = (char*)Vc + sq * vs;
    }
    const float tau = is_full ? 1.0f + log_alpha * logf(fmaxf(1.0f, (float)(pos + 1) / log_floor)) : 1.0f;
    float* q = qrg + (size_t)t * qs;
    float* r = q + roff;
    if (task < 24) {
        const float y = q[task * 128 + tid] * qclip[task * 128 + tid];
        const float f = fminf(1.0f, qk_clip / (sqrtf(sum128(y * y)) + 1e-6f)) * tau;
        q[task * 128 + tid] = y * f;
    } else if (task < 40) {
        const bool isk = task < 32;
        const int h = isk ? task - 24 : task - 32, c = h * 128 + tid, ch = 1024;
        const float* x = isk ? kraw : vraw;
        const float* st = isk ? kstate : vstate;
        const float* wc = (isk ? kconv : vconv) + (size_t)c * 4;
        const float xv = x[(size_t)t * ch + c];
        const float* xc = x + (size_t)(t - tc) * ch;  // parallel streams: no earlier rows of the same sequence
        const float s2 = conv_in(xc, st, tc, 1, c, ch), s1 = conv_in(xc, st, tc, 2, c, ch), s0 = conv_in(xc, st, tc, 3, c, ch);
        float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + xv * wc[3];
        float scale = rsqrtf(5.0f);
        if (has_dyn) {
            const float* td = dyn + (size_t)t * dyns + (isk ? 0 : 32) + h * 4;
            y += xv * td[0] + s2 * td[1] + s1 * td[2] + s0 * td[3];
            scale = rsqrtf(9.0f);
        }
        const float o = (xv + y) * scale;
        float f;
        if (isk) f = fminf(1.0f, qk_clip / (sqrtf(sum128(o * o)) + 1e-6f));
        else f = rsqrtf(sum128(o * o) / 128.0f + eps);
        kv_store(isk ? Kc : Vc, isk ? kv_kfmt(kvfmt) : kv_vfmt(kvfmt), kv_cap, (size_t)(pos % kv_cap), h, tid, kv_fakequant(o * f, isk ? kbits : vbits));
    } else {
        const int h = task - 40;
        const float y = tid < 16 ? r[h * 16 + tid] * rclip[h * 16 + tid] : 0.0f;
        const float f = fminf(1.0f, rel_clip / (sqrtf(sum128(y * y)) + 1e-6f)) * tau;
        if (tid < 16) r[h * 16 + tid] = y * f;
    }
}

// Causal post conv + residual over a chunk, grid (ch/256, T): o = (x + conv(x)) / sqrt(5); mode 0: resid = o,
// mode 1: resid += o. The state is read only. Parallel streams (rt != null): row t's state is s * cs floats in
// (s = rt[2t]) and the conv reads only the state.
INK_EXTERN void ink_conv_res_b(const float* __restrict__ xin, const float* __restrict__ w, const float* __restrict__ state,
                               float* __restrict__ resid, int mode, int ch, const int* __restrict__ rt, int cs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= ch) return;
    const float xv = xin[(size_t)t * ch + c];
    const int tc = rt ? 0 : t;
    if (rt) state += (size_t)rt[2 * t] * cs;
    const float* xc = xin + (size_t)(t - tc) * ch;
    const float s2 = conv_in(xc, state, tc, 1, c, ch), s1 = conv_in(xc, state, tc, 2, c, ch), s0 = conv_in(xc, state, tc, 3, c, ch);
    const float* wc = w + (size_t)c * 4;
    const float o = (xv + s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + xv * wc[3]) * rsqrtf(5.0f);
    float* rr = resid + (size_t)t * ch + c;
    *rr = mode ? *rr + o : o;
}

// Advance a short-conv state past a chunk of T inputs: state <- x(T-3), x(T-2), x(T-1).
INK_EXTERN void ink_conv_state_b(const float* __restrict__ x, int T, float* __restrict__ state, int ch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float n0 = conv_in(x, state, T, 3, c, ch), n1 = conv_in(x, state, T, 2, c, ch), n2 = conv_in(x, state, T, 1, c, ch);
    state[c] = n0;
    state[ch + c] = n1;
    state[2 * ch + c] = n2;
}

// Parallel streams: advance the conv state of each row's stream by its one input. Grid (ch / 256, T).
INK_EXTERN void ink_conv_state_m(const float* __restrict__ x, const int* __restrict__ rt, float* __restrict__ state, int cs, int ch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= ch) return;
    float* st = state + (size_t)rt[2 * t] * cs;
    const float a = st[ch + c], b = st[2 * ch + c];
    st[c] = a;
    st[ch + c] = b;
    st[2 * ch + c] = x[(size_t)t * ch + c];
}

// Router logits per token: grid (E, T), 192 threads (as ink_router4 without the top-k).
INK_EXTERN void __launch_bounds__(192) ink_router_b(const float* __restrict__ x, const float* __restrict__ W, float* __restrict__ logits, int E) {
    constexpr int D = 2304, NT = 192, PT = D / 4 / NT;
    __shared__ float red[NT / 32][32];
    __shared__ float tot[1];
    const int e = blockIdx.x, t = blockIdx.y;
    const float4* row = reinterpret_cast<const float4*>(W + (size_t)e * D);
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)t * D);
    float4 wv[PT], xv[PT];
#pragma unroll
    for (int i = 0; i < PT; ++i) {
        wv[i] = __ldg(row + threadIdx.x + i * NT);
        xv[i] = x4[threadIdx.x + i * NT];
    }
    float acc[1] = {0.0f};
#pragma unroll
    for (int i = 0; i < PT; ++i) acc[0] += wv[i].x * xv[i].x + wv[i].y * xv[i].y + wv[i].z * xv[i].z + wv[i].w * xv[i].w;
    block_rs<1, NT / 32>(acc, red, tot);
    if (threadIdx.x == 0) logits[(size_t)t * E + e] = tot[0];
}

// Top-k routing per token (grid T, one warp): sigmoid probabilities, bias for selection, weights normalized * wscale.
INK_EXTERN void ink_topk_b(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ sel, float* __restrict__ wts,
                           int E, int k, float wscale) {
    const int lane = threadIdx.x, t = blockIdx.x;
    logits += (size_t)t * E;
    float p[4], s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int ee = lane + 32 * i;
        const float lg = ee < E ? logits[ee] : 0.0f;
        p[i] = ee < E ? 1.0f / (1.0f + __expf(-lg)) : 0.0f;
        s[i] = ee < E ? p[i] + (bias ? bias[ee] : 0.0f) : -INFINITY;
    }
    float sum = 0.0f, mine = 0.0f;
    for (int rr = 0; rr < k; ++rr) {
        float bv = s[0];
        int bi = lane;
#pragma unroll
        for (int i = 1; i < 4; ++i)
            if (s[i] > bv) { bv = s[i]; bi = lane + 32 * i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const float pb = __shfl_sync(0xffffffffu, p[bi >> 5], bi & 31);
        if ((bi & 31) == lane) s[bi >> 5] = -INFINITY;
        sum += pb;
        if (lane == rr) mine = pb;
        if (lane == 0) sel[t * k + rr] = bi;
    }
    if (lane < k) wts[t * k + lane] = mine / (sum + 1e-20f) * wscale;
}

// MoE routing of A = T*8 assignments (a = 8t + slot) by expert, one block of 1024 threads:
//   alist[p] = a sorted by expert, inv[a] = p, tiles = (expert, first p, count <= 64), ntiles[0] = #tiles.
INK_EXTERN void ink_moe_route(const int* __restrict__ sel, int A, int* __restrict__ alist, int* __restrict__ inv,
                              int4* __restrict__ tiles, int* __restrict__ ntiles) {
    __shared__ int cnt[128], off[128], cur[128];
    for (int e = threadIdx.x; e < 128; e += blockDim.x) { cnt[e] = 0; cur[e] = 0; }
    __syncthreads();
    for (int a = threadIdx.x; a < A; a += blockDim.x) atomicAdd(&cnt[sel[a]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int e = 0; e < 128; ++e) {
            off[e] = o;
            for (int s = 0; s < cnt[e]; s += 64) tiles[nt++] = make_int4(e, o + s, min(64, cnt[e] - s), 0);
            o += cnt[e];
        }
        ntiles[0] = nt;
    }
    __syncthreads();
    for (int a = threadIdx.x; a < A; a += blockDim.x) {
        const int e = sel[a];
        const int p = off[e] + atomicAdd(&cur[e], 1);
        alist[p] = a;
        inv[a] = p;
    }
}

// Grouped expert GEMM on int8 tensor cores. Block = (n-tile, routing tile): 64 assignment rows of one expert x 64
// weight rows, K staged 128 at a time (same structure as ink_gemm_q8).
//   GATEUP: K = D, rows = 32 gate features f0.. and the 32 matching up features; X rows are tokens alist[p] / 8 of
//           xq [T][D]. Epilogue: h = SiTU(gate) * linear(up) (per-expert scale applied), quantized per 32 features
//           -> hq [A][F], hd [A][F/32] at the sorted position p.
//   DOWN:   K = F, rows n0.. of the expert's down matrix; X rows are hq[p]. Epilogue: y [A][D] at position p.
// FP4 weights are decoded to int8 (E2M1 x 2) in shared memory and multiplied per 16-value sub-block (m16n8k16) so
// each UE4M3 scale applies exactly; Q8 weights use m16n8k32 with their per-32 scale. Natively stored formats
// (fmt >= 2, codecs.cuh) are decoded per group of 32 straight into the int8 tile, with two scales per group and a
// group offset m (applied as -m * sum(x) with per-token activation sums).
#include "codecs.cuh"
namespace moeg {
constexpr int D = 2304, F = 1152, BT = 64, BN = 64, KC = 128, RS = 144;

__device__ __forceinline__ void mma_s8_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    // Turing: two int8 m8n8k16 (rows g and g+8)
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+r"(c[2]), "+r"(c[3]) : "r"(a1), "r"(b0));
#endif
}
}  // namespace moeg

template <bool GATEUP>
__device__ __forceinline__ void moe_gemm(const ExpW& W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                         const int* __restrict__ alist, const int4* __restrict__ tiles, const int* __restrict__ ntiles,
                                         int8_t* __restrict__ hq, float* __restrict__ hd, float* __restrict__ y, float beta, float lb) {
    using namespace moeg;
    constexpr int K = GATEUP ? D : F;
    __shared__ __align__(16) int8_t sm[2 * BT * RS];  // xs | ws, reused as the f32 output tile by GATEUP
    __shared__ float xds[BT][KC / 32];
    __shared__ float wsc[BN][KC / 16];  // weight scale per 16 (FP4, generic) or per 32 (Q8, first 4 used)
    __shared__ float wmin[BN][KC / 32]; // generic formats: group offset m
    __shared__ int xsum[BT][KC / 32];   // generic formats: sum of the int8 activations per group
    __shared__ int rowsrc[BT];
    int8_t* xs = sm;
    int8_t* ws = sm + BT * RS;
    if ((int)blockIdx.y >= ntiles[0]) return;
    const int4 tl = tiles[blockIdx.y];
    const int e = tl.x, p0 = tl.y, cnt = tl.z;
    if (W.erange && (e < (W.erange & 0xffff) || e >= (W.erange >> 16))) return;  // expert group streamed in
    const bool fp4 = W.fmt == 1, gen = W.fmt >= 2;
    const size_t grb = gen ? cdc::row_bytes(W.fmt, K) : 0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    if (tid < BT) rowsrc[tid] = tid < cnt ? (GATEUP ? alist[p0 + tid] >> 3 : p0 + tid) : -1;
    __syncthreads();
    // weight row of tile row r
    auto wrow = [&](int r) -> size_t {
        if (GATEUP) {
            const int f0 = blockIdx.x * 32;
            return (size_t)e * 2 * F + (r < 32 ? f0 + r : F + f0 + r - 32);
        }
        return (size_t)e * D + blockIdx.x * BN + r;
    };

    uint4 rx[4], rw[4];
    float rxd[2];
    uint32_t rws[2];
    int k0_commit = 0;  // K offset of the tile commit() stores (generic formats decode it there)
    auto fetch = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
            const int src = rowsrc[r];
            rx[i] = src >= 0 ? *reinterpret_cast<const uint4*>(xq + (size_t)src * K + k0 + 16 * c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
            const int src = rowsrc[r];
            rxd[i] = src >= 0 ? xd[(size_t)src * (K / 32) + k0 / 32 + c] : 0.0f;
        }
        if (gen) {
            // generic formats are decoded from global memory in commit()
        } else if (fp4) {
            // 64 rows x 64 qs bytes = 256 uint4 (2 per thread); 64 rows x 8 scale bytes (one uint2 per thread < 64)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
                rw[i] = __ldg(reinterpret_cast<const uint4*>(W.qs + wrow(r) * (K / 2) + k0 / 2) + c);
            }
            if (tid < BN) {
                const uint2 s = __ldg(reinterpret_cast<const uint2*>(W.sc + wrow(tid) * (K / 16) + k0 / 16));
                rws[0] = s.x;
                rws[1] = s.y;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
                rw[i] = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(W.qs) + wrow(r) * K + k0) + c);
            }
            if (tid < BN) {
                const uint2 s = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(W.sc) + wrow(tid) * (K / 32) + k0 / 32));
                rws[0] = s.x;
                rws[1] = s.y;
            }
        }
    };
    auto commit = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
            *reinterpret_cast<uint4*>(xs + r * RS + 16 * c) = rx[i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
            xds[r][c] = rxd[i];
        }
        if (gen) {
            const int kg = (int)(k0_commit / 32);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
                const cdc::G32 q = cdc::g32(W.fmt, W.qs + wrow(r) * grb, kg + c);
                int4* dst = reinterpret_cast<int4*>(ws + r * RS + 32 * c);
                dst[0] = make_int4(q.q[0], q.q[1], q.q[2], q.q[3]);
                dst[1] = make_int4(q.q[4], q.q[5], q.q[6], q.q[7]);
                wsc[r][2 * c] = q.s0;
                wsc[r][2 * c + 1] = q.s1;
                wmin[r][c] = q.m;
            }
            // activation sums per (token row, group): lanes c and c ^ 1 hold the two 16-byte halves of a group
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
                int sum = __dp4a((int)rx[i].x, 0x01010101, 0);
                sum = __dp4a((int)rx[i].y, 0x01010101, sum);
                sum = __dp4a((int)rx[i].z, 0x01010101, sum);
                sum = __dp4a((int)rx[i].w, 0x01010101, sum);
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                if ((c & 1) == 0) xsum[r][c >> 1] = sum;
            }
        } else if (fp4) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
                // 16 bytes = two 16-value sub-blocks; byte b of a sub-block holds elements b (low) and b + 8 (high)
                const int2 g0 = e2m1x2_gather(rw[i].x), g1 = e2m1x2_gather(rw[i].y);
                const int2 g2 = e2m1x2_gather(rw[i].z), g3 = e2m1x2_gather(rw[i].w);
                int4* dst = reinterpret_cast<int4*>(ws + r * RS + 32 * c);
                dst[0] = make_int4(g0.x, g1.x, g0.y, g1.y);
                dst[1] = make_int4(g2.x, g3.x, g2.y, g3.y);
            }
            if (tid < BN) {
#pragma unroll
                for (int b = 0; b < 8; ++b) wsc[tid][b] = ue4m3_fast((rws[b >> 2] >> (8 * (b & 3))) & 0xff);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
                *reinterpret_cast<uint4*>(ws + r * RS + 16 * c) = rw[i];
            }
            if (tid < BN) {
                const __half2 h01 = *reinterpret_cast<const __half2*>(&rws[0]), h23 = *reinterpret_cast<const __half2*>(&rws[1]);
                wsc[tid][0] = __low2float(h01);
                wsc[tid][1] = __high2float(h01);
                wsc[tid][2] = __low2float(h23);
                wsc[tid][3] = __high2float(h23);
            }
        }
    };

    float acc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.0f;

    fetch(0);
    for (int k0 = 0; k0 < K; k0 += KC) {
        k0_commit = k0;
        __syncthreads();
        commit();
        __syncthreads();
        if (k0 + KC < K) fetch(k0 + KC);
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
            uint32_t af[2][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int8_t* p = xs + (wm * 32 + mt * 16 + g) * RS + kb * 32 + 4 * tig;
                af[mt][0] = *reinterpret_cast<const uint32_t*>(p);
                af[mt][1] = *reinterpret_cast<const uint32_t*>(p + 8 * RS);
                af[mt][2] = *reinterpret_cast<const uint32_t*>(p + 16);
                af[mt][3] = *reinterpret_cast<const uint32_t*>(p + 8 * RS + 16);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int8_t* p = ws + (wn * 32 + nt * 8 + g) * RS + kb * 32 + 4 * tig;
                bf[nt][0] = *reinterpret_cast<const uint32_t*>(p);
                bf[nt][1] = *reinterpret_cast<const uint32_t*>(p + 16);
            }
            float sx[2][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                sx[mt][0] = xds[wm * 32 + mt * 16 + g][kb];
                sx[mt][1] = xds[wm * 32 + mt * 16 + g + 8][kb];
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int n = wn * 32 + nt * 8 + 2 * tig;
                if (gen) {
                    const float w00 = wsc[n][2 * kb], w01 = wsc[n][2 * kb + 1], w10 = wsc[n + 1][2 * kb], w11 = wsc[n + 1][2 * kb + 1];
                    const float m0 = wmin[n][kb], m1 = wmin[n + 1][kb];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                        mma_s8_k16(c0, af[mt][0], af[mt][1], bf[nt][0]);
                        mma_s8_k16(c1, af[mt][2], af[mt][3], bf[nt][1]);
                        const float xs0 = (float)xsum[wm * 32 + mt * 16 + g][kb], xs1 = (float)xsum[wm * 32 + mt * 16 + g + 8][kb];
                        acc[mt][nt][0] = fmaf(sx[mt][0], fmaf((float)c0[0], w00, fmaf((float)c1[0], w01, -m0 * xs0)), acc[mt][nt][0]);
                        acc[mt][nt][1] = fmaf(sx[mt][0], fmaf((float)c0[1], w10, fmaf((float)c1[1], w11, -m1 * xs0)), acc[mt][nt][1]);
                        acc[mt][nt][2] = fmaf(sx[mt][1], fmaf((float)c0[2], w00, fmaf((float)c1[2], w01, -m0 * xs1)), acc[mt][nt][2]);
                        acc[mt][nt][3] = fmaf(sx[mt][1], fmaf((float)c0[3], w10, fmaf((float)c1[3], w11, -m1 * xs1)), acc[mt][nt][3]);
                    }
                } else if (fp4) {
                    const float w00 = wsc[n][2 * kb], w01 = wsc[n][2 * kb + 1], w10 = wsc[n + 1][2 * kb], w11 = wsc[n + 1][2 * kb + 1];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        // A fragment of m16n8k32 = the two k16 halves: regs (0,1) and (2,3)
                        int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                        mma_s8_k16(c0, af[mt][0], af[mt][1], bf[nt][0]);
                        mma_s8_k16(c1, af[mt][2], af[mt][3], bf[nt][1]);
                        acc[mt][nt][0] = fmaf(sx[mt][0], fmaf((float)c0[0], w00, (float)c1[0] * w01), acc[mt][nt][0]);
                        acc[mt][nt][1] = fmaf(sx[mt][0], fmaf((float)c0[1], w10, (float)c1[1] * w11), acc[mt][nt][1]);
                        acc[mt][nt][2] = fmaf(sx[mt][1], fmaf((float)c0[2], w00, (float)c1[2] * w01), acc[mt][nt][2]);
                        acc[mt][nt][3] = fmaf(sx[mt][1], fmaf((float)c0[3], w10, (float)c1[3] * w11), acc[mt][nt][3]);
                    }
                } else {
                    const float w0 = wsc[n][kb], w1 = wsc[n + 1][kb];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int c[4] = {0, 0, 0, 0};
                        gq8::mma_s8(c, af[mt], bf[nt]);
                        acc[mt][nt][0] = fmaf((float)c[0], sx[mt][0] * w0, acc[mt][nt][0]);
                        acc[mt][nt][1] = fmaf((float)c[1], sx[mt][0] * w1, acc[mt][nt][1]);
                        acc[mt][nt][2] = fmaf((float)c[2], sx[mt][1] * w0, acc[mt][nt][2]);
                        acc[mt][nt][3] = fmaf((float)c[3], sx[mt][1] * w1, acc[mt][nt][3]);
                    }
                }
            }
        }
    }

    if (!GATEUP) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * 32 + mt * 16 + g + 8 * h;
                    if (r >= cnt) continue;
                    const int n = blockIdx.x * BN + wn * 32 + nt * 8 + 2 * tig;
                    float* yy = y + (size_t)(p0 + r) * D + n;
                    yy[0] = acc[mt][nt][2 * h];
                    yy[1] = acc[mt][nt][2 * h + 1];
                }
        return;
    }
    // GATEUP epilogue: output tile through shared memory (64 x 64 f32 = 16 KB over xs | ws), then one warp per row
    __syncthreads();
    float* ct = reinterpret_cast<float*>(sm);  // [64][65]
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * 32 + mt * 16 + g + 8 * h, n = wn * 32 + nt * 8 + 2 * tig;
                ct[r * 65 + n] = acc[mt][nt][2 * h];
                ct[r * 65 + n + 1] = acc[mt][nt][2 * h + 1];
            }
    __syncthreads();
    const float sc = W.scale ? W.scale[e] : 1.0f;
    for (int r = warp; r < cnt; r += 4) {
        const float gt = ct[r * 65 + lane] * sc, up = ct[r * 65 + 32 + lane] * sc;
        const float a = beta * tanhf(gt / beta) * (1.0f / (1.0f + __expf(-gt)));
        const float v = a * (lb * tanhf(up / lb));
        warp_quant32(v, hq + (size_t)(p0 + r) * F + blockIdx.x * 32, hd + (size_t)(p0 + r) * (F / 32) + blockIdx.x);
    }
}

extern "C" __global__ void __launch_bounds__(128) ink_moe_gate_up_g(ExpW w, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                                    const int* __restrict__ alist, const int4* __restrict__ tiles,
                                                                    const int* __restrict__ ntiles, int8_t* __restrict__ hq,
                                                                    float* __restrict__ hd, float beta, float lb) {
    moe_gemm<true>(w, xq, xd, alist, tiles, ntiles, hq, hd, nullptr, beta, lb);
}
extern "C" __global__ void __launch_bounds__(128) ink_moe_down_g(ExpW w, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                                                                 const int4* __restrict__ tiles, const int* __restrict__ ntiles,
                                                                 float* __restrict__ y) {
    moe_gemm<false>(w, hq, hd, nullptr, tiles, ntiles, nullptr, nullptr, y, 0.0f, 0.0f);
}

// ysum[t][n] = sum_k wts[t][k] * scale(e_tk) * y[inv[8t + k]][n], grid (D/256, T)
INK_EXTERN void ink_moe_combine(const float* __restrict__ y, const int* __restrict__ inv, const int* __restrict__ sel,
                                const float* __restrict__ wts, const float* __restrict__ escale, float* __restrict__ ysum, int D) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= D) return;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int a = 8 * t + k;
        s = fmaf(wts[a] * (escale ? escale[sel[a]] : 1.0f), y[(size_t)inv[a] * D + n], s);
    }
    ysum[(size_t)t * D + n] = s;
}

// ---------------------------------------------------------------------------------------------------------------
// Small SIMT GEMM for the low-rank / small-N prefill matmuls (loop controls, router):
//   Y[t][n] = (acc ? Y[t][n] : 0) + rs[t] * sum_k X[t][k] W[n][k]     (rs optional, W f32 (wt 0) or f16 (wt 1))
// Tile 64 x 64, K step 16, 256 threads of 4 x 4 outputs. K % (16 * gridDim.z) == 0.
// Split K (gridDim.z = S > 1): raw partials go to part[z][T][N]; ink_sgemm_reduce sums them in a fixed order.
INK_EXTERN void __launch_bounds__(256) ink_sgemm(const float* __restrict__ X, int ldx, const void* __restrict__ Wv, int ldw, int wt,
                                                 float* __restrict__ Y, int ldy, const float* __restrict__ rs, int acc, int T, int N, int K,
                                                 float* __restrict__ part) {
    __shared__ float xs[16][68], ws[16][68];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int t0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float c[4][4] = {};
    // loader: row lr = tid / 4 (0..63), k quad lq = tid % 4
    const int lr = tid >> 2, lq = (tid & 3) * 4;
    const int kspan = K / gridDim.z, kbeg = blockIdx.z * kspan;
    for (int k0 = kbeg; k0 < kbeg + kspan; k0 += 16) {
        float4 xv = make_float4(0, 0, 0, 0), wv = make_float4(0, 0, 0, 0);
        if (t0 + lr < T) xv = *reinterpret_cast<const float4*>(X + (size_t)(t0 + lr) * ldx + k0 + lq);
        if (n0 + lr < N) {
            if (wt == 0) {
                wv = __ldg(reinterpret_cast<const float4*>(reinterpret_cast<const float*>(Wv) + (size_t)(n0 + lr) * ldw + k0 + lq));
            } else {
                const uint2 u = __ldg(reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(Wv) + (size_t)(n0 + lr) * ldw + k0 + lq));
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
                wv = make_float4(a.x, a.y, b.x, b.y);
            }
        }
        __syncthreads();
        xs[lq][lr] = xv.x; xs[lq + 1][lr] = xv.y; xs[lq + 2][lr] = xv.z; xs[lq + 3][lr] = xv.w;
        ws[lq][lr] = wv.x; ws[lq + 1][lr] = wv.y; ws[lq + 2][lr] = wv.z; ws[lq + 3][lr] = wv.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[k][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[k][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) c[i][j] = fmaf(av[i], bv[j], c[i][j]);
        }
    }
    if (gridDim.z > 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t0 + ty * 4 + i;
            if (t >= T) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = n0 + tx * 4 + j;
                if (n < N) part[((size_t)blockIdx.z * T + t) * N + n] = c[i][j];
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + ty * 4 + i;
        if (t >= T) continue;
        const float s = rs ? rs[t] : 1.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            float* y = Y + (size_t)t * ldy + n;
            *y = (acc ? *y : 0.0f) + s * c[i][j];
        }
    }
}

// inv[t] = rsqrt(mean(X[t]^2) + eps), one block per row
INK_EXTERN void ink_rowrms(const float* __restrict__ X, int ld, int n, float* __restrict__ inv, float eps) {
    const float* x = X + (size_t)blockIdx.x * ld;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss = fmaf(x[i], x[i], ss);
    ss = block_sum(ss);
    if (threadIdx.x == 0) inv[blockIdx.x] = rsqrtf(ss / n + eps);
}

// loop control output: out = (1 - tanh(fa)) c + tanh(ga) m, elementwise over n values
INK_EXTERN void ink_ctrl_combine(const float* __restrict__ ga, const float* __restrict__ fa, const float* __restrict__ c,
                                 const float* __restrict__ m, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.0f - tanhf(fa[i])) * c[i] + tanhf(ga[i]) * m[i];
}

// Y[t][n] = (acc ? Y : 0) + rs[t] * sum_z part[z][t][n], grid (ceil(T*N/256))
INK_EXTERN void ink_sgemm_reduce(const float* __restrict__ part, int S, float* __restrict__ Y, int ldy, const float* __restrict__ rs,
                                 int acc, int T, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int t = i / N, n = i - t * N;
    float v = 0.0f;
    for (int z = 0; z < S; ++z) v += part[(size_t)z * T * N + i];
    float* y = Y + (size_t)t * ldy + n;
    *y = (acc ? *y : 0.0f) + (rs ? rs[t] : 1.0f) * v;
}
