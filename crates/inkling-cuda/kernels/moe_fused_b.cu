// MoE for a chain of T <= 8 consecutive tokens (speculative verify). Included after moe_fused.cu (uses ex8).
//
// Every expert of the union of the T routings is read once and applied to all tokens routed to it. Persistent
// (grid = all resident blocks), two grid barriers (hq / hd are unused):
//   A: gate_up. Task = (union expert u, feature pair), rows as in ink_moe8; each loaded chunk is dotted with the
//      activations of every user of u -> h[a][F] (float; a = 8 t + k is the assignment).
//   B: down. Block task = (union expert u, 64 channels): the block stages h of the users of u (quantized to int8 on
//      the way); lane group g of warp w takes channel n0 + 8 w + g (4 lanes per row) -> raw dot products ya[a][n].
//   C: thread per channel: y_t = sum_k w[a] scale[e] ya[a][n], causal short conv over the chain (state = the 3
//      inputs before token 0) + residual into x[t][n]. With yin != null the inputs y_t go to yin[t][n] and the
//      state is left as it was (committed later for the accepted prefix); else the state advances past all T.

namespace ex8b {
constexpr int EMAX = 128;
}

// TMAX = longest chain of this instantiation (4 or 8): per-user accumulators live in registers
template <int TMAX>
__device__ __forceinline__ void moe8b_body(ExpW wg, ExpW wd, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                               const int* __restrict__ sel, const float* __restrict__ wts, int T,
                                                               float* __restrict__ h, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                               float* __restrict__ ya, const float* __restrict__ cw, float* __restrict__ state,
                                                               float* __restrict__ x, float* __restrict__ yin, unsigned int* __restrict__ bar,
                                                               float beta, float lb, unsigned long long* __restrict__ tl,
                                                               const int* __restrict__ rt, int cs) {
    using namespace ex8;
    using namespace ex8b;
    constexpr int AMAX = 8 * TMAX;
    // optional per-block phase timeline (globaltimer ns): start, A done, Q start, Q done, B start, B done, C start, end
    auto mark = [&](int k) {
        if (tl && threadIdx.x == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            tl[blockIdx.x * 8 + k] = t;
        }
    };
    mark(0);
    __shared__ __align__(16) int8_t xs[TMAX][D];
    __shared__ float xds[TMAX][D / 32];
    __shared__ __align__(16) int8_t hs[2][TMAX][F];
    __shared__ __align__(8) float hds[2][TMAX][F / 16];
    __shared__ int emap[EMAX];
    __shared__ int ue[AMAX], un[AMAX];
    __shared__ unsigned char uu[AMAX][TMAX];
    __shared__ float gsc[AMAX];
    __shared__ int nU_s;
    __shared__ int sa[AMAX];
    const int NA = 8 * T;
    for (int i = threadIdx.x; i < EMAX; i += blockDim.x) emap[i] = -1;
    if (threadIdx.x < NA) sa[threadIdx.x] = sel[threadIdx.x];
    __syncthreads();
    // union of the routed experts in first-use order (every block builds the same list, from shared memory)
    if (threadIdx.x == 0) {
        int U = 0;
        for (int a = 0; a < NA; ++a) {
            const int e = sa[a];
            int u = emap[e];
            if (u < 0) {
                u = U++;
                emap[e] = u;
                ue[u] = e;
                un[u] = 0;
            }
            uu[u][un[u]++] = (unsigned char)a;
        }
        nU_s = U;
    }
    __syncthreads();
    const int nU = nU_s;
    if (tl && blockIdx.x == 0 && threadIdx.x == 0) {
        // union statistics after the timeline: [sum of nU, launches, sum of fp4-layer nU, fp4 launches]
        atomicAdd(tl + gridDim.x * 8, (unsigned long long)nU);
        atomicAdd(tl + gridDim.x * 8 + 1, 1ull);
        if (wg.fmt == 1) {
            atomicAdd(tl + gridDim.x * 8 + 2, (unsigned long long)nU);
            atomicAdd(tl + gridDim.x * 8 + 3, 1ull);
        }
    }
    if (threadIdx.x < nU) gsc[threadIdx.x] = wg.scale ? wg.scale[ue[threadIdx.x]] : 1.0f;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NWt = gridDim.x * 8, gw = blockIdx.x * 8 + warp;
    Chunk A, B;

    // ---------------- phase A: gate_up
    {
        const bool fp4 = wg.fmt == 1;
        const int cp = fp4 ? 1 : 2;
        const int NT = nU * (F / 2);
        const int nch = (gw < NT ? (NT - 1 - gw) / NWt + 1 : 0) * cp;
        const int g = lane >> 3, l8 = lane & 7, swz = (l8 >> 2) & 1;
        auto load = [&](Chunk& c, int j) {
            const int task = gw + (j / cp) * NWt, half = j % cp;
            const int u = task / (F / 2), f = 2 * (task % (F / 2)) + (g >> 1);
            const size_t row = (size_t)ue[u] * 2 * F + (g & 1) * F + f;
            if (fp4) {
                const uint4* qr = reinterpret_cast<const uint4*>(wg.qs + row * (D / 2));
                const unsigned short* sr = reinterpret_cast<const unsigned short*>(wg.sc + row * (D / 16));
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    c.q[i] = __ldg(qr + l8 + 8 * i);
                    c.s[i] = __ldg(sr + l8 + 8 * i);
                }
            } else {
                const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(wg.qs) + row * D) + 72 * half;
                const unsigned short* dr = reinterpret_cast<const unsigned short*>(wg.sc) + row * (D / 32);
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    c.q[i] = __ldg(qr + l8 + 8 * i);
                    c.s[i] = __ldg(dr + ((72 * half + l8 + 8 * i) >> 1));
                }
            }
        };
        float acc[TMAX];
#pragma unroll
        for (int k = 0; k < TMAX; ++k) acc[k] = 0.0f;
        auto compute = [&](const Chunk& c, int j) {
            const int half = j % cp;
            const int task = gw + (j / cp) * NWt, u = task / (F / 2);
            const int nu = un[u];
            if (fp4) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int v = l8 + 8 * i;
                    const int2 w0 = e2m1x2_gather(c.q[i].x), w1 = e2m1x2_gather(c.q[i].y);
                    const int2 w2 = e2m1x2_gather(c.q[i].z), w3 = e2m1x2_gather(c.q[i].w);
                    const float2 sc = ue4m3x2_normal(c.s[i]);
#pragma unroll
                    for (int k = 0; k < TMAX; ++k) {
                        if (k < nu) {
                            const int t = uu[u][k] >> 3;
                            const int4* xp = reinterpret_cast<const int4*>(xs[t] + 32 * v);
                            const int4 xa = xp[swz], xb = xp[swz ^ 1];
                            int s0 = __dp4a(w0.y, xa.y, __dp4a(w0.x, xa.x, 0));
                            s0 = __dp4a(w1.y, xa.w, __dp4a(w1.x, xa.z, s0));
                            int s1 = __dp4a(w2.y, xb.y, __dp4a(w2.x, xb.x, 0));
                            s1 = __dp4a(w3.y, xb.w, __dp4a(w3.x, xb.z, s1));
                            acc[k] = fmaf(fmaf((float)s0, sc.x, (float)s1 * sc.y), xds[t][v], acc[k]);
                        }
                    }
                }
            } else {
#pragma unroll
            for (int k = 0; k < TMAX; ++k) {
                if (k < nu) {
                    const int t = uu[u][k] >> 3;
                    {
#pragma unroll
                        for (int i = 0; i < 9; ++i) {
                            const int cc = 72 * half + l8 + 8 * i;
                            const int4 xv = *reinterpret_cast<const int4*>(xs[t] + 16 * cc);
                            acc[k] = fmaf((float)q8_isum(c.q[i], xv), __half2float(__ushort_as_half((unsigned short)c.s[i])) * xds[t][cc >> 1], acc[k]);
                        }
                    }
                }
            }
            }
            if (half == cp - 1) {
                const float sc = gsc[u];
                const int fo = 2 * (task % (F / 2)) + (lane >> 4);
#pragma unroll
                for (int k = 0; k < TMAX; ++k) {
                    if (k < nu) {
                        float s = acc[k];
                        s += __shfl_xor_sync(0xffffffffu, s, 4);
                        s += __shfl_xor_sync(0xffffffffu, s, 2);
                        s += __shfl_xor_sync(0xffffffffu, s, 1);
                        const float up = __shfl_sync(0xffffffffu, s, (lane & 16) | 8) * sc;
                        if ((lane & 15) == 0) {
                            const float gt = s * sc;
                            const float a = beta * tanhf(gt / beta) * (1.0f / (1.0f + __expf(-gt)));
                            h[(size_t)uu[u][k] * F + fo] = a * (lb * tanhf(up / lb));
                        }
                    }
                    acc[k] = 0.0f;
                }
            }
        };
        if (nch > 0) load(A, 0);
        for (int t = 0; t < T; ++t) {
            if (fp4) stage_x_fp4(xs[t], xq + (size_t)t * D, D);
            else stage16(xs[t], xq + (size_t)t * D, D, false);
        }
        for (int i = threadIdx.x; i < T * (D / 32); i += blockDim.x) xds[i / (D / 32)][i % (D / 32)] = xd[i];
        __syncthreads();
        for (int j = 0; j < nch; j += 2) {
            if (j + 1 < nch) load(B, j + 1);
            compute(A, j);
            if (j + 2 < nch) load(A, j + 2);
            if (j + 1 < nch) compute(B, j + 1);
        }
    }

    // ---------------- phase B: down. Block task = (union expert u, 64 channels); lane group g of warp w takes channel
    // n0 + 8 w + g (4 lanes per row). The block stages h (float, global) of the users of u, quantized to int8 in
    // groups of 16 on the way into shared memory (FP4 quad permutation when the down weights are FP4).
    // Double-buffered: the next task's h (and, for FP4, its weights) load while the current task's dot products run;
    // the first task's weights are requested before the grid barrier.
    const bool dfp4 = wd.fmt == 1;
    {
        const int g = lane >> 2, l4 = lane & 3;
        constexpr int CT = D / 64;
        const int NBT = nU * CT;
        auto load = [&](Chunk& c0, Chunk& c1, int bt) {
            const int u = bt / CT, n = (bt % CT) * 64 + 8 * warp + g;
            const size_t row = (size_t)ue[u] * D + n;
            if (dfp4) {
                const uint4* qr = reinterpret_cast<const uint4*>(wd.qs + row * (F / 2));
                const unsigned short* sr = reinterpret_cast<const unsigned short*>(wd.sc + row * (F / 16));
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    c0.q[i] = __ldg(qr + l4 + 4 * i);
                    c0.s[i] = __ldg(sr + l4 + 4 * i);
                }
            } else {
                const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(wd.qs) + row * F);
                const unsigned short* dr = reinterpret_cast<const unsigned short*>(wd.sc) + row * (F / 32);
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    c0.q[i] = __ldg(qr + l4 + 4 * i);
                    c0.s[i] = __ldg(dr + ((l4 + 4 * i) >> 1));
                    c1.q[i] = __ldg(qr + 36 + l4 + 4 * i);
                    c1.s[i] = __ldg(dr + ((36 + l4 + 4 * i) >> 1));
                }
            }
        };
        auto stage = [&](int bt, int buf) {
            const int u = bt / CT, nu = un[u];
            for (int i = threadIdx.x; i < nu * (F / 16); i += blockDim.x) {
                const int k = i / (F / 16), q = i - k * (F / 16);
                const float4* src = reinterpret_cast<const float4*>(h + (size_t)uu[u][k] * F) + 4 * q;
                float v[16];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float4 f4 = __ldcg(src + r);
                    v[4 * r] = f4.x;
                    v[4 * r + 1] = f4.y;
                    v[4 * r + 2] = f4.z;
                    v[4 * r + 3] = f4.w;
                }
                float amax = 0.0f;
#pragma unroll
                for (int r = 0; r < 16; ++r) amax = fmaxf(amax, fabsf(v[r]));
                const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
                int w4[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    w4[r] = 0;
#pragma unroll
                    for (int b = 0; b < 4; ++b) w4[r] |= (__float2int_rn(v[4 * r + b] * id) & 0xff) << (8 * b);
                }
                reinterpret_cast<int4*>(hs[buf][k])[q] = dfp4 ? make_int4(w4[0], w4[2], w4[1], w4[3]) : make_int4(w4[0], w4[1], w4[2], w4[3]);
                hds[buf][k][q] = amax / 127.0f;
            }
        };
        auto compute = [&](const Chunk& c0, const Chunk& c1, int bt, int buf) {
            const int u = bt / CT, n = (bt % CT) * 64 + 8 * warp + g;
            const int nu = un[u];
            float accb[TMAX];
#pragma unroll
            for (int k = 0; k < TMAX; ++k) accb[k] = 0.0f;
            if (dfp4) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int v = l4 + 4 * i;
                    const int2 w0 = e2m1x2_gather(c0.q[i].x), w1 = e2m1x2_gather(c0.q[i].y);
                    const int2 w2 = e2m1x2_gather(c0.q[i].z), w3 = e2m1x2_gather(c0.q[i].w);
                    const float2 sc = ue4m3x2_normal(c0.s[i]);
#pragma unroll
                    for (int k = 0; k < TMAX; ++k) {
                        if (k < nu) {
                            const int4* xp = reinterpret_cast<const int4*>(hs[buf][k] + 32 * v);
                            const int4 xa = xp[0], xb = xp[1];
                            int s0 = __dp4a(w0.y, xa.y, __dp4a(w0.x, xa.x, 0));
                            s0 = __dp4a(w1.y, xa.w, __dp4a(w1.x, xa.z, s0));
                            int s1 = __dp4a(w2.y, xb.y, __dp4a(w2.x, xb.x, 0));
                            s1 = __dp4a(w3.y, xb.w, __dp4a(w3.x, xb.z, s1));
                            const float2 hd2 = *reinterpret_cast<const float2*>(&hds[buf][k][2 * v]);
                            accb[k] = fmaf((float)s0, sc.x * hd2.x, fmaf((float)s1, sc.y * hd2.y, accb[k]));
                        }
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < TMAX; ++k) {
                if (k < nu) {
                    float acc = accb[k];
                    if (!dfp4) {
#pragma unroll
                        for (int i = 0; i < 9; ++i) {
                            const int ca = l4 + 4 * i, cb = 36 + l4 + 4 * i;
                            const int4 x0 = *reinterpret_cast<const int4*>(hs[buf][k] + 16 * ca);
                            const int4 x1 = *reinterpret_cast<const int4*>(hs[buf][k] + 16 * cb);
                            acc = fmaf((float)q8_isum(c0.q[i], x0), __half2float(__ushort_as_half((unsigned short)c0.s[i])) * hds[buf][k][ca], acc);
                            acc = fmaf((float)q8_isum(c1.q[i], x1), __half2float(__ushort_as_half((unsigned short)c1.s[i])) * hds[buf][k][cb], acc);
                        }
                    }
                    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
                    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
                    if (l4 == 0) ya[(size_t)uu[u][k] * D + n] = acc;
                }
            }
        };
        // one task: FP4 computes with cur while the next task's weights load into nxt; Q8 (two chunks per row)
        // computes with A, B and reloads them afterwards. Then the next task's h is staged into the other buffer.
        auto task = [&](Chunk& cur, Chunk& nxt, int bt, int buf) {
            const int nbt = bt + gridDim.x;
            if (dfp4) {
                if (nbt < NBT) load(nxt, nxt, nbt);
                compute(cur, cur, bt, buf);
            } else {
                compute(A, B, bt, buf);
                if (nbt < NBT) load(A, B, nbt);
            }
            if (nbt < NBT) stage(nbt, buf ^ 1);
            __syncthreads();
        };
        int bt = blockIdx.x;
        if (bt < NBT) load(A, B, bt);
        mark(1);
        grid_wait(bar, grid_arrive(bar));
        mark(2);
        if (bt < NBT) stage(bt, 0);
        __syncthreads();
        mark(3);
        mark(4);
        while (bt < NBT) {
            task(A, B, bt, 0);
            bt += gridDim.x;
            if (bt >= NBT) break;
            task(B, A, bt, 1);
            bt += gridDim.x;
        }
    }
    mark(5);
    grid_wait(bar, grid_arrive(bar));
    mark(6);

    // ---------------- phase C: expert sum, causal short conv over the chain, residual
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < D; n += gridDim.x * blockDim.x) {
        float y[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            y[t] = 0.0f;
            if (t < T) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int a = 8 * t + k;
                    y[t] = fmaf(wts[a] * (wd.scale ? wd.scale[sel[a]] : 1.0f), __ldcg(ya + (size_t)a * D + n), y[t]);
                }
            }
        }
        const float w0 = cw[(size_t)n * 4], w1 = cw[(size_t)n * 4 + 1], w2 = cw[(size_t)n * 4 + 2], w3 = cw[(size_t)n * 4 + 3];
        if (rt) {
            // parallel streams: every row is its own sequence with its own state
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < T) {
                    float* st = state + (size_t)rt[2 * t] * cs;
                    const float a0 = st[n], a1 = st[D + n], a2 = st[2 * D + n];
                    x[(size_t)t * D + n] += (y[t] + a0 * w0 + a1 * w1 + a2 * w2 + y[t] * w3) * rsqrtf(5.0f);
                    st[n] = a1;
                    st[D + n] = a2;
                    st[2 * D + n] = y[t];
                }
            }
            continue;
        }
        float s0 = state[n], s1 = state[D + n], s2 = state[2 * D + n];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < T) {
                const float o = (y[t] + s0 * w0 + s1 * w1 + s2 * w2 + y[t] * w3) * rsqrtf(5.0f);
                x[(size_t)t * D + n] += o;
                if (yin) yin[(size_t)t * D + n] = y[t];
                s0 = s1;
                s1 = s2;
                s2 = y[t];
            }
        }
        if (!yin) {
            state[n] = s0;
            state[D + n] = s1;
            state[2 * D + n] = s2;
        }
    }
    __syncthreads();
    mark(7);
}

extern "C" __global__ void __launch_bounds__(256, 2) ink_moe8b4(ExpW wg, ExpW wd, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                               const int* __restrict__ sel, const float* __restrict__ wts, int T,
                                                               float* __restrict__ h, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                               float* __restrict__ ya, const float* __restrict__ cw, float* __restrict__ state,
                                                               float* __restrict__ x, float* __restrict__ yin, unsigned int* __restrict__ bar,
                                                               float beta, float lb, unsigned long long* __restrict__ tl,
                                                               const int* __restrict__ rt, int cs) {
    moe8b_body<4>(wg, wd, xq, xd, sel, wts, T, h, hq, hd, ya, cw, state, x, yin, bar, beta, lb, tl, rt, cs);
}

extern "C" __global__ void __launch_bounds__(256, 2) ink_moe8b8(ExpW wg, ExpW wd, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                               const int* __restrict__ sel, const float* __restrict__ wts, int T,
                                                               float* __restrict__ h, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                               float* __restrict__ ya, const float* __restrict__ cw, float* __restrict__ state,
                                                               float* __restrict__ x, float* __restrict__ yin, unsigned int* __restrict__ bar,
                                                               float beta, float lb, unsigned long long* __restrict__ tl,
                                                               const int* __restrict__ rt, int cs) {
    moe8b_body<8>(wg, wd, xq, xd, sel, wts, T, h, hq, hd, ya, cw, state, x, yin, bar, beta, lb, tl, rt, cs);
}

