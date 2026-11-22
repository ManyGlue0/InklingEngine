// Decode attention on tensor cores: split-K flash-decoding for GQA 24 q heads / 8 kv heads, head_dim 128, any KV
// cache type (layout and formats: kv_planes in decode.cu). Same arguments and grid as ink_attn_split.
//
// Grid (8 kv heads, NS splits [, query tokens]), 128 threads. A split covers a contiguous position range sized at run
// time (the grid is fixed inside the CUDA graph); warp w walks 16-position tiles t0 = s0 + 16w, +64, ... with the
// next tile's raw cache data prefetched into registers. Lane = 4g + t.
//   Scores S^T = Q K^T with the 3 query heads as the A rows (rows 3..15 zero) and 8 positions per MMA (n = g):
//     quantized K: int8 MMA m16n8k32, one MMA per 32-channel quant group, so the group scales of q (int8, per head
//     and group) and of K multiply the integer dot product afterwards. q8 K bytes feed the MMA as stored; q4 / q6
//     values are rebuilt as unsigned bytes (even channels in b0, odd in b1; q uses the same order) and the zero
//     point is removed with the per-group sum of q.
//     f16 K (sliding-window rings): f16 MMA m16n8k16, lane t holds channels 32t..32t+31 (q in the same order).
//   Thread (g < 3, t) then owns head g at positions 2t, 2t+1, 8+2t, 9+2t of the tile: relative-position bias, mask,
//   online softmax in the exp2 domain (q carries 1/sqrt(128) * log2 e).
//   O^T += V^T P^T: A = V^T (16 channels x 16 positions; thread g holds channels 16g..16g+15, rows g / g+8 of
//   m-tile mc are channels 16g + 2mc / + 1), dequantized to f16; B = P^T straight from the score accumulators.
// Warps merge in shared memory, splits through `part` (m, l in the log2 domain) and the last block of each kv head,
// which adds the attention sink, applies the sigmoid output gate and writes the output and its q8_1 copy.

namespace da2 {
constexpr int HD = 128, NQ = 3, TILE = 16;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }

// A rows 8..15 are zero here (a1 = a3 = 0): only c0, c1 are meaningful.
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a2), "r"(a3), "r"(b1));
#endif
}
// int8 m16n8k32 with A rows 8..15 zero: c0, c1 = row g, columns 2t, 2t+1.
__device__ __forceinline__ void mma_s8(int (&c)[2], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    int d2 = 0, d3 = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+r"(c[0]), "+r"(c[1]) : "r"(a2), "r"(b1));
#endif
}

// zero point of the stored unsigned values
template <int F> __host__ __device__ constexpr int kv_off() { return F == 1 ? 32 : F == 2 ? 8 : 128; }

// Raw cache data of one tile for one thread.
template <int KF, int VF>
struct Tile {
    // K: positions t0 + g and t0 + 8 + g. f16: 16 words (channels 32t..32t+31); quantized: per group lo / hi words
    uint32_t k[2][KF == 0 ? 16 : KF == 3 ? 8 : KF == 1 ? 8 : 4];
    uint2 ks[2][2];  // K scales [n-block][position 2t / 2t+1] (4 halves = 4 groups)
    // V: positions 2t, 2t+1, 8+2t, 9+2t; channels 16g..16g+15
    uint32_t v[4][VF == 0 ? 8 : VF == 3 ? 4 : VF == 1 ? 3 : 2];
    __half vs[4];
};

template <int KF, int VF>
__device__ __forceinline__ void load_tile(Tile<KF, VF>& T, const KvPlanes& PK, const KvPlanes& PV, int t0, int kv_cap, int g, int t) {
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
        const size_t row = (size_t)((t0 + 8 * nb + g) % kv_cap);
        if (KF == 0) {
            const uint4* s = reinterpret_cast<const uint4*>(PK.q + row * 256 + 64 * t);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 w = s[i];
                T.k[nb][4 * i] = w.x; T.k[nb][4 * i + 1] = w.y; T.k[nb][4 * i + 2] = w.z; T.k[nb][4 * i + 3] = w.w;
            }
        } else {
#pragma unroll
            for (int G = 0; G < 4; ++G) {
                if (KF == 3) {
                    const uint2 w = *reinterpret_cast<const uint2*>(PK.q + row * 128 + 32 * G + 8 * t);
                    T.k[nb][2 * G] = w.x;
                    T.k[nb][2 * G + 1] = w.y;
                } else if (KF == 1) {
                    T.k[nb][2 * G] = *reinterpret_cast<const uint32_t*>(PK.q + row * 64 + 16 * G + 4 * t);
                    T.k[nb][2 * G + 1] = *reinterpret_cast<const uint32_t*>(PK.hi + row * 32 + 4 * (2 * G + (t >> 1)));
                } else {
                    T.k[nb][G] = *reinterpret_cast<const uint32_t*>(PK.q + row * 64 + 16 * G + 4 * t);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
                T.ks[nb][i] = *reinterpret_cast<const uint2*>(PK.sc + (size_t)((t0 + 8 * nb + 2 * t + i) % kv_cap) * 4);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t row = (size_t)((t0 + 2 * t + (i & 1) + 8 * (i >> 1)) % kv_cap);
        if (VF == 0) {
            const uint4* s = reinterpret_cast<const uint4*>(PV.q + row * 256 + 32 * g);
            const uint4 a = s[0], b = s[1];
            T.v[i][0] = a.x; T.v[i][1] = a.y; T.v[i][2] = a.z; T.v[i][3] = a.w;
            T.v[i][4] = b.x; T.v[i][5] = b.y; T.v[i][6] = b.z; T.v[i][7] = b.w;
        } else if (VF == 3) {
            const uint4 a = *reinterpret_cast<const uint4*>(PV.q + row * 128 + 16 * g);
            T.v[i][0] = a.x; T.v[i][1] = a.y; T.v[i][2] = a.z; T.v[i][3] = a.w;
        } else {
            const uint2 a = *reinterpret_cast<const uint2*>(PV.q + row * 64 + 8 * g);
            T.v[i][0] = a.x;
            T.v[i][1] = a.y;
            if (VF == 1) T.v[i][2] = *reinterpret_cast<const uint32_t*>(PV.hi + row * 32 + 4 * g);
        }
        if (VF != 0) T.vs[i] = PV.sc[row * 4 + (g >> 1)];
    }
}

// q6 / q4: even / odd channel bytes of a lo word (+ high bits) as unsigned values. h is the hi word already shifted
// for this 8-channel half of its 16-channel chunk.
__device__ __forceinline__ void unpack_eo(uint32_t lo, uint32_t h, bool q6, uint32_t& ev, uint32_t& od) {
    ev = lo & 0x0F0F0F0Fu;
    od = (lo >> 4) & 0x0F0F0F0Fu;
    if (q6) {
        ev |= (h << 4) & 0x30303030u;
        od |= (h << 2) & 0x30303030u;
    }
}

// V of one position: 8 half2 words, word k = channels (2k, 2k+1) of the thread's 16.
template <int VF>
__device__ __forceinline__ void v_deq(const uint32_t* v, __half s, uint32_t (&w)[8]) {
    if (VF == 0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = v[k];
        return;
    }
    // channel-order unsigned bytes, 4 words of 4 channels
    uint32_t cb[4];
    if (VF == 3) {
#pragma unroll
        for (int i = 0; i < 4; ++i) cb[i] = v[i] ^ 0x80808080u;
    } else {
#pragma unroll
        for (int hw = 0; hw < 2; ++hw) {  // channels 8hw .. 8hw+7 from lo word hw
            uint32_t ev, od;
            unpack_eo(v[hw], VF == 1 ? v[2] >> (4 * hw) : 0u, VF == 1, ev, od);
            cb[2 * hw] = prmt(ev, od, 0x5140);      // ch 0,1,2,3 of this half
            cb[2 * hw + 1] = prmt(ev, od, 0x7362);  // ch 4,5,6,7
        }
    }
    const __half2 bias = __float2half2_rn(1024.0f + (float)kv_off<VF>());
    const __half2 s2 = __half2half2(s);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[2 * i] = h2u(__hmul2(__hsub2(u2h(prmt(cb[i], 0x64646464u, 0x4140)), bias), s2));
        w[2 * i + 1] = h2u(__hmul2(__hsub2(u2h(prmt(cb[i], 0x64646464u, 0x4342)), bias), s2));
    }
}

template <int KF, int VF>
__device__ __forceinline__ void attn_body(const float* __restrict__ sink, const void* __restrict__ Kc, const void* __restrict__ Vc,
                                          const float* __restrict__ relbias, const float* __restrict__ gate, float* __restrict__ out,
                                          int8_t* __restrict__ oq, float* __restrict__ od, int pos, int kv_cap, int window, int extent,
                                          float* __restrict__ part, unsigned int* __restrict__ counters, const float (*qs)[HD],
                                          const float (*rs)[16], float (*wm)[NQ], float (*wl)[NQ], float (*wo)[NQ][HD], float (*ml)[NQ][2],
                                          bool& last) {
    const int kvh = blockIdx.x, split = blockIdx.y, NS = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int j0 = window ? max(0, pos - window + 1) : 0;
    const int npos = pos - j0 + 1;
    const int span = max(4 * TILE, ((npos + NS - 1) / NS + TILE - 1) / TILE * TILE);
    const int s0 = j0 + split * span, s1 = min(pos + 1, s0 + span);
    const int nsplit = min(NS, (npos + span - 1) / span);
    const KvPlanes PK = kv_planes(Kc, KF, kv_cap, kvh), PV = kv_planes(Vc, VF, kv_cap, kvh);
    const bool hrow = g < NQ;  // this thread's score row is a real head

    // q as MMA A fragments (row g = head g; rows >= 3 zero)
    const float qscale = rsqrtf((float)HD) * LOG2E;
    uint32_t qa[KF == 0 ? 8 : 4][2];
    float qd[4];
    int qsum[4];
    if (KF == 0) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float* qq = hrow ? &qs[g][32 * t + 4 * c] : nullptr;
            qa[c][0] = hrow ? h2u(__floats2half2_rn(qq[0] * qscale, qq[1] * qscale)) : 0u;
            qa[c][1] = hrow ? h2u(__floats2half2_rn(qq[2] * qscale, qq[3] * qscale)) : 0u;
        }
    } else {
#pragma unroll
        for (int G = 0; G < 4; ++G) {
            qa[G][0] = qa[G][1] = 0u;
            qd[G] = 0.0f;
            qsum[G] = 0;
            if (hrow) {
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(qs[g][32 * G + i]));
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                qd[G] = amax / 127.0f * qscale;
                int sum = 0;
                for (int i = 0; i < 32; ++i) sum += (int)rintf(qs[g][32 * G + i] * inv);
                qsum[G] = sum * kv_off<KF>();
                // byte order matching the K bytes: q8 channels 8t..8t+3 | 8t+4..; q4 / q6 even | odd channels
                uint32_t w0 = 0, w1 = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int c0 = KF == 3 ? 8 * t + b : 8 * t + 2 * b, c1 = KF == 3 ? 8 * t + 4 + b : 8 * t + 2 * b + 1;
                    w0 |= (uint32_t)((int)rintf(qs[g][32 * G + c0] * inv) & 0xFF) << (8 * b);
                    w1 |= (uint32_t)((int)rintf(qs[g][32 * G + c1] * inv) & 0xFF) << (8 * b);
                }
                qa[G][0] = w0;
                qa[G][1] = w1;
            }
        }
    }

    float m = -INFINITY, l = 0.0f;  // head g (rows g < 3)
    float acc[8][4];
#pragma unroll
    for (int mc = 0; mc < 8; ++mc) acc[mc][0] = acc[mc][1] = acc[mc][2] = acc[mc][3] = 0.0f;

    int t0 = split < nsplit ? s0 + warp * TILE : s1;
    Tile<KF, VF> cur, nxt;
    if (t0 < s1) load_tile(nxt, PK, PV, t0, kv_cap, g, t);
    for (; t0 < s1; t0 += 4 * TILE) {
        cur = nxt;
        if (t0 + 4 * TILE < s1) load_tile(nxt, PK, PV, t0 + 4 * TILE, kv_cap, g, t);

        // scores of head g at positions t0 + 8nb + 2t + i
        float s[2][2];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            if (KF == 0) {
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int cc = 0; cc < 8; ++cc) mma_f16(c, qa[cc][0], 0u, qa[cc][1], 0u, cur.k[nb][2 * cc], cur.k[nb][2 * cc + 1]);
                s[nb][0] = c[0];
                s[nb][1] = c[1];
            } else {
                float a0 = 0.0f, a1 = 0.0f;
                const __half* k0 = reinterpret_cast<const __half*>(&cur.ks[nb][0]);
                const __half* k1 = reinterpret_cast<const __half*>(&cur.ks[nb][1]);
#pragma unroll
                for (int G = 0; G < 4; ++G) {
                    uint32_t b0, b1;
                    if (KF == 3) {
                        b0 = cur.k[nb][2 * G];
                        b1 = cur.k[nb][2 * G + 1];
                    } else {
                        unpack_eo(KF == 1 ? cur.k[nb][2 * G] : cur.k[nb][G], KF == 1 ? cur.k[nb][2 * G + 1] >> (4 * (t & 1)) : 0u, KF == 1, b0, b1);
                    }
                    int c[2] = {0, 0};
                    mma_s8(c, qa[G][0], qa[G][1], b0, b1);
                    // q8 K is signed (zero point 0): the unsigned zero point applies to q4 / q6 only
                    const int zp = KF == 3 ? 0 : qsum[G];
                    a0 = fmaf((float)(c[0] - zp) * qd[G], __half2float(k0[G]), a0);
                    a1 = fmaf((float)(c[1] - zp) * qd[G], __half2float(k1[G]), a1);
                }
                s[nb][0] = a0;
                s[nb][1] = a1;
            }
        }
        // relative-position bias, mask (rows >= 3: no head, probability 0)
        float tmax = -INFINITY;
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int j = t0 + 8 * nb + 2 * t + i;
                const int dist = pos - j;
                float v = s[nb][i];
                if (hrow && dist < extent && dist >= 0) {
                    const float4* rb = reinterpret_cast<const float4*>(relbias + (size_t)dist * 16);
                    float b = 0.0f;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float4 x = __ldg(rb + k);
                        b = fmaf(rs[g][4 * k], x.x, fmaf(rs[g][4 * k + 1], x.y, fmaf(rs[g][4 * k + 2], x.z, fmaf(rs[g][4 * k + 3], x.w, b))));
                    }
                    v = fmaf(b, LOG2E, v);
                }
                v = (hrow && j < s1) ? v : -INFINITY;
                s[nb][i] = v;
                tmax = fmaxf(tmax, v);
            }
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        float corr = 1.0f, p[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        if (hrow) {  // a tile always holds a valid position, so tmax is finite
            const float mn = fmaxf(m, tmax);
            corr = exp2f(m - mn);
#pragma unroll
            for (int nb = 0; nb < 2; ++nb)
#pragma unroll
                for (int i = 0; i < 2; ++i) p[nb][i] = exp2f(s[nb][i] - mn);
            l = l * corr + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
            m = mn;
        }
        // rescale O^T: columns 2t, 2t+1 are heads whose correction lives in lanes 4 * head
        const float cA = __shfl_sync(0xffffffffu, corr, (8 * t) & 31), cB = __shfl_sync(0xffffffffu, corr, (8 * t + 4) & 31);
        const uint32_t pb0 = h2u(__floats2half2_rn(p[0][0], p[0][1])), pb1 = h2u(__floats2half2_rn(p[1][0], p[1][1]));
        uint32_t wa[8], wb[8], wc[8], wd[8];
        v_deq<VF>(cur.v[0], cur.vs[0], wa);
        v_deq<VF>(cur.v[1], cur.vs[1], wb);
        v_deq<VF>(cur.v[2], cur.vs[2], wc);
        v_deq<VF>(cur.v[3], cur.vs[3], wd);
#pragma unroll
        for (int mc = 0; mc < 8; ++mc) {
            acc[mc][0] *= cA;
            acc[mc][1] *= cB;
            acc[mc][2] *= cA;
            acc[mc][3] *= cB;
            mma_f16(acc[mc], prmt(wa[mc], wb[mc], 0x5410), prmt(wa[mc], wb[mc], 0x7632), prmt(wc[mc], wd[mc], 0x5410),
                    prmt(wc[mc], wd[mc], 0x7632), pb0, pb1);
        }
    }

    // merge the 4 warps (block partial m, l, o per head)
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (hrow && t == 0) {
        wm[warp][g] = m;
        wl[warp][g] = l;
    }
    if (t < 2) {
#pragma unroll
        for (int mc = 0; mc < 8; ++mc) {
            const int c0 = 16 * g + 2 * mc;
            wo[warp][2 * t][c0] = acc[mc][0];
            wo[warp][2 * t][c0 + 1] = acc[mc][2];
            if (t == 0) {
                wo[warp][1][c0] = acc[mc][1];
                wo[warp][1][c0 + 1] = acc[mc][3];
            }
        }
    }
    __syncthreads();
    if (split >= nsplit) return;  // empty split (whole block): nothing to contribute
    const int dim = threadIdx.x;
    float bM[NQ], bL[NQ], bo[NQ];
#pragma unroll
    for (int h = 0; h < NQ; ++h) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w][h]);
        float L = 0.0f, o = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (wm[w][h] == -INFINITY) continue;
            const float e = exp2f(wm[w][h] - M);
            L += wl[w][h] * e;
            o += wo[w][h][dim] * e;
        }
        bM[h] = M;
        bL[h] = L;
        bo[h] = o;
    }
    auto finish = [&](int h, float M, float L, float o) {  // sink, normalization, gate, outputs (thread = dim)
        const int qh = kvh * NQ + h;
        const float sk = sink[qh] * LOG2E;
        const float Mx = fmaxf(M, sk);
        const float e = M == -INFINITY ? 0.0f : exp2f(M - Mx);
        const float Lt = exp2f(sk - Mx) + L * e;
        const float gt = gate[qh * HD + dim];
        const float y = o * e / Lt * (1.0f / (1.0f + __expf(-gt)));
        out[qh * HD + dim] = y;
        warp_quant32(y, oq + qh * HD + warp * 32, od + qh * 4 + warp);
    };
    if (nsplit == 1) {
#pragma unroll
        for (int h = 0; h < NQ; ++h) finish(h, bM[h], bL[h], bo[h]);
        return;
    }
    float* pm = part + ((size_t)kvh * NS + split) * (NQ * (HD + 2));
#pragma unroll
    for (int h = 0; h < NQ; ++h) {
        pm[NQ * 2 + h * HD + dim] = bo[h];
        if (dim == 0) { pm[2 * h] = bM[h]; pm[2 * h + 1] = bL[h]; }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&counters[kvh], 1u) == (unsigned)nsplit - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < nsplit * NQ * 2; i += blockDim.x)
        (&ml[0][0][0])[i] = __ldcg(part + ((size_t)kvh * NS + i / (NQ * 2)) * (NQ * (HD + 2)) + i % (NQ * 2));
    __syncthreads();
#pragma unroll
    for (int h = 0; h < NQ; ++h) {
        float M = -INFINITY;
        for (int sp = 0; sp < nsplit; ++sp) M = fmaxf(M, ml[sp][h][0]);
        float L = 0.0f, o = 0.0f;
#pragma unroll 8
        for (int sp = 0; sp < nsplit; ++sp) {
            const float ms = ml[sp][h][0];
            const float e = ms == -INFINITY ? 0.0f : exp2f(ms - M);
            L += ml[sp][h][1] * e;
            o += __ldcg(part + ((size_t)kvh * NS + sp) * (NQ * (HD + 2)) + NQ * 2 + h * HD + dim) * e;
        }
        finish(h, M, L, o);
    }
    if (threadIdx.x == 0) counters[kvh] = 0;
}
}  // namespace da2

extern "C" __global__ void __launch_bounds__(128) ink_attn_mma(const float* __restrict__ q, const float* __restrict__ r,
                                                               const float* __restrict__ sink, const void* __restrict__ Kc,
                                                               const void* __restrict__ Vc, const float* __restrict__ relbias,
                                                               const float* __restrict__ gate, float* __restrict__ out,
                                                               int8_t* __restrict__ oq, float* __restrict__ od, const int* __restrict__ step,
                                                               int kv_cap, int window, int extent, float* __restrict__ part,
                                                               unsigned int* __restrict__ counters, int qstride, int sstride, int kvfmt) {
    using namespace da2;
    __shared__ __align__(16) float qs[NQ][HD];
    __shared__ float rs[NQ][16];
    __shared__ float wm[4][NQ], wl[4][NQ];
    __shared__ float wo[4][NQ][HD];
    __shared__ float ml[64][NQ][2];  // NS <= 64
    __shared__ bool last;
    // batched prefill: query token t = grid.z at position step[1] + t (its K/V are already in the cache)
    const int tq = blockIdx.z;
    const int pos = step[1] + tq;
    q += (size_t)tq * qstride;
    r += (size_t)tq * qstride;
    gate += (size_t)tq * qstride;
    sink += (size_t)tq * sstride;
    out += (size_t)tq * 24 * HD;
    oq += (size_t)tq * 24 * HD;
    od += (size_t)tq * 24 * (HD / 32);
    counters += (size_t)tq * gridDim.x;
    part += (size_t)tq * gridDim.x * gridDim.y * (NQ * (HD + 2));
    const int kvh = blockIdx.x;
    for (int i = threadIdx.x; i < NQ * HD; i += blockDim.x) qs[i / HD][i % HD] = q[(kvh * NQ) * HD + i];
    if (threadIdx.x < NQ * 16) rs[threadIdx.x / 16][threadIdx.x % 16] = r[(kvh * NQ) * 16 + threadIdx.x];
    __syncthreads();
#define INK_ATTN_CASE(KF, VF)                                                                                                        \
    case KF | (VF << 4):                                                                                                             \
        attn_body<KF, VF>(sink, Kc, Vc, relbias, gate, out, oq, od, pos, kv_cap, window, extent, part, counters, qs, rs, wm, wl, wo, ml, \
                          last);                                                                                                     \
        break;
    switch (kvfmt) {
        INK_ATTN_CASE(0, 0) INK_ATTN_CASE(1, 1) INK_ATTN_CASE(2, 2) INK_ATTN_CASE(3, 3)
        INK_ATTN_CASE(0, 1) INK_ATTN_CASE(0, 2) INK_ATTN_CASE(0, 3)
        INK_ATTN_CASE(1, 0) INK_ATTN_CASE(1, 2) INK_ATTN_CASE(1, 3)
        INK_ATTN_CASE(2, 0) INK_ATTN_CASE(2, 1) INK_ATTN_CASE(2, 3)
        INK_ATTN_CASE(3, 0) INK_ATTN_CASE(3, 1) INK_ATTN_CASE(3, 2)
        default: break;
    }
#undef INK_ATTN_CASE
}

// Benchmark helper: fill rows [0, n) of a K or V cache (8 kv heads) with pseudo-random values of rms `amp`.
extern "C" __global__ void ink_kv_fill(void* cache, int fmt, int kv_cap, int n, unsigned int seed, float amp) {
    const int row = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
    if (row >= n) return;
    unsigned int x = seed ^ (row * 0x9E3779B9u) ^ (h * 0x85EBCA6Bu) ^ (c * 0xC2B2AE35u);
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) {  // sum of 4 uniforms ~ normal-ish
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        s += (float)(x & 0xFFFFFF) / 16777216.0f - 0.5f;
    }
    kv_store(cache, fmt, kv_cap, (size_t)row, h, c, s * amp * 1.7320508f);
}
