// Prefill attention on tensor cores (flash-attention style, mma.sync m16n8k16 f16 -> f32; two m16n8k8 on sm_75).
//
// Grid (ceil(T / 64), 24 q heads), 128 threads; warp w owns query tokens q0 + 16w .. +15 of the chunk (position
// step[1] + token). Q (already clipped and tau-scaled) lives in registers as A fragments; K/V tiles of 64 keys are
// staged in shared memory with a 272-byte row stride (ldmatrix conflict-free; V read with .trans). Per key tile:
//   S = Q K^T / sqrt(128) + bias, bias(q, j) = r_q . relbias[pos_q - j] for 0 <= dist < extent: per warp one MMA
//       per 8 distances over the tile's 80-wide distance band (k = 16 = rel dim), staged in shared memory as f16
//       and gathered per score;
//   causal / sliding-window mask, online softmax, O += P V with P reused from the S accumulators as A fragments.
// The attention sink joins the final normalization; the sigmoid output gate and the q8_1 copy of the output (for
// the wo projection) are fused in the epilogue. KV cache rows are pos % kv_cap (ring for sliding-window layers).

namespace fa {
constexpr int HD = 128, BQ = 64, BK = 64, KS = 136;  // KS: K/V shared row stride in halves (272 B)
constexpr int NB = 80;                               // distance band per warp tile (79 used)

__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    // Turing: k 0..7 (a0, a1, b0) then k 8..15 (a2, a3, b1)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
}  // namespace fa

extern "C" __global__ void __launch_bounds__(128) ink_attn_prefill(const float* __restrict__ qrg, int qs, int roff, int goff,
                                                                   const float* __restrict__ sink, int ss, const void* __restrict__ Kc,
                                                                   const void* __restrict__ Vc, int kv_cap, int window, int extent,
                                                                   const float* __restrict__ relbias, const int* __restrict__ step, int T,
                                                                   float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, int kvfmt) {
    using namespace fa;
    __shared__ __align__(16) __half ks[BK * KS];
    __shared__ __align__(16) __half vs[BK * KS];
    __shared__ __align__(16) __half bs[4][16 * NB];

    const int h = blockIdx.y, kvh = h / 3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int P0 = step[1];
    const int q0 = blockIdx.x * BQ, qw = q0 + 16 * warp;  // chunk token index of this warp's first row
    const int tqa = qw + g, tqb = qw + g + 8;             // the two rows of this thread
    const bool va = tqa < T, vb = tqb < T;
    const int pa = P0 + tqa, pb = P0 + tqb;
    const int tlast = min(q0 + BQ, T) - 1;
    const int jhi = P0 + tlast;
    const int jlo = window ? max(0, P0 + q0 - window + 1) : 0;

    // Q and R fragments (rows g, g+8 of the warp tile)
    uint32_t qf[8][4], rf[4];
    {
        const float* qa = qrg + (size_t)min(tqa, T - 1) * qs + h * HD;
        const float* qb = qrg + (size_t)min(tqb, T - 1) * qs + h * HD;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float2 x0 = *reinterpret_cast<const float2*>(qa + 16 * k + 2 * tig);
            const float2 x1 = *reinterpret_cast<const float2*>(qb + 16 * k + 2 * tig);
            const float2 x2 = *reinterpret_cast<const float2*>(qa + 16 * k + 8 + 2 * tig);
            const float2 x3 = *reinterpret_cast<const float2*>(qb + 16 * k + 8 + 2 * tig);
            qf[k][0] = pack2(x0.x, x0.y);
            qf[k][1] = pack2(x1.x, x1.y);
            qf[k][2] = pack2(x2.x, x2.y);
            qf[k][3] = pack2(x3.x, x3.y);
        }
        const float* ra = qrg + (size_t)min(tqa, T - 1) * qs + roff + h * 16;
        const float* rb = qrg + (size_t)min(tqb, T - 1) * qs + roff + h * 16;
        rf[0] = pack2(ra[2 * tig], ra[2 * tig + 1]);
        rf[1] = pack2(rb[2 * tig], rb[2 * tig + 1]);
        rf[2] = pack2(ra[8 + 2 * tig], ra[9 + 2 * tig]);
        rf[3] = pack2(rb[8 + 2 * tig], rb[9 + 2 * tig]);
    }

    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float scale = rsqrtf((float)HD);
    __half* bw = bs[warp];

    for (int k0 = jlo; k0 <= jhi; k0 += BK) {
        // stage K, V (dequantized to f16): 64 rows x 8 parts of 16 channels, 4 + 4 per thread
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = threadIdx.x + 128 * i, rr = idx >> 3, pp = idx & 7;
            const int j = k0 + rr;
            if (j <= jhi) {
                kv_load16(Kc, kvfmt, (size_t)(j % kv_cap), kvh, pp, ks + rr * KS + pp * 16);
                kv_load16(Vc, kvfmt, (size_t)(j % kv_cap), kvh, pp, vs + rr * KS + pp * 16);
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(ks + rr * KS + pp * 16)[0] = z;
                reinterpret_cast<uint4*>(ks + rr * KS + pp * 16)[1] = z;
                reinterpret_cast<uint4*>(vs + rr * KS + pp * 16)[0] = z;
                reinterpret_cast<uint4*>(vs + rr * KS + pp * 16)[1] = z;
            }
        }
        __syncthreads();
        // warp-level skip: keys entirely after this warp's last row or before its first row's window
        const int wlast = P0 + min(qw + 15, T - 1);
        if (qw >= T || k0 > wlast || (window && k0 + BK - 1 < P0 + qw - window + 1)) continue;

        // S = Q K^T
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                const int mi = lane >> 3, row = lane & 7;
                uint32_t b[4];
                ldsm_x4(b, ks + (16 * np + (mi >> 1) * 8 + row) * KS + 16 * k + (mi & 1) * 8);
                mma_f16(s[2 * np], qf[k], b[0], b[1]);
                mma_f16(s[2 * np + 1], qf[k], b[2], b[3]);
            }
        }
        // relative-position bias band: dist(qr, kk) = dmin + qr + 63 - kk, qr in 0..15, kk in 0..63
        const int dmin = P0 + qw - (k0 + BK - 1);
        const bool use_bias = dmin < extent && dmin + 78 >= 0;
        if (use_bias) {
#pragma unroll
            for (int nb = 0; nb < NB / 8; ++nb) {
                const int dist = dmin + 8 * nb + g;
                uint32_t b0 = 0, b1 = 0;
                if (dist >= 0 && dist < extent) {
                    const float* rbp = relbias + (size_t)dist * 16;
                    b0 = pack2(rbp[2 * tig], rbp[2 * tig + 1]);
                    b1 = pack2(rbp[8 + 2 * tig], rbp[9 + 2 * tig]);
                }
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_f16(c, rf, b0, b1);
                *reinterpret_cast<__half2*>(bw + g * NB + 8 * nb + 2 * tig) = __floats2half2_rn(c[0], c[1]);
                *reinterpret_cast<__half2*>(bw + (g + 8) * NB + 8 * nb + 2 * tig) = __floats2half2_rn(c[2], c[3]);
            }
            __syncwarp();
        }
        // scale, bias, mask; tile row maxima
        float tma = -INFINITY, tmb = -INFINITY;
        const int loa = window ? pa - window + 1 : 0, lob = window ? pb - window + 1 : 0;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int kk = 8 * nt + 2 * tig + e, j = k0 + kk;
                float x0 = s[nt][e] * scale, x1 = s[nt][2 + e] * scale;
                if (use_bias) {
                    x0 += __half2float(bw[g * NB + g + 63 - kk]);
                    x1 += __half2float(bw[(g + 8) * NB + g + 8 + 63 - kk]);
                }
                if (!va || j > pa || j < loa) x0 = -INFINITY;
                if (!vb || j > pb || j < lob) x1 = -INFINITY;
                s[nt][e] = x0;
                s[nt][2 + e] = x1;
                tma = fmaxf(tma, x0);
                tmb = fmaxf(tmb, x1);
            }
        }
        if (use_bias) __syncwarp();  // bias reads done before the next tile overwrites the band
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 1));
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 2));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 1));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 2));
        const float na = fmaxf(ma, tma), nbm = fmaxf(mb, tmb);
        const float ua = na == -INFINITY ? 0.0f : na, ub = nbm == -INFINITY ? 0.0f : nbm;
        const float ca = __expf(ma - ua), cb = __expf(mb - ub);  // 0 while the row had no valid key
        ma = na;
        mb = nbm;
        float sa = 0.0f, sb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = __expf(s[nt][0] - ua);
            s[nt][1] = __expf(s[nt][1] - ua);
            s[nt][2] = __expf(s[nt][2] - ub);
            s[nt][3] = __expf(s[nt][3] - ub);
            sa += s[nt][0] + s[nt][1];
            sb += s[nt][2] + s[nt][3];
        }
        la = la * ca + sa;
        lb = lb * cb + sb;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            o[i][0] *= ca;
            o[i][1] *= ca;
            o[i][2] *= cb;
            o[i][3] *= cb;
        }
        // O += P V
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint32_t pf[4] = {pack2(s[2 * kk][0], s[2 * kk][1]), pack2(s[2 * kk][2], s[2 * kk][3]), pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                    pack2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 8; ++dp) {
                const int mi = lane >> 3, row = lane & 7;
                uint32_t b[4];
                ldsm_x4_t(b, vs + (16 * kk + (mi & 1) * 8 + row) * KS + 16 * dp + (mi >> 1) * 8);
                mma_f16(o[2 * dp], pf, b[0], b[1]);
                mma_f16(o[2 * dp + 1], pf, b[2], b[3]);
            }
        }
    }

    // epilogue: row sums over the quad, sink, normalization, gate, outputs (+ q8_1 per 32 dims)
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int tq = half ? tqb : tqa;
        const bool valid = half ? vb : va;
        const float m = half ? mb : ma, l = half ? lb : la;
        const float sk = valid ? sink[(size_t)tq * ss + h] : 0.0f;
        const float M = fmaxf(m, sk);
        const float e = m == -INFINITY ? 0.0f : __expf(m - M);
        const float inv = e / (l * e + __expf(sk - M));
        const float* gp = qrg + (size_t)min(tq, T - 1) * qs + goff + h * HD;
        float v[16][2];
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const int dim = 8 * nt + 2 * tig;
            const float2 gt = *reinterpret_cast<const float2*>(gp + dim);
            v[nt][0] = o[nt][2 * half] * inv * (1.0f / (1.0f + __expf(-gt.x)));
            v[nt][1] = o[nt][2 * half + 1] * inv * (1.0f / (1.0f + __expf(-gt.y)));
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {  // 32-dim group b = n-tiles 4b..4b+3 across the 4 lanes of the quad
            float amax = 0.0f;
#pragma unroll
            for (int nt = 4 * b; nt < 4 * b + 4; ++nt) amax = fmaxf(amax, fmaxf(fabsf(v[nt][0]), fabsf(v[nt][1])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
            if (valid) {
#pragma unroll
                for (int nt = 4 * b; nt < 4 * b + 4; ++nt) {
                    const int dim = 8 * nt + 2 * tig;
                    const size_t base = (size_t)tq * 24 * HD + h * HD + dim;
                    *reinterpret_cast<float2*>(out + base) = make_float2(v[nt][0], v[nt][1]);
                    oq[base] = (int8_t)__float2int_rn(v[nt][0] * id);
                    oq[base + 1] = (int8_t)__float2int_rn(v[nt][1] * id);
                }
                if (tig == 0) od[(size_t)tq * 24 * 4 + h * 4 + b] = amax / 127.0f;
            }
        }
    }
}
