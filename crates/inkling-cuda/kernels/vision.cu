// Vision encoder (ViT of the mmproj) kernels: f16 tensor-core GEMM with fused epilogues, LayerNorm, patch /
// position embeddings, 2-D RoPE, non-causal flash attention (head dim 64) and the 2x2 patch merge.
//
// Layout: the residual stream is f32 [N][D] (row 0 = CLS, rows 1.. = patches in row-major order); GEMM inputs are
// f16 row-major, weights f16 [N_out][K] (y = x W^T), K a multiple of 32. Uses mma.sync m16n8k16 (sm_80+) through
// the fa:: helpers of prefill_attn.cu.

namespace vit {
constexpr int BM = 128, BN = 128, BK = 32, SS = 40;  // SS: shared row stride in halves (80 B, ldmatrix conflict-free)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
}  // namespace vit

// Y = X W^T (+ bias), epilogue by mode:
//   0: f16 out = acc + b      1: f16 out = gelu(acc + b)
//   2: f32 out += s[n] * (acc + b)  (residual with layer scale)      3: f32 out = acc + b
// Grid (N / 128, ceil(M / 128)), 256 threads: 8 warps as 2 (m) x 4 (n), warp tile 64 x 32. N % 128 == 0.
extern "C" __global__ void __launch_bounds__(256) ink_vit_gemm(const __half* __restrict__ X, int ldx, const __half* __restrict__ W,
                                                               const float* __restrict__ bias, const float* __restrict__ scale, void* out,
                                                               int ldo, int M, int K, int mode) {
    using namespace vit;
    __shared__ __align__(16) __half as[2][BM * SS];
    __shared__ __align__(16) __half ws[2][BN * SS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // each thread copies 2 x 16 B of A and of W per stage (128 rows x 4 chunks of 8 halves)
    auto stage = [&](int buf, int k0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 256 * i, r = idx >> 2, c = (idx & 3) * 8;
            const int m = m0 + r;
            cp_async16(&as[buf][r * SS + c], X + (size_t)min(m, M - 1) * ldx + k0 + c, m < M);
            cp_async16(&ws[buf][r * SS + c], W + (size_t)(n0 + r) * K + k0 + c, true);
        }
        cp_commit();
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int nk = K / BK;
    stage(0, 0);
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) {
            stage(buf ^ 1, (kt + 1) * BK);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const __half* A = as[buf];
        const __half* B = ws[buf];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bf[2][4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) fa::ldsm_x4(af[mt], A + (wm * 64 + mt * 16 + (lane & 15)) * SS + kk + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int mi = lane >> 3, row = lane & 7;
                fa::ldsm_x4(bf[np], B + (wn * 32 + np * 16 + (mi >> 1) * 8 + row) * SS + kk + (mi & 1) * 8);
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    fa::mma_f16(acc[mt][2 * np], af[mt], bf[np][0], bf[np][1]);
                    fa::mma_f16(acc[mt][2 * np + 1], af[mt], bf[np][2], bf[np][3]);
                }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int n = n0 + wn * 32 + nt * 8 + 2 * tig;
            const float b0 = bias ? bias[n] : 0.0f, b1 = bias ? bias[n + 1] : 0.0f;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int m = m0 + wm * 64 + mt * 16 + g + 8 * hh;
                if (m >= M) continue;
                float v0 = acc[mt][nt][2 * hh] + b0, v1 = acc[mt][nt][2 * hh + 1] + b1;
                if (mode <= 1) {
                    if (mode == 1) {
                        v0 = gelu_erf(v0);
                        v1 = gelu_erf(v1);
                    }
                    *reinterpret_cast<__half2*>(static_cast<__half*>(out) + (size_t)m * ldo + n) = __floats2half2_rn(v0, v1);
                } else {
                    float2* p = reinterpret_cast<float2*>(static_cast<float*>(out) + (size_t)m * ldo + n);
                    if (mode == 2) {
                        float2 r = *p;
                        r.x += scale[n] * v0;
                        r.y += scale[n + 1] * v1;
                        *p = r;
                    } else {
                        *p = make_float2(v0, v1);
                    }
                }
            }
        }
    }
}

// LayerNorm per row (one block of 256 threads per row): f32 in, f16 out (`out16`) or f32 out (`out32`, may alias x).
extern "C" __global__ void __launch_bounds__(256) ink_vit_ln(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                             __half* __restrict__ out16, float* out32, int D, float eps) {
    const float* r = x + (size_t)blockIdx.x * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) s += r[i];
    const float mean = block_sum(s) / D;
    float v = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) {
        const float t = r[i] - mean;
        v += t * t;
    }
    const float inv = rsqrtf(block_sum(v) / D + eps);
    for (int i = threadIdx.x; i < D; i += 256) {
        const float y = (r[i] - mean) * inv * w[i] + b[i];
        if (out16) out16[(size_t)blockIdx.x * D + i] = __float2half_rn(y);
        else out32[(size_t)blockIdx.x * D + i] = y;
    }
}

// Embeddings: row 0 = CLS + its position embedding; row 1 + y*Wp + x (already holding the patch projection) += the
// learned position grid (G x G) resized to Wp x Hp, bilinear with half-pixel centers (ggml_interpolate semantics).
extern "C" __global__ void __launch_bounds__(256) ink_vit_embed(float* __restrict__ res, const float* __restrict__ grid, int G,
                                                                const float* __restrict__ cls, const float* __restrict__ cls_pos, int Wp,
                                                                int Hp, int D) {
    const int row = blockIdx.x;
    float* o = res + (size_t)row * D;
    if (row == 0) {
        for (int i = threadIdx.x; i < D; i += 256) o[i] = cls[i] + cls_pos[i];
        return;
    }
    const int p = row - 1, px = p % Wp, py = p / Wp;
    const float fx = ((float)px + 0.5f) * (float)G / (float)Wp - 0.5f;
    const float fy = ((float)py + 0.5f) * (float)G / (float)Hp - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = max(0, min(x0, G - 1));
    x1 = max(0, min(x1, G - 1));
    y0 = max(0, min(y0, G - 1));
    y1 = max(0, min(y1, G - 1));
    const float dx = fminf(fmaxf(fx - (float)x0, 0.0f), 1.0f), dy = fminf(fmaxf(fy - (float)y0, 0.0f), 1.0f);
    const float* a = grid + (size_t)(y0 * G + x0) * D;
    const float* bq = grid + (size_t)(y0 * G + x1) * D;
    const float* c = grid + (size_t)(y1 * G + x0) * D;
    const float* d = grid + (size_t)(y1 * G + x1) * D;
    if (Wp == G && Hp == G) {
        for (int i = threadIdx.x; i < D; i += 256) o[i] += grid[(size_t)p * D + i];
        return;
    }
    for (int i = threadIdx.x; i < D; i += 256)
        o[i] += a[i] * (1 - dx) * (1 - dy) + bq[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy + d[i] * dx * dy;
}

// 2-D RoPE on q and k of the fused qkv rows (f16 [N][3 * 1024], 16 heads x 64): dims [0, 32) rotate with the
// column position, [32, 64) with the row position (adjacent pairs, theta = pos * 10000^(-2j/32)). CLS is (0, 0),
// patch (x, y) is (x + 1, y + 1). Grid N, 256 threads: one thread per (q|k, head, pair) = 2 * 16 * 32 pairs / 4.
extern "C" __global__ void __launch_bounds__(256) ink_vit_rope(__half* __restrict__ qkv, int Wp, float theta_base) {
    const int row = blockIdx.x;
    const int pw = row == 0 ? 0 : (row - 1) % Wp + 1;
    const int ph = row == 0 ? 0 : (row - 1) / Wp + 1;
    __half* r = qkv + (size_t)row * 3072;
    for (int t = threadIdx.x; t < 2 * 16 * 32; t += 256) {
        const int which = t >> 9, head = (t >> 5) & 15, pr = t & 31;  // pr: pair 0..31 within the head
        const int half_ = pr >> 4, j = pr & 15;
        const float pos = half_ ? (float)ph : (float)pw;
        const float ang = pos * powf(theta_base, -2.0f * (float)j / 32.0f);
        float sn, cs;
        sincosf(ang, &sn, &cs);
        __half2* p = reinterpret_cast<__half2*>(r + which * 1024 + head * 64 + half_ * 32 + 2 * j);
        const float2 x = __half22float2(*p);
        *p = __floats2half2_rn(x.x * cs - x.y * sn, x.x * sn + x.y * cs);
    }
}

// Non-causal attention over all N tokens, 16 heads of 64: grid (ceil(N / 64), 16), 128 threads (warp = 16 query
// rows). Q/K/V come from the fused qkv rows; output f16 [N][1024].
extern "C" __global__ void __launch_bounds__(128) ink_vit_attn(const __half* __restrict__ qkv, __half* __restrict__ out, int N) {
    using namespace fa;
    constexpr int D64 = 64, B = 64, S = 72;
    __shared__ __align__(16) __half ks[B * S];
    __shared__ __align__(16) __half vs[B * S];
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int qw = blockIdx.x * 64 + 16 * warp;
    const int ra = min(qw + g, N - 1), rb = min(qw + g + 8, N - 1);

    uint32_t qf[4][4];
    {
        const __half* qa = qkv + (size_t)ra * 3072 + h * D64;
        const __half* qb = qkv + (size_t)rb * 3072 + h * D64;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            qf[k][0] = *reinterpret_cast<const uint32_t*>(qa + 16 * k + 2 * tig);
            qf[k][1] = *reinterpret_cast<const uint32_t*>(qb + 16 * k + 2 * tig);
            qf[k][2] = *reinterpret_cast<const uint32_t*>(qa + 16 * k + 8 + 2 * tig);
            qf[k][3] = *reinterpret_cast<const uint32_t*>(qb + 16 * k + 8 + 2 * tig);
        }
    }
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float scale = 0.125f;  // 1 / sqrt(64)

    for (int k0 = 0; k0 < N; k0 += B) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = threadIdx.x + 128 * i, rr = idx >> 3, pp = idx & 7;
            const int j = k0 + rr;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (j < N) {
                kv = *reinterpret_cast<const uint4*>(qkv + (size_t)j * 3072 + 1024 + h * D64 + pp * 8);
                vv = *reinterpret_cast<const uint4*>(qkv + (size_t)j * 3072 + 2048 + h * D64 + pp * 8);
            }
            *reinterpret_cast<uint4*>(ks + rr * S + pp * 8) = kv;
            *reinterpret_cast<uint4*>(vs + rr * S + pp * 8) = vv;
        }
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                const int mi = lane >> 3, row = lane & 7;
                uint32_t b[4];
                ldsm_x4(b, ks + (16 * np + (mi >> 1) * 8 + row) * S + 16 * k + (mi & 1) * 8);
                mma_f16(s[2 * np], qf[k], b[0], b[1]);
                mma_f16(s[2 * np + 1], qf[k], b[2], b[3]);
            }
        }
        float tma = -INFINITY, tmb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int j = k0 + 8 * nt + 2 * tig + e;
                float x0 = s[nt][e] * scale, x1 = s[nt][2 + e] * scale;
                if (j >= N) x0 = x1 = -INFINITY;
                s[nt][e] = x0;
                s[nt][2 + e] = x1;
                tma = fmaxf(tma, x0);
                tmb = fmaxf(tmb, x1);
            }
        }
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 1));
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 2));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 1));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 2));
        const float na = fmaxf(ma, tma), nb = fmaxf(mb, tmb);
        const float ca = __expf(ma - na), cb = __expf(mb - nb);
        ma = na;
        mb = nb;
        float sa = 0.0f, sb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = __expf(s[nt][0] - na);
            s[nt][1] = __expf(s[nt][1] - na);
            s[nt][2] = __expf(s[nt][2] - nb);
            s[nt][3] = __expf(s[nt][3] - nb);
            sa += s[nt][0] + s[nt][1];
            sb += s[nt][2] + s[nt][3];
        }
        la = la * ca + sa;
        lb = lb * cb + sb;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[i][0] *= ca;
            o[i][1] *= ca;
            o[i][2] *= cb;
            o[i][3] *= cb;
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint32_t pf[4] = {pack2(s[2 * kk][0], s[2 * kk][1]), pack2(s[2 * kk][2], s[2 * kk][3]), pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                    pack2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
                const int mi = lane >> 3, row = lane & 7;
                uint32_t b[4];
                ldsm_x4_t(b, vs + (16 * kk + (mi & 1) * 8 + row) * S + 16 * dp + (mi >> 1) * 8);
                mma_f16(o[2 * dp], pf, b[0], b[1]);
                mma_f16(o[2 * dp + 1], pf, b[2], b[3]);
            }
        }
    }
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    const float ia = 1.0f / la, ib = 1.0f / lb;
    const int ta = qw + g, tb = qw + g + 8;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int dim = h * D64 + 8 * nt + 2 * tig;
        if (ta < N) *reinterpret_cast<__half2*>(out + (size_t)ta * 1024 + dim) = __floats2half2_rn(o[nt][0] * ia, o[nt][1] * ia);
        if (tb < N) *reinterpret_cast<__half2*>(out + (size_t)tb * 1024 + dim) = __floats2half2_rn(o[nt][2] * ib, o[nt][3] * ib);
    }
}

// 2x2 patch merge (CLS dropped): merged token (i, j) = [p(2i, 2j), p(2i, 2j+1), p(2i+1, 2j), p(2i+1, 2j+1)] as f16.
// Grid (Wp/2 * Hp/2), 256 threads.
extern "C" __global__ void __launch_bounds__(256) ink_vit_merge(const float* __restrict__ res, __half* __restrict__ out, int Wp, int D) {
    const int t = blockIdx.x, W2 = Wp / 2, i = t / W2, j = t % W2;
    __half* o = out + (size_t)t * 4 * D;
    for (int e = threadIdx.x; e < 4 * D; e += 256) {
        const int q = e / D, d = e % D;
        const int y = 2 * i + (q >> 1), x = 2 * j + (q & 1);
        o[e] = __float2half_rn(res[(size_t)(1 + y * Wp + x) * D + d]);
    }
}
