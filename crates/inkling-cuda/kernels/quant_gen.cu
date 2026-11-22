// Generic dense kernels for weight matrices in any supported format (codecs.cuh): the decode matvec, the prefill
// GEMM and the token-embedding gather. fmt 0 = repacked Q8_0 (qs int8 [N][K] + d half [N][K/32], the tuned
// layout), otherwise a ggml type id with the matrix in its GGUF block layout. Used when a layer's matrices are not
// all Q8_0 (the all-Q8_0 case keeps ink_mv_q8q / ink_gemm_q8). Included after prefill_ops.cu.
#include "codecs.cuh"

namespace qg {
// group g of row `row` (fmt 0: repacked Q8_0)
__device__ __forceinline__ cdc::G32 load_group(const uint8_t* w, const __half* wd, int fmt, size_t row, int K, int g) {
    if (fmt == 0) {
        cdc::G32 r;
        const int4* p = reinterpret_cast<const int4*>(w + row * K + 32 * g);
        const int4 a = __ldg(p), b = __ldg(p + 1);
        r.q[0] = a.x; r.q[1] = a.y; r.q[2] = a.z; r.q[3] = a.w;
        r.q[4] = b.x; r.q[5] = b.y; r.q[6] = b.z; r.q[7] = b.w;
        r.s0 = r.s1 = __half2float(wd[row * (K / 32) + g]);
        r.m = 0.0f;
        return r;
    }
    return cdc::g32(fmt, w + row * cdc::row_bytes(fmt, K), g);
}
}  // namespace qg

// ---- decode matvec: y[n] = W[n] . x over up to 8 segments, int8 activations (q8_1 per 32); warp per row
struct MvSegG {
    const uint8_t* w;
    const __half* wd;
    const int8_t* xq;
    const float* xd;
    float* y;
    int n, fmt;
};
struct MvGen {
    MvSegG s[8];
    int nseg, K;
};

extern "C" __global__ void __launch_bounds__(256) ink_mv_gen(MvGen a) {
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    int seg = 0;
    while (seg < a.nseg && row >= a.s[seg].n) { row -= a.s[seg].n; ++seg; }
    if (seg >= a.nseg) return;
    const MvSegG& s = a.s[seg];
    const int K = a.K, lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int g = lane; g < K / 32; g += 32) {
        const cdc::G32 q = qg::load_group(s.w, s.wd, s.fmt, row, K, g);
        const int4 xa = *reinterpret_cast<const int4*>(s.xq + 32 * g), xb = *reinterpret_cast<const int4*>(s.xq + 32 * g + 16);
        const int x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        const float xs = s.xd[g];
        acc += cdc::g32_dot(q, x, xs, xs);
    }
    acc = warp_sum(acc);
    if (lane == 0) s.y[row] = acc;
}

// ---- prefill GEMM: Y[t][n] = sum_k X[t][k] W[n][k], tiles as ink_gemm_q8 (64 tokens x 64 rows, K by 128); the
// weight tile is decoded per group of 32 into int8 + two scales + offset, multiplied per 16 (m16n8k16)
struct GemmSegG {
    const uint8_t* w;
    const __half* wd;
    const int8_t* x;  // [T][K]
    const float* xd;  // [T][K/32]
    float* y;         // [T][ldy]
    int n, ldy, fmt, pad;
};
struct GemmGen {
    GemmSegG s[8];
    int nseg, T, K;
};

extern "C" __global__ void __launch_bounds__(128) ink_gemm_gen(GemmGen A) {
    using namespace gq8;
    __shared__ __align__(16) int8_t xs[BT * RS];
    __shared__ __align__(16) int8_t ws[BN * RS];
    __shared__ float xds[BT][KC / 32];
    __shared__ float wsc[BN][KC / 16];
    __shared__ float wmin[BN][KC / 32];
    __shared__ int xsum[BT][KC / 32];

    int tile = blockIdx.x, seg = 0;
    while (seg < A.nseg - 1 && tile >= (A.s[seg].n + BN - 1) / BN) { tile -= (A.s[seg].n + BN - 1) / BN; ++seg; }
    const GemmSegG& S = A.s[seg];
    const int n0 = tile * BN, t0 = blockIdx.y * BT, K = A.K, T = A.T;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;

    float acc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += KC) {
        __syncthreads();
        // activations: 64 rows x 128 bytes (4 uint4 per thread) + scales, and the per-group sums
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
            const uint4 v = t0 + r < T ? *reinterpret_cast<const uint4*>(S.x + (size_t)(t0 + r) * K + k0 + 16 * c) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(xs + r * RS + 16 * c) = v;
            int sum = __dp4a((int)v.x, 0x01010101, 0);
            sum = __dp4a((int)v.y, 0x01010101, sum);
            sum = __dp4a((int)v.z, 0x01010101, sum);
            sum = __dp4a((int)v.w, 0x01010101, sum);
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            if ((c & 1) == 0) xsum[r][c >> 1] = sum;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
            xds[r][c] = t0 + r < T ? S.xd[(size_t)(t0 + r) * (K / 32) + k0 / 32 + c] : 0.0f;
        }
        // weights: 64 rows x 4 groups, two per thread
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
            cdc::G32 q = {};
            if (n0 + r < S.n) q = qg::load_group(S.w, S.wd, S.fmt, n0 + r, K, k0 / 32 + c);
            int4* dst = reinterpret_cast<int4*>(ws + r * RS + 32 * c);
            dst[0] = make_int4(q.q[0], q.q[1], q.q[2], q.q[3]);
            dst[1] = make_int4(q.q[4], q.q[5], q.q[6], q.q[7]);
            wsc[r][2 * c] = q.s0;
            wsc[r][2 * c + 1] = q.s1;
            wmin[r][c] = q.m;
        }
        __syncthreads();
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
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int n = wn * 32 + nt * 8 + 2 * tig;
                const float w00 = wsc[n][2 * kb], w01 = wsc[n][2 * kb + 1], w10 = wsc[n + 1][2 * kb], w11 = wsc[n + 1][2 * kb + 1];
                const float m0 = wmin[n][kb], m1 = wmin[n + 1][kb];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int r0 = wm * 32 + mt * 16 + g;
                    const float sx0 = xds[r0][kb], sx1 = xds[r0 + 8][kb];
                    const float xs0 = (float)xsum[r0][kb], xs1 = (float)xsum[r0 + 8][kb];
                    int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                    moeg::mma_s8_k16(c0, af[mt][0], af[mt][1], bf[nt][0]);
                    moeg::mma_s8_k16(c1, af[mt][2], af[mt][3], bf[nt][1]);
                    acc[mt][nt][0] = fmaf(sx0, fmaf((float)c0[0], w00, fmaf((float)c1[0], w01, -m0 * xs0)), acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf(sx0, fmaf((float)c0[1], w10, fmaf((float)c1[1], w11, -m1 * xs0)), acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf(sx1, fmaf((float)c0[2], w00, fmaf((float)c1[2], w01, -m0 * xs1)), acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf(sx1, fmaf((float)c0[3], w10, fmaf((float)c1[3], w11, -m1 * xs1)), acc[mt][nt][3]);
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = t0 + wm * 32 + mt * 16 + g + 8 * h;
                const int n = n0 + wn * 32 + nt * 8 + 2 * tig;
                if (t >= T) continue;
                float* y = S.y + (size_t)t * S.ldy;
                if (n < S.n) y[n] = acc[mt][nt][2 * h];
                if (n + 1 < S.n) y[n + 1] = acc[mt][nt][2 * h + 1];
            }
}

// ---- token embedding gather for a natively stored table: out[b] = rmsnorm(dequant(emb[tok[b]])) * w, grid = tokens
// (decode passes the step buffer: its first int is the token)
extern "C" __global__ void ink_embed_gen(const uint8_t* __restrict__ emb, int fmt, const int* __restrict__ tokens, const float* __restrict__ w,
                                         float* __restrict__ out, int D, float eps) {
    const uint8_t* row = emb + (size_t)tokens[blockIdx.x] * cdc::row_bytes(fmt, D);
    out += (size_t)blockIdx.x * D;
    float ss = 0.0f;
    for (int g = threadIdx.x; g < D / 32; g += blockDim.x) {
        const cdc::G32 q = cdc::g32(fmt, row, g);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float v = (i < 16 ? q.s0 : q.s1) * (float)(int8_t)((q.q[i >> 2] >> (8 * (i & 3))) & 0xff) - q.m;
            out[32 * g + i] = v;
            ss += v * v;
        }
    }
    const float inv = rsqrtf(block_sum(ss) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = out[i] * inv * w[i];
}
