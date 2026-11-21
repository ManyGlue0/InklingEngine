// Prefill kernels (batches of T tokens). Included after decode.cu.
//
// ink_gemm_q8: Y[t][n] = sum_k X[t][k] W[n][k] for Q8_0 weights (SoA: int8 [N][K], half scale per 32) and q8_1
// activations (int8 [T][K], f32 scale per 32) on int8 tensor cores (mma.sync m16n8k32, sm_80+). Integer sums are
// formed per 32-wide scale block and scaled in f32, exactly the decode dp4a numerics up to summation order.
// Block tile 64 tokens x 64 rows, 4 warps (2 x 2) of 32 x 32; K staged 128 at a time in shared memory with a
// 144-byte row stride (fragment loads conflict-free), the next chunk prefetched into registers.
// Up to 4 segments (independent X / W / Y / N, shared K) in one launch; the grid's x walks the n-tiles of all.

struct GemmSeg {
    const int8_t* w;   // [N][K]
    const __half* wd;  // [N][K/32]
    const int8_t* x;   // [T][K]
    const float* xd;   // [T][K/32]
    float* y;          // [T][ldy]
    int n, ldy;
};
struct GemmArgs {
    GemmSeg s[4];
    int nseg, T, K;
};

namespace gq8 {
constexpr int BT = 64, BN = 64, KC = 128, RS = 144;  // RS: shared row stride (bytes)

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
}  // namespace gq8

extern "C" __global__ void __launch_bounds__(128) ink_gemm_q8(GemmArgs A) {
    using namespace gq8;
    __shared__ __align__(16) int8_t xs[BT * RS];
    __shared__ __align__(16) int8_t ws[BN * RS];
    __shared__ float xds[BT][KC / 32];
    __shared__ float wds[BN][KC / 32];

    // segment of this n-tile
    int tile = blockIdx.x, seg = 0;
    while (seg < A.nseg - 1 && tile >= (A.s[seg].n + BN - 1) / BN) { tile -= (A.s[seg].n + BN - 1) / BN; ++seg; }
    const GemmSeg& S = A.s[seg];
    const int n0 = tile * BN, t0 = blockIdx.y * BT, K = A.K, T = A.T;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;  // warp tile: tokens wm*32.., rows wn*32..

    // global -> register staging: 4 uint4 of X and of W per thread (row = idx / 8, chunk = idx % 8), 2+2 scales
    uint4 rx[4], rw[4];
    float rxd[2], rwd[2];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
            rx[i] = t0 + r < T ? *reinterpret_cast<const uint4*>(S.x + (size_t)(t0 + r) * K + k0 + 16 * c) : make_uint4(0, 0, 0, 0);
            rw[i] = n0 + r < S.n ? __ldg(reinterpret_cast<const uint4*>(S.w + (size_t)(n0 + r) * K + k0 + 16 * c)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
            rxd[i] = t0 + r < T ? S.xd[(size_t)(t0 + r) * (K / 32) + k0 / 32 + c] : 0.0f;
            rwd[i] = n0 + r < S.n ? __half2float(S.wd[(size_t)(n0 + r) * (K / 32) + k0 / 32 + c]) : 0.0f;
        }
    };
    auto commit = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 128 * i, r = idx >> 3, c = idx & 7;
            *reinterpret_cast<uint4*>(xs + r * RS + 16 * c) = rx[i];
            *reinterpret_cast<uint4*>(ws + r * RS + 16 * c) = rw[i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + 128 * i, r = idx >> 2, c = idx & 3;
            xds[r][c] = rxd[i];
            wds[r][c] = rwd[i];
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
        __syncthreads();  // previous chunk consumed
        commit();
        __syncthreads();
        if (k0 + KC < K) fetch(k0 + KC);  // in flight during the MMAs
#pragma unroll
        for (int kb = 0; kb < KC / 32; ++kb) {
            uint32_t af[2][4], bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r = wm * 32 + mt * 16 + g;
                const int8_t* p = xs + r * RS + kb * 32 + 4 * tig;
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
            float sx[2][2], sw[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                sx[mt][0] = xds[wm * 32 + mt * 16 + g][kb];
                sx[mt][1] = xds[wm * 32 + mt * 16 + g + 8][kb];
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                sw[nt][0] = wds[wn * 32 + nt * 8 + 2 * tig][kb];
                sw[nt][1] = wds[wn * 32 + nt * 8 + 2 * tig + 1][kb];
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, af[mt], bf[nt]);
                    acc[mt][nt][0] = fmaf((float)c[0], sx[mt][0] * sw[nt][0], acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf((float)c[1], sx[mt][0] * sw[nt][1], acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf((float)c[2], sx[mt][1] * sw[nt][0], acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf((float)c[3], sx[mt][1] * sw[nt][1], acc[mt][nt][3]);
                }
        }
    }
    // epilogue: C fragment rows g, g+8; cols 2 tig, 2 tig + 1
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
