// MoE expert kernels, v7 (included after decode.cu; uses ExpW, e2m1x2_gather, ue4m3_fast, perm16).
//
// Every lane streams 16-byte weight chunks: FP4 = 4 words (two 16-value sub-blocks, one ushort of UE4M3 scales),
// Q8 = 16 int8 values. All weight loads of a warp are issued before the activations are staged, so the DRAM latency
// overlaps the staging and the block barrier. Activations are int8 in shared memory (FP4 path: permuted per
// sub-block, uint4 u <-> bytes [32u, 32u + 32)); the two 16-byte halves of a lane's 32 bytes are read in an order
// swizzled so each quarter-warp touches all 32 banks once.
//
// The expert hidden h is q8_1-quantized in groups of 16 features (one FP4 sub-block of the down weights).

namespace ex7 {

constexpr int D = 2304, F = 1152;

__device__ __forceinline__ void ldx32(const int8_t* p, int swz, int4& a, int4& b) {
    const int4* v = reinterpret_cast<const int4*>(p);
    const int4 first = v[swz], second = v[swz ^ 1];
    a = swz ? second : first;
    b = swz ? first : second;
}

// dp4a of one FP4 uint4 (4 words) against 32 permuted int8 activations: per-sub-block integer sums.
__device__ __forceinline__ void fp4_isum(const uint4 q, const int4 xa, const int4 xb, int& s0, int& s1) {
    int2 v;
    v = e2m1x2_gather(q.x);
    s0 = __dp4a(v.y, xa.y, __dp4a(v.x, xa.x, 0));
    v = e2m1x2_gather(q.y);
    s0 = __dp4a(v.y, xa.w, __dp4a(v.x, xa.z, s0));
    v = e2m1x2_gather(q.z);
    s1 = __dp4a(v.y, xb.y, __dp4a(v.x, xb.x, 0));
    v = e2m1x2_gather(q.w);
    s1 = __dp4a(v.y, xb.w, __dp4a(v.x, xb.z, s1));
}

__device__ __forceinline__ int q8_isum(const uint4 q, const int4 x) {
    int s = __dp4a((int)q.x, x.x, 0);
    s = __dp4a((int)q.y, x.y, s);
    s = __dp4a((int)q.z, x.z, s);
    return __dp4a((int)q.w, x.w, s);
}

// Stage n int8 values (n % 16 == 0) from global into shared, permuting quads [0,1,2,3] -> [0,2,1,3] for FP4.
__device__ __forceinline__ void stage16(int8_t* dst, const int8_t* __restrict__ src, int n, bool perm) {
    for (int i = threadIdx.x; i < n / 16; i += blockDim.x) {
        int4 v = reinterpret_cast<const int4*>(src)[i];
        if (perm) v = make_int4(v.x, v.z, v.y, v.w);
        reinterpret_cast<int4*>(dst)[i] = v;
    }
}

// Stage the 8 experts' hidden vectors (8 x F int8, global contiguous) into shared rows HS = F + 64 apart, in one pass.
__device__ __forceinline__ void stage_h(int8_t* dst, const int8_t* __restrict__ src, bool perm) {
    constexpr int HS = F + 64, PER = F / 16;
    for (int i = threadIdx.x; i < 8 * PER; i += blockDim.x) {
        int4 v = reinterpret_cast<const int4*>(src)[i];
        if (perm) v = make_int4(v.x, v.z, v.y, v.w);
        const int j = i / PER, k = i - j * PER;
        reinterpret_cast<int4*>(dst + j * HS)[k] = v;
    }
}

}  // namespace ex7

// gate_up: grid (F/16, k), 256 threads. Warp w computes features f, f+1 (f = 16 blockIdx.x + 2w); lane group
// g = lane/8 owns row {gate f, up f, gate f+1, up f+1}[g], 8 lanes per row. The block quantizes its 16 outputs:
// hq int8 [k][F], hd [k][F/16].
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_gate_up7(ExpW w, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                                      const int* __restrict__ sel, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                                      float beta, float lb) {
    using namespace ex7;
    __shared__ __align__(16) int8_t xs[D];
    __shared__ float xds[D / 32];
    __shared__ float hv[16];
    const int t = blockIdx.y, e = sel[t];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 3, l8 = lane & 7;
    const int f = blockIdx.x * 16 + 2 * warp + (g >> 1);
    const size_t row = (size_t)e * 2 * F + (g & 1) * F + f;
    const int swz = (l8 >> 2) & 1;
    float acc = 0.0f;
    if (w.fmt == 1) {
        // FP4: row = 1152 B = 72 uint4 -> 9 per lane (u = l8 + 8i), scales 144 B -> one ushort per uint4
        const uint4* qr = reinterpret_cast<const uint4*>(w.qs + row * (D / 2));
        const unsigned short* sr = reinterpret_cast<const unsigned short*>(w.sc + row * (D / 16));
        uint4 q[9];
        unsigned short s[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            q[i] = __ldg(qr + l8 + 8 * i);
            s[i] = __ldg(sr + l8 + 8 * i);
        }
        stage16(xs, xq, D, true);
        for (int i = threadIdx.x; i < D / 32; i += blockDim.x) xds[i] = xd[i];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            const int u = l8 + 8 * i;
            int4 xa, xb;
            ldx32(xs + 32 * u, swz, xa, xb);
            int s0, s1;
            fp4_isum(q[i], xa, xb, s0, s1);
            acc = fmaf(fmaf((float)s0, ue4m3_fast(s[i] & 0xff), (float)s1 * ue4m3_fast(s[i] >> 8)), xds[u], acc);
        }
    } else {
        // Q8: row = 2304 B = 144 uint4 -> 18 per lane in two batches of 9; one half scale per 2 uint4
        const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(w.qs) + row * D);
        const __half* dr = reinterpret_cast<const __half*>(w.sc) + row * (D / 32);
        uint4 q[9];
        __half s[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            q[i] = __ldg(qr + l8 + 8 * i);
            s[i] = dr[(l8 + 8 * i) >> 1];
        }
        stage16(xs, xq, D, false);
        for (int i = threadIdx.x; i < D / 32; i += blockDim.x) xds[i] = xd[i];
        __syncthreads();
#pragma unroll 1
        for (int b = 0; b < 2; ++b) {
            if (b == 1) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    q[i] = __ldg(qr + 72 + l8 + 8 * i);
                    s[i] = dr[(72 + l8 + 8 * i) >> 1];
                }
            }
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                const int c = 72 * b + l8 + 8 * i;
                const int4 x = *reinterpret_cast<const int4*>(xs + 16 * c);
                acc = fmaf((float)q8_isum(q[i], x), __half2float(s[i]) * xds[c >> 1], acc);
            }
        }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    const float sc = w.scale ? w.scale[e] : 1.0f;
    const float up = __shfl_sync(0xffffffffu, acc, (lane & 16) | 8) * sc;  // lane 0 <- lane 8, lane 16 <- lane 24
    if ((lane & 15) == 0) {
        const float gt = acc * sc;
        const float a = beta * tanhf(gt / beta) * (1.0f / (1.0f + __expf(-gt)));
        hv[2 * warp + (lane >> 4)] = a * (lb * tanhf(up / lb));
    }
    __syncthreads();
    if (warp == 0) {
        const float v = hv[lane & 15];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        if (lane < 16) hq[t * F + blockIdx.x * 16 + lane] = amax == 0.0f ? 0 : (int8_t)roundf(v / d);
        if (lane == 0) hd[t * (F / 16) + blockIdx.x] = d;
    }
}

// down + expert sum + post conv + residual: grid D/8, 256 threads. Warp w owns output channel n = 8 blockIdx.x + w;
// lane group t = lane/4 is the t-th selected expert (4 lanes per row). y_n = sum_t wts_t s_e (W_e[n] . h_t), then
// x_n += (y_n + conv(y_n)) / sqrt(5) with the per-slot conv state; dbg (optional) gets the conv output.
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_down7(ExpW w, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                                                                   const int* __restrict__ sel, const float* __restrict__ wts,
                                                                   const float* __restrict__ cw, float* __restrict__ state,
                                                                   float* __restrict__ x, float* __restrict__ dbg) {
    using namespace ex7;
    constexpr int K = 8, HS = F + 64;  // expert stride in shared memory (== 64 mod 128 bytes)
    __shared__ __align__(16) int8_t hs[K * HS];
    __shared__ float hds[K][F / 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane >> 2, l4 = lane & 3;
    const int n = blockIdx.x * 8 + warp;
    const int e = sel[t];
    const size_t row = (size_t)e * D + n;
    const int swz = t & 1;
    // conv/residual operands for this channel, fetched early
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0, st0 = 0, st1 = 0, st2 = 0, xr = 0;
    if (lane == 0) {
        const float4 cv = *reinterpret_cast<const float4*>(cw + (size_t)n * 4);
        c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
        st0 = state[n]; st1 = state[D + n]; st2 = state[2 * D + n];
        xr = x[n];
    }
    const float sw = wts[t] * (w.scale ? w.scale[e] : 1.0f);
    float acc = 0.0f;
    if (w.fmt == 1) {
        // FP4: row = 576 B = 36 uint4 -> 9 per lane (u = l4 + 4i), one ushort of scales per uint4
        const uint4* qr = reinterpret_cast<const uint4*>(w.qs + row * (F / 2));
        const unsigned short* sr = reinterpret_cast<const unsigned short*>(w.sc + row * (F / 16));
        uint4 q[9];
        unsigned short s[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            q[i] = __ldg(qr + l4 + 4 * i);
            s[i] = __ldg(sr + l4 + 4 * i);
        }
        stage_h(hs, hq, true);
        for (int i = threadIdx.x; i < K * (F / 16); i += blockDim.x) hds[i / (F / 16)][i % (F / 16)] = hd[i];
        __syncthreads();
        const int8_t* h = hs + t * HS;
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            const int u = l4 + 4 * i;
            int4 xa, xb;
            ldx32(h + 32 * u, swz, xa, xb);
            int s0, s1;
            fp4_isum(q[i], xa, xb, s0, s1);
            acc = fmaf((float)s0, ue4m3_fast(s[i] & 0xff) * hds[t][2 * u], acc);
            acc = fmaf((float)s1, ue4m3_fast(s[i] >> 8) * hds[t][2 * u + 1], acc);
        }
    } else {
        // Q8: row = 1152 B = 72 uint4 -> 18 per lane in two batches of 9
        const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(w.qs) + row * F);
        const __half* dr = reinterpret_cast<const __half*>(w.sc) + row * (F / 32);
        uint4 q[9];
        __half s[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            q[i] = __ldg(qr + l4 + 4 * i);
            s[i] = dr[(l4 + 4 * i) >> 1];
        }
        stage_h(hs, hq, false);
        for (int i = threadIdx.x; i < K * (F / 16); i += blockDim.x) hds[i / (F / 16)][i % (F / 16)] = hd[i];
        __syncthreads();
        const int8_t* h = hs + t * HS;
#pragma unroll 1
        for (int b = 0; b < 2; ++b) {
            if (b == 1) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    q[i] = __ldg(qr + 36 + l4 + 4 * i);
                    s[i] = dr[(36 + l4 + 4 * i) >> 1];
                }
            }
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                const int c = 36 * b + l4 + 4 * i;
                const int4 xv = *reinterpret_cast<const int4*>(h + 16 * c);
                acc = fmaf((float)q8_isum(q[i], xv), __half2float(s[i]) * hds[t][c], acc);
            }
        }
    }
    acc *= sw;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        const float y = st0 * c0 + st1 * c1 + st2 * c2 + acc * c3;
        const float o = (acc + y) * rsqrtf(5.0f);
        state[n] = st1;
        state[D + n] = st2;
        state[2 * D + n] = acc;
        if (dbg) dbg[n] = o;
        x[n] = xr + o;
    }
}
