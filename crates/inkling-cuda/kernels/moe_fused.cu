// MoE v8: gate_up + down + expert sum + post conv + residual in ONE persistent kernel (grid = all resident blocks).
// Included after experts.cu (uses namespace ex7).
//
// The down weights depend only on the routing (known at launch), not on h, so every warp issues its first down
// chunk before the grid barrier that waits for h: the second matmul starts streaming while the first drains, and the
// layer pays one ramp-up and one tail instead of two. Each warp walks its tasks in "chunks" (9 x 16 bytes per lane)
// with a two-deep register pipeline (load chunk j+1, compute chunk j). FP4 rows are one chunk, Q8 rows two.
// h goes through global memory as float; after the barrier every block quantizes all of it (8 x F, groups of 16).

namespace ex8 {
using namespace ex7;
constexpr int HS = F + 64;

// Two UE4M3 scales (low, high byte of s) -> floats, with the NVFP4 0.5 factor: 2^(e-8) (1 + m/8). Exact for normal
// scales only (e != 0): the loader checks that the model has no subnormal or zero scales.
__device__ __forceinline__ float2 ue4m3x2_normal(uint32_t s) {
    return make_float2(__int_as_float(((s << 20) & 0x07F00000u) + (119u << 23)), __int_as_float(((s << 12) & 0x07F00000u) + (119u << 23)));
}

// Activations for FP4 rows are staged with the two 16-byte halves of every 32-byte group u swapped when bit 2 of u
// is set: a lane reading group u then loads its first half at offset 16 * swz with swz = (u >> 2) & 1 constant per
// lane, and each quarter-warp still touches all 32 banks once (no per-element select needed).
__device__ __forceinline__ void stage_x_fp4(int8_t* dst, const int8_t* __restrict__ src, int n) {
    for (int i = threadIdx.x; i < n / 16; i += blockDim.x) {
        int4 v = reinterpret_cast<const int4*>(src)[i];
        v = make_int4(v.x, v.z, v.y, v.w);  // perm16
        const int u = i >> 1;
        reinterpret_cast<int4*>(dst)[(u << 1) | ((i & 1) ^ ((u >> 2) & 1))] = v;
    }
}

struct Chunk {
    uint4 q[9];
    uint32_t s[9];  // FP4: ushort holding two UE4M3 sub-block scales; Q8: half bits of the row scale
};

// Split sense-reversal grid barrier; bar[0] = arrivals, bar[1] = generation. Every block must be resident.
// arrive: publish this block's writes and count it (thread 0 returns the generation to wait on). The fence sits
// BEFORE the block issues its prefetch loads: a fence waits for the thread's outstanding memory operations.
__device__ __forceinline__ unsigned int grid_arrive(unsigned int* bar) {
    __syncthreads();
    unsigned int g = 0;
    if (threadIdx.x == 0) {
        g = *reinterpret_cast<volatile unsigned int*>(bar + 1);
        __threadfence();
        if (atomicAdd(bar, 1u) == gridDim.x - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        }
    }
    return g;
}
// wait: spin until the generation moves on (acquire load: no trailing fence to drain prefetches).
__device__ __forceinline__ void grid_wait(const unsigned int* bar, unsigned int g) {
    if (threadIdx.x == 0) {
        unsigned int cur;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(bar + 1) : "memory");
        } while (cur == g);
    }
    __syncthreads();
}
}  // namespace ex8

extern "C" __global__ void __launch_bounds__(256, 2) ink_moe8(ExpW wg, ExpW wd, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                              const int* __restrict__ sel, const float* __restrict__ wts, float* __restrict__ h,
                                                              const float* __restrict__ cw, float* __restrict__ state, float* __restrict__ x,
                                                              float* __restrict__ dbg, unsigned int* __restrict__ bar, float beta, float lb,
                                                              unsigned long long* __restrict__ tl) {
    using namespace ex8;
    // optional per-block phase timeline (globaltimer ns): start, staged, A done, barrier, h staged, end
    auto mark = [&](int k) {
        if (tl && threadIdx.x == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            tl[blockIdx.x * 8 + k] = t;
        }
    };
    mark(0);
    __shared__ __align__(16) int8_t xs[D];
    __shared__ float xds[D / 32];
    __shared__ __align__(16) int8_t hs[8 * HS];
    __shared__ __align__(8) float hds[8][F / 16];
    __shared__ int es[8];
    __shared__ float gsc[8], dsw[8];
    if (threadIdx.x < 8) {
        const int e = sel[threadIdx.x];
        es[threadIdx.x] = e;
        gsc[threadIdx.x] = wg.scale ? wg.scale[e] : 1.0f;
        dsw[threadIdx.x] = wts[threadIdx.x] * (wd.scale ? wd.scale[e] : 1.0f);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NWt = gridDim.x * 8, gw = blockIdx.x * 8 + warp;
    Chunk A, B;

    // ---------------- phase A: gate_up. Task = feature pair (f, f+1) of one expert: 4 rows, 8 lanes per row.
    {
        const bool fp4 = wg.fmt == 1;
        const int cp = fp4 ? 1 : 2;
        constexpr int NT = 8 * (F / 2);
        const int nch = (gw < NT ? (NT - 1 - gw) / NWt + 1 : 0) * cp;
        const int g = lane >> 3, l8 = lane & 7, swz = (l8 >> 2) & 1;
        auto load = [&](Chunk& c, int j) {
            const int task = gw + (j / cp) * NWt, half = j % cp;
            const int t = task / (F / 2), f = 2 * (task % (F / 2)) + (g >> 1);
            const size_t row = (size_t)es[t] * 2 * F + (g & 1) * F + f;
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
        float acc = 0.0f;
        auto compute = [&](const Chunk& c, int j) {
            const int half = j % cp;
            if (fp4) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int u = l8 + 8 * i;
                    const int4* xp = reinterpret_cast<const int4*>(xs + 32 * u);
                    const int4 xa = xp[swz], xb = xp[swz ^ 1];
                    int s0, s1;
                    fp4_isum(c.q[i], xa, xb, s0, s1);
                    const float2 sc = ue4m3x2_normal(c.s[i]);
                    acc = fmaf(fmaf((float)s0, sc.x, (float)s1 * sc.y), xds[u], acc);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int cc = 72 * half + l8 + 8 * i;
                    const int4 xv = *reinterpret_cast<const int4*>(xs + 16 * cc);
                    acc = fmaf((float)q8_isum(c.q[i], xv), __half2float(__ushort_as_half((unsigned short)c.s[i])) * xds[cc >> 1], acc);
                }
            }
            if (half == cp - 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, 4);
                acc += __shfl_xor_sync(0xffffffffu, acc, 2);
                acc += __shfl_xor_sync(0xffffffffu, acc, 1);
                const int task = gw + (j / cp) * NWt, t = task / (F / 2);
                const float sc = gsc[t];
                const float up = __shfl_sync(0xffffffffu, acc, (lane & 16) | 8) * sc;  // lane 0 <- 8, lane 16 <- 24
                if ((lane & 15) == 0) {
                    const float gt = acc * sc;
                    const float a = beta * tanhf(gt / beta) * (1.0f / (1.0f + __expf(-gt)));
                    h[t * F + 2 * (task % (F / 2)) + (lane >> 4)] = a * (lb * tanhf(up / lb));
                }
                acc = 0.0f;
            }
        };
        if (nch > 0) load(A, 0);
        if (fp4) stage_x_fp4(xs, xq, D);
        else stage16(xs, xq, D, false);
        for (int i = threadIdx.x; i < D / 32; i += blockDim.x) xds[i] = xd[i];
        __syncthreads();
        mark(1);
        for (int j = 0; j < nch; j += 2) {
            if (j + 1 < nch) load(B, j + 1);
            compute(A, j);
            if (j + 2 < nch) load(A, j + 2);
            if (j + 1 < nch) compute(B, j + 1);
        }
    }

    // ---------------- phase B: down. Task = output channel n: 8 rows (one per selected expert), 4 lanes per row.
    {
        const bool fp4 = wd.fmt == 1;
        const int cp = fp4 ? 1 : 2;
        const int nch = (gw < D ? (D - 1 - gw) / NWt + 1 : 0) * cp;
        const int t = lane >> 2, l4 = lane & 3, swz = t & 1;
        const int e = es[t];
        const float sw = dsw[t];
        auto load = [&](Chunk& c, int j) {
            const int n = gw + (j / cp) * NWt, half = j % cp;
            const size_t row = (size_t)e * D + n;
            if (fp4) {
                const uint4* qr = reinterpret_cast<const uint4*>(wd.qs + row * (F / 2));
                const unsigned short* sr = reinterpret_cast<const unsigned short*>(wd.sc + row * (F / 16));
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    c.q[i] = __ldg(qr + l4 + 4 * i);
                    c.s[i] = __ldg(sr + l4 + 4 * i);
                }
            } else {
                const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(wd.qs) + row * F) + 36 * half;
                const unsigned short* dr = reinterpret_cast<const unsigned short*>(wd.sc) + row * (F / 32);
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    c.q[i] = __ldg(qr + l4 + 4 * i);
                    c.s[i] = __ldg(dr + ((36 * half + l4 + 4 * i) >> 1));
                }
            }
        };
        float acc = 0.0f;
        const int8_t* hr = hs + t * HS;
        // conv/residual operands of this warp's first 4 channels, fetched before the barrier: lane 8q + k holds
        // operand k of task q (k < 4: conv taps, 4..6: state, 7: residual x). A warp has D / NWt channels (rounded
        // up): 2 with 144 blocks, 3 with 96 (66 SMs at 2 blocks each). Tasks past the 4th read theirs in compute.
        float opnd = 0.0f;
        {
            const int q = lane >> 3, k = lane & 7, n = gw + q * NWt;
            if (n < D) opnd = k < 4 ? cw[(size_t)n * 4 + k] : k < 7 ? state[(k - 4) * D + n] : x[n];
        }
        auto compute = [&](const Chunk& c, int j) {
            const int half = j % cp;
            if (fp4) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int u = l4 + 4 * i;
                    const int4* xp = reinterpret_cast<const int4*>(hr + 32 * u);
                    const int4 xa = xp[swz], xb = xp[swz ^ 1];
                    int s0, s1;
                    fp4_isum(c.q[i], xa, xb, s0, s1);
                    const float2 sc = ue4m3x2_normal(c.s[i]);
                    const float2 hd2 = *reinterpret_cast<const float2*>(&hds[t][2 * u]);
                    acc = fmaf((float)s0, sc.x * hd2.x, fmaf((float)s1, sc.y * hd2.y, acc));
                }
            } else {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const int cc = 36 * half + l4 + 4 * i;
                    const int4 xv = *reinterpret_cast<const int4*>(hr + 16 * cc);
                    acc = fmaf((float)q8_isum(c.q[i], xv), __half2float(__ushort_as_half((unsigned short)c.s[i])) * hds[t][cc], acc);
                }
            }
            if (half == cp - 1) {
                float y = acc * sw;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) y += __shfl_xor_sync(0xffffffffu, y, o);
                const int task = j / cp;  // warp-uniform
                const int n = gw + task * NWt;
                float op[8];
                if (task < 4) {
#pragma unroll
                    for (int k = 0; k < 8; ++k) op[k] = __shfl_sync(0xffffffffu, opnd, 8 * task + k);
                } else {
#pragma unroll
                    for (int k = 0; k < 8; ++k) op[k] = k < 4 ? cw[(size_t)n * 4 + k] : k < 7 ? state[(k - 4) * D + n] : x[n];
                }
                if (lane == 0) {
                    const float s1 = op[5], s2 = op[6];
                    const float o = (y + op[4] * op[0] + s1 * op[1] + s2 * op[2] + y * op[3]) * rsqrtf(5.0f);
                    state[n] = s1;
                    state[D + n] = s2;
                    state[2 * D + n] = y;
                    if (dbg) dbg[n] = o;
                    x[n] = op[7] + o;
                }
                acc = 0.0f;
            }
        };
        const unsigned int gen = grid_arrive(bar);
        mark(2);
        // the warp's first two down chunks stay in flight across the barrier wait
        if (nch > 0) load(A, 0);
        if (nch > 1) load(B, 1);
        grid_wait(bar, gen);
        mark(3);
        // quantize h (8 x F floats written by all blocks) into shared int8, groups of 16
        for (int gi = threadIdx.x; gi < 8 * (F / 16); gi += blockDim.x) {
            const int te = gi / (F / 16), k = gi - te * (F / 16);
            const float4* src = reinterpret_cast<const float4*>(h + te * F + 16 * k);
            float v[16];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 f4 = __ldcg(src + q);
                v[4 * q] = f4.x;
                v[4 * q + 1] = f4.y;
                v[4 * q + 2] = f4.z;
                v[4 * q + 3] = f4.w;
            }
            float amax = 0.0f;
#pragma unroll
            for (int q = 0; q < 16; ++q) amax = fmaxf(amax, fabsf(v[q]));
            const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
            int w4[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                w4[q] = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b) w4[q] |= (__float2int_rn(v[4 * q + b] * id) & 0xff) << (8 * b);
            }
            // FP4: odd experts (swz = 1) keep the two halves of each 32-byte group swapped, as for x
            reinterpret_cast<int4*>(hs + te * HS)[fp4 ? k ^ (te & 1) : k] = fp4 ? make_int4(w4[0], w4[2], w4[1], w4[3]) : make_int4(w4[0], w4[1], w4[2], w4[3]);
            hds[te][k] = amax / 127.0f;
        }
        __syncthreads();
        mark(4);
        for (int j = 0; j < nch; j += 2) {
            if (j > 0 && j + 1 < nch) load(B, j + 1);
            compute(A, j);
            if (j + 2 < nch) load(A, j + 2);
            if (j + 1 < nch) compute(B, j + 1);
        }
    }
    __syncthreads();
    mark(5);
}
