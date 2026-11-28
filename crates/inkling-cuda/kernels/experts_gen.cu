// Generic decode expert kernels for natively stored formats (codecs.cuh), same interface as ink_moe_gate_up7 /
// ink_moe_down7: gate_up writes the int8 hidden hq [k][F] with scales per 16 hd [k][F/16]; down consumes it and
// applies the expert sum, post conv and residual. A layer may mix one of these with the v7 kernel for the other
// tensor (e.g. IQ gate_up + NVFP4 down). ExpW.qs points at the tensor's blocks, ExpW.fmt is the ggml type id.
#include "codecs.cuh"

// grid (F/16, k), 256 threads; warp w: features f, f+1 (f = 16 blockIdx.x + 2w); lane group g = lane/8 owns row
// {gate f, up f, gate f+1, up f+1}[g]; lane l8 takes groups u = l8 + 8i of the 72 groups of 32 in a row.
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_gate_up_gen(ExpW w, const int8_t* __restrict__ xq, const float* __restrict__ xd,
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
    const uint8_t* rp = w.qs + row * cdc::row_bytes(w.fmt, D);
    stage16(xs, xq, D, false);
    for (int i = threadIdx.x; i < D / 32; i += blockDim.x) xds[i] = xd[i];
    __syncthreads();
    float acc = 0.0f;
#pragma unroll 3
    for (int i = 0; i < D / 32 / 8; ++i) {
        const int u = l8 + 8 * i;
        const cdc::G32 q = cdc::g32(w.fmt, rp, u);
        const int4 a = *reinterpret_cast<const int4*>(xs + 32 * u), b = *reinterpret_cast<const int4*>(xs + 32 * u + 16);
        const int x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        acc += cdc::g32_dot(q, x, xds[u], xds[u]);
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

// grid D/8, 256 threads; warp w: output channel n = 8 blockIdx.x + w; lane group t = lane/4 is the t-th selected
// expert, lane l4 takes groups u = l4 + 4i of the 36 groups of 32 in a row.
extern "C" __global__ void __launch_bounds__(256, 4) ink_moe_down_gen(ExpW w, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                                                                      const int* __restrict__ sel, const float* __restrict__ wts,
                                                                      const float* __restrict__ cw, float* __restrict__ state,
                                                                      float* __restrict__ x, float* __restrict__ dbg) {
    using namespace ex7;
    constexpr int K = 8, HS = F + 64;
    __shared__ __align__(16) int8_t hs[K * HS];
    __shared__ float hds[K][F / 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane >> 2, l4 = lane & 3;
    const int n = blockIdx.x * 8 + warp;
    const int e = sel[t];
    const uint8_t* rp = w.qs + ((size_t)e * D + n) * cdc::row_bytes(w.fmt, F);
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0, st0 = 0, st1 = 0, st2 = 0, xr = 0;
    if (lane == 0) {
        const float4 cv = *reinterpret_cast<const float4*>(cw + (size_t)n * 4);
        c0 = cv.x; c1 = cv.y; c2 = cv.z; c3 = cv.w;
        st0 = state[n]; st1 = state[D + n]; st2 = state[2 * D + n];
        xr = x[n];
    }
    const float sw = wts[t] * (w.scale ? w.scale[e] : 1.0f);
    stage_h(hs, hq, false);
    for (int i = threadIdx.x; i < K * (F / 16); i += blockDim.x) hds[i / (F / 16)][i % (F / 16)] = hd[i];
    __syncthreads();
    const int8_t* h = hs + t * HS;
    float acc = 0.0f;
#pragma unroll 3
    for (int i = 0; i < F / 32 / 4; ++i) {
        const int u = l4 + 4 * i;
        const cdc::G32 q = cdc::g32(w.fmt, rp, u);
        const int4 a = *reinterpret_cast<const int4*>(h + 32 * u), b = *reinterpret_cast<const int4*>(h + 32 * u + 16);
        const int xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        acc += cdc::g32_dot(q, xv, hds[t][2 * u], hds[t][2 * u + 1]);
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

// ---------------------------------------------------------------------------------------------------------------
// Short chain (T <= 8 tokens: speculative verify, parallel streams) for layers whose expert tensors are not both
// NVFP4/Q8_0 (those have ink_moe8b): each tensor natively stored or repacked NVFP4/Q8_0 (w.fmt 1 / 0). Every expert of
// the union of the T routings is read once and applied to all tokens routed to it. Per token the arithmetic is exactly
// that of the decode kernels for the tensor's format (ink_moe_*_gen, or ink_moe_*7 for NVFP4/Q8_0: same values per
// lane, accumulation order and shuffle trees), so a batched row equals the same token decoded alone.
// Assignment a = 8 t + slot (k = 8 experts per token).
//   union_genb:   one block -> the union (int buffer, layout of genb::Union)
//   gate_up_genb: grid (F/16, 8T) -> hq [a][F], hd [a][F/16] (block y = union expert; blocks past the union exit)
//   down_genb:    grid (D/64, 8T) -> part [a][D][4]: the 4 lane partials of ink_moe_down_gen for each row
//   sum_genb:     grid (D/8)      -> weighted sum over the 8 experts (ink_moe_down_gen's 32-lane tree), post conv
//                                    and residual for a chain (state, or yin: inputs kept, state untouched) or for
//                                    parallel rows (rt: each row its own stream's state, cs floats apart)
namespace genb {
constexpr int K = 8, TMAX = 8, AMAX = K * TMAX;

// union of the routed experts in first-use order; uu[j][i] = the assignments using union expert j, ascending
struct Union {
    int n;
    int e[AMAX];
    int nu[AMAX];
    int uu[AMAX][TMAX];
};
}  // namespace genb

// One block of 64 threads, thread a = assignment a.
extern "C" __global__ void __launch_bounds__(64) ink_moe_union_genb(const int* __restrict__ sel, int T, genb::Union* __restrict__ un) {
    using namespace genb;
    __shared__ int se[AMAX], first[AMAX], uix[AMAX];
    const int a = threadIdx.x, na = K * T;
    se[a] = a < na ? sel[a] : -1 - a;
    __syncthreads();
    // first occurrence of my expert, and my rank among its users
    int f = a, rank = 0;
    for (int b = a - 1; b >= 0; --b) {
        if (se[b] == se[a]) {
            f = b;
            ++rank;
        }
    }
    first[a] = f;
    __syncthreads();
    // union index of a first occurrence = number of first occurrences before it
    if (a < na && f == a) {
        int j = 0;
        for (int b = 0; b < a; ++b) j += first[b] == b;
        uix[a] = j;
    }
    __syncthreads();
    if (a < na) {
        const int j = uix[f];
        un->uu[j][rank] = a;
        if (f == a) un->e[j] = se[a];
        // the last user of each expert writes the count
        bool last = true;
        for (int b = a + 1; b < na; ++b) last &= se[b] != se[a];
        if (last) un->nu[j] = rank + 1;
    }
    if (a == 0) {
        int n = 0;
        for (int b = 0; b < na; ++b) n += first[b] == b;
        un->n = n;
    }
}

extern "C" __global__ void __launch_bounds__(256) ink_moe_gate_up_genb(ExpW w, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                                       const genb::Union* __restrict__ un, int8_t* __restrict__ hq,
                                                                       float* __restrict__ hd, float beta, float lb) {
    using namespace ex7;
    using genb::TMAX;
    __shared__ __align__(16) int8_t xs[TMAX][D];
    __shared__ float xds[TMAX][D / 32];
    __shared__ float hv[TMAX][16];
    const int ui = blockIdx.y;
    if (ui >= un->n) return;
    const int e = un->e[ui], nu = un->nu[ui];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 3, l8 = lane & 7;
    const int f = blockIdx.x * 16 + 2 * warp + (g >> 1);
    const size_t row = (size_t)e * 2 * F + (g & 1) * F + f;
    const uint8_t* rp = w.fmt > 1 ? w.qs + row * cdc::row_bytes(w.fmt, D) : nullptr;
    for (int k = 0; k < nu; ++k) {
        const int t = un->uu[ui][k] >> 3;
        stage16(xs[k], xq + (size_t)t * D, D, w.fmt == 1);
        for (int i = threadIdx.x; i < D / 32; i += blockDim.x) xds[k][i] = xd[(size_t)t * (D / 32) + i];
    }
    __syncthreads();
    float acc[TMAX];
#pragma unroll
    for (int k = 0; k < TMAX; ++k) acc[k] = 0.0f;
    if (w.fmt == 1) {
        // NVFP4 as ink_moe_gate_up7: 9 uint4 per lane (u = l8 + 8i), x permuted for the E2M1 gather
        const uint4* qr = reinterpret_cast<const uint4*>(w.qs + row * (D / 2));
        const unsigned short* sr = reinterpret_cast<const unsigned short*>(w.sc + row * (D / 16));
        const int swz = (l8 >> 2) & 1;
#pragma unroll 3
        for (int i = 0; i < 9; ++i) {
            const int u = l8 + 8 * i;
            const uint4 q = __ldg(qr + u);
            const unsigned short sv = __ldg(sr + u);
            const float slo = ue4m3_fast(sv & 0xff), shi = ue4m3_fast(sv >> 8);
#pragma unroll
            for (int k = 0; k < TMAX; ++k) {
                if (k < nu) {
                    int4 xa, xb;
                    ldx32(xs[k] + 32 * u, swz, xa, xb);
                    int s0, s1;
                    fp4_isum(q, xa, xb, s0, s1);
                    acc[k] = fmaf(fmaf((float)s0, slo, (float)s1 * shi), xds[k][u], acc[k]);
                }
            }
        }
    } else if (w.fmt == 0) {
        // Q8_0 as ink_moe_gate_up7: 18 uint4 per lane in two batches of 9 (c = 72 b + l8 + 8i)
        const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(w.qs) + row * D);
        const __half* dr = reinterpret_cast<const __half*>(w.sc) + row * (D / 32);
#pragma unroll 1
        for (int b = 0; b < 2; ++b) {
#pragma unroll 3
            for (int i = 0; i < 9; ++i) {
                const int c = 72 * b + l8 + 8 * i;
                const uint4 q = __ldg(qr + c);
                const float sv = __half2float(dr[c >> 1]);
#pragma unroll
                for (int k = 0; k < TMAX; ++k) {
                    if (k < nu) {
                        const int4 x = *reinterpret_cast<const int4*>(xs[k] + 16 * c);
                        acc[k] = fmaf((float)q8_isum(q, x), sv * xds[k][c >> 1], acc[k]);
                    }
                }
            }
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < D / 32 / 8; ++i) {
            const int u = l8 + 8 * i;
            const cdc::G32 q = cdc::g32(w.fmt, rp, u);
#pragma unroll
            for (int k = 0; k < TMAX; ++k) {
                if (k < nu) {
                    const int4 a = *reinterpret_cast<const int4*>(xs[k] + 32 * u), b = *reinterpret_cast<const int4*>(xs[k] + 32 * u + 16);
                    const int x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                    acc[k] += cdc::g32_dot(q, x, xds[k][u], xds[k][u]);
                }
            }
        }
    }
    const float sc = w.scale ? w.scale[e] : 1.0f;
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
                hv[k][2 * warp + (lane >> 4)] = a * (lb * tanhf(up / lb));
            }
        }
    }
    __syncthreads();
    // warp k quantizes the 16 outputs of user k
    if (warp < nu) {
        const int a = un->uu[ui][warp];
        const float v = hv[warp][lane & 15];
        float amax = fabsf(v);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        if (lane < 16) hq[(size_t)a * F + blockIdx.x * 16 + lane] = amax == 0.0f ? 0 : (int8_t)roundf(v / d);
        if (lane == 0) hd[(size_t)a * (F / 16) + blockIdx.x] = d;
    }
}

// warp w: channels 64 blockIdx.x + 8 w + c (c = lane/4); lane l4 takes groups u = l4 + 4i of the 36 groups of 32 in
// a row (as in ink_moe_down_gen), decodes each once and applies it to every user of the union expert.
extern "C" __global__ void __launch_bounds__(256) ink_moe_down_genb(ExpW w, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                                                                    const genb::Union* __restrict__ un, float* __restrict__ part) {
    using namespace ex7;
    using genb::TMAX;
    __shared__ __align__(16) int8_t hs[TMAX][F];
    __shared__ float hds[TMAX][F / 16];
    const int ui = blockIdx.y;
    if (ui >= un->n) return;
    const int e = un->e[ui], nu = un->nu[ui];
    for (int k = 0; k < nu; ++k) {
        const int a = un->uu[ui][k];
        stage16(hs[k], hq + (size_t)a * F, F, w.fmt == 1);
        for (int i = threadIdx.x; i < F / 16; i += blockDim.x) hds[k][i] = hd[(size_t)a * (F / 16) + i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l4 = lane & 3;
    const int n = blockIdx.x * 64 + warp * 8 + (lane >> 2);
    const size_t row = (size_t)e * D + n;
    float acc[TMAX];
#pragma unroll
    for (int k = 0; k < TMAX; ++k) acc[k] = 0.0f;
    if (w.fmt == 1) {
        // NVFP4 as ink_moe_down7: 9 uint4 per lane (u = l4 + 4i), h permuted for the E2M1 gather
        const uint4* qr = reinterpret_cast<const uint4*>(w.qs + row * (F / 2));
        const unsigned short* sr = reinterpret_cast<const unsigned short*>(w.sc + row * (F / 16));
#pragma unroll 3
        for (int i = 0; i < 9; ++i) {
            const int u = l4 + 4 * i;
            const uint4 q = __ldg(qr + u);
            const unsigned short sv = __ldg(sr + u);
            const float slo = ue4m3_fast(sv & 0xff), shi = ue4m3_fast(sv >> 8);
#pragma unroll
            for (int k = 0; k < TMAX; ++k) {
                if (k < nu) {
                    int4 xa, xb;
                    ldx32(hs[k] + 32 * u, 0, xa, xb);
                    int s0, s1;
                    fp4_isum(q, xa, xb, s0, s1);
                    acc[k] = fmaf((float)s0, slo * hds[k][2 * u], acc[k]);
                    acc[k] = fmaf((float)s1, shi * hds[k][2 * u + 1], acc[k]);
                }
            }
        }
    } else if (w.fmt == 0) {
        // Q8_0 as ink_moe_down7: 18 uint4 per lane in two batches of 9 (c = 36 b + l4 + 4i)
        const uint4* qr = reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(w.qs) + row * F);
        const __half* dr = reinterpret_cast<const __half*>(w.sc) + row * (F / 32);
#pragma unroll 1
        for (int b = 0; b < 2; ++b) {
#pragma unroll 3
            for (int i = 0; i < 9; ++i) {
                const int c = 36 * b + l4 + 4 * i;
                const uint4 q = __ldg(qr + c);
                const float sv = __half2float(dr[c >> 1]);
#pragma unroll
                for (int k = 0; k < TMAX; ++k) {
                    if (k < nu) {
                        const int4 xv = *reinterpret_cast<const int4*>(hs[k] + 16 * c);
                        acc[k] = fmaf((float)q8_isum(q, xv), sv * hds[k][c], acc[k]);
                    }
                }
            }
        }
    } else {
        const uint8_t* rp = w.qs + row * cdc::row_bytes(w.fmt, F);
#pragma unroll 3
        for (int i = 0; i < F / 32 / 4; ++i) {
            const int u = l4 + 4 * i;
            const cdc::G32 q = cdc::g32(w.fmt, rp, u);
#pragma unroll
            for (int k = 0; k < TMAX; ++k) {
                if (k < nu) {
                    const int4 x0 = *reinterpret_cast<const int4*>(hs[k] + 32 * u), x1 = *reinterpret_cast<const int4*>(hs[k] + 32 * u + 16);
                    const int xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                    acc[k] += cdc::g32_dot(q, xv, hds[k][2 * u], hds[k][2 * u + 1]);
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < TMAX; ++k) {
        if (k < nu) part[((size_t)un->uu[ui][k] * D + n) * 4 + l4] = acc[k];
    }
}

extern "C" __global__ void __launch_bounds__(256) ink_moe_sum_genb(ExpW w, const float* __restrict__ part, const int* __restrict__ sel,
                                                                   const float* __restrict__ wts, int T, const float* __restrict__ cw,
                                                                   float* __restrict__ state, float* __restrict__ x, float* __restrict__ yin,
                                                                   const int* __restrict__ rt, int cs) {
    using namespace ex7;
    using genb::TMAX;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, slot = lane >> 2, l4 = lane & 3;
    const int n = blockIdx.x * 8 + warp;
    float y[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
        y[t] = 0.0f;
        if (t < T) {
            const int a = 8 * t + slot;
            const float sw = wts[a] * (w.scale ? w.scale[sel[a]] : 1.0f);
            float acc = part[((size_t)a * D + n) * 4 + l4];
            acc *= sw;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            y[t] = acc;
        }
    }
    if (lane != 0) return;
    const float4 cv = *reinterpret_cast<const float4*>(cw + (size_t)n * 4);
    const float c0 = cv.x, c1 = cv.y, c2 = cv.z, c3 = cv.w;
    if (rt) {
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < T) {
                float* st = state + (size_t)rt[2 * t] * cs;
                const float st0 = st[n], st1 = st[D + n], st2 = st[2 * D + n];
                const float acc = y[t];
                const float yc = st0 * c0 + st1 * c1 + st2 * c2 + acc * c3;
                const float o = (acc + yc) * rsqrtf(5.0f);
                st[n] = st1;
                st[D + n] = st2;
                st[2 * D + n] = acc;
                x[(size_t)t * D + n] += o;
            }
        }
        return;
    }
    float st0 = state[n], st1 = state[D + n], st2 = state[2 * D + n];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
        if (t < T) {
            const float acc = y[t];
            const float yc = st0 * c0 + st1 * c1 + st2 * c2 + acc * c3;
            const float o = (acc + yc) * rsqrtf(5.0f);
            x[(size_t)t * D + n] += o;
            if (yin) yin[(size_t)t * D + n] = acc;
            st0 = st1;
            st1 = st2;
            st2 = acc;
        }
    }
    if (!yin) {
        state[n] = st0;
        state[D + n] = st1;
        state[2 * D + n] = st2;
    }
}

// Test: dequantize `rows` rows of k values (format fmt, rows rb bytes apart) to f32, one block per row.
extern "C" __global__ void ink_dequant_rows(const uint8_t* __restrict__ w, int fmt, int k, float* __restrict__ out) {
    const uint8_t* rp = w + (size_t)blockIdx.x * cdc::row_bytes(fmt, k);
    for (int g = threadIdx.x; g < k / 32; g += blockDim.x) {
        const cdc::G32 q = cdc::g32(fmt, rp, g);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int v = (int)(int8_t)((q.q[i >> 2] >> (8 * (i & 3))) & 0xff);
            out[(size_t)blockIdx.x * k + 32 * g + i] = (i < 16 ? q.s0 : q.s1) * (float)v - q.m;
        }
    }
}
