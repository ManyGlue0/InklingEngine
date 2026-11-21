// Split-K decode attention (flash-decoding) for GQA 24 q heads / 8 kv heads, head_dim 128, f16 KV cache
// [kv_cap][8 * 128] (ring buffer for sliding-window layers). Included after decode.cu (warp_sum, warp_quant32).
//
// Grid (8 kv heads, NS splits), 128 threads. A split covers a contiguous position range sized from the current
// position at run time (the grid is fixed inside the CUDA graph), each warp walks 32-position tiles of it:
//   scores: 2 lanes per position (16-position tiles), lane parity picks the even or odd 16-byte chunks of the row.
//           The K tile is staged in shared memory with a 288-byte row stride, so the 16-byte reads of a quarter-warp
//           hit distinct bank groups; q of the 3 heads sharing this kv head is read as broadcasts. Relative-position
//           bias and the 1/sqrt(128) scale are added per position.
//   values: lane = 4 output dims; probabilities are broadcast with shuffles.
// Online softmax per warp, merged across warps, then across splits by the last block of each kv head (atomic
// counter, reset by that block), which adds the attention sink, applies the sigmoid output gate and writes the
// output and its q8_1 copy for the wo projection.

namespace att {
constexpr int HD = 128, NQ = 3, KVW = 1024, TILE = 16, KSTRIDE = 144;  // KSTRIDE in halves = 288 B
}

extern "C" __global__ void __launch_bounds__(128) ink_attn_split(const float* __restrict__ q, const float* __restrict__ r,
                                                                 const float* __restrict__ sink, const __half* __restrict__ Kc,
                                                                 const __half* __restrict__ Vc, const float* __restrict__ relbias,
                                                                 const float* __restrict__ gate, float* __restrict__ out,
                                                                 int8_t* __restrict__ oq, float* __restrict__ od, const int* __restrict__ step,
                                                                 int kv_cap, int window, int extent, float* __restrict__ part,
                                                                 unsigned int* __restrict__ counters, int qstride, int sstride) {
    using namespace att;
    __shared__ __align__(16) float qs[NQ][HD];
    __shared__ float rs[NQ][16];
    __shared__ __align__(16) __half ks[4][TILE * KSTRIDE];  // per-warp K tiles
    __shared__ __align__(16) __half vs[4][TILE * HD];       // per-warp V tiles
    __shared__ float wm[4][NQ], wl[4][NQ];
    float (*wo)[NQ][HD] = reinterpret_cast<float (*)[NQ][HD]>(&ks[0][0]);  // reuses the K tiles after the loop (6 KB)
    __shared__ bool last;

    const int kvh = blockIdx.x, split = blockIdx.y, NS = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // batched prefill: query token t = grid.z at position step[1] + t (its K/V are already in the cache);
    // q, r, gate advance by qstride per token, outputs by 24 heads, partials and counters per token
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
    part += (size_t)tq * gridDim.x * NS * (NQ * (HD + 2));
    const int j0 = window ? max(0, pos - window + 1) : 0;
    const int npos = pos - j0 + 1;
    // split span: at least 4 tiles (one per warp), multiple of the tile
    const int span = max(4 * TILE, ((npos + NS - 1) / NS + TILE - 1) / TILE * TILE);
    const int s0 = j0 + split * span, s1 = min(pos + 1, s0 + span);
    const int nsplit = min(NS, (npos + span - 1) / span);  // non-empty splits; the others exit at once
    if (split >= nsplit) return;

    for (int i = threadIdx.x; i < NQ * HD; i += blockDim.x) qs[i / HD][i % HD] = q[(kvh * NQ) * HD + i];
    if (threadIdx.x < NQ * 16) rs[threadIdx.x / 16][threadIdx.x % 16] = r[(kvh * NQ) * 16 + threadIdx.x];
    __syncthreads();

    float m[NQ], l[NQ], acc[NQ][4];
#pragma unroll
    for (int h = 0; h < NQ; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[h][d] = 0.0f;
    }
    __half* kt = ks[warp];
    __half* vt = vs[warp];
    const float scale = rsqrtf((float)HD);

    for (int t0 = s0 + warp * TILE; t0 < s1; t0 += 4 * TILE) {
        // stage the tile: 16 rows x 256 B of K and of V, 8 uint4 per lane each
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = lane + 32 * i, rr = idx >> 4, cc = idx & 15;
            const int j = t0 + rr;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (j < s1) {
                const size_t off = (size_t)(j % kv_cap) * KVW + kvh * HD;
                kv = *reinterpret_cast<const uint4*>(Kc + off + cc * 8);
                vv = *reinterpret_cast<const uint4*>(Vc + off + cc * 8);
            }
            *reinterpret_cast<uint4*>(kt + rr * KSTRIDE + cc * 8) = kv;
            *reinterpret_cast<uint4*>(vt + rr * HD + cc * 8) = vv;
        }
        // relative-position bias row of this lane's position, in flight with the tile
        const int pi = lane >> 1, par = lane & 1;
        const int j = t0 + pi;
        const bool valid = j < s1;
        const int dist = pos - j;
        const bool has_rb = valid && dist < extent;
        float4 rbv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) rbv[i] = has_rb ? __ldg(reinterpret_cast<const float4*>(relbias + (size_t)dist * 16) + i) : make_float4(0, 0, 0, 0);
        __syncwarp();
        // scores: lanes 2i, 2i+1 = position t0 + i (even / odd 16-byte chunks)
        float sc[NQ] = {0.0f, 0.0f, 0.0f};
#pragma unroll 4
        for (int k2 = 0; k2 < HD / 16; ++k2) {
            const int k = 2 * k2 + par;
            const uint4 kr = *reinterpret_cast<const uint4*>(kt + pi * KSTRIDE + k * 8);
            const __half2* kh2 = reinterpret_cast<const __half2*>(&kr);
            float kf[8];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float2 f = __half22float2(kh2[t]);
                kf[2 * t] = f.x;
                kf[2 * t + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < NQ; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&qs[h][k * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&qs[h][k * 8 + 4]);
                sc[h] += qa.x * kf[0] + qa.y * kf[1] + qa.z * kf[2] + qa.w * kf[3] + qb.x * kf[4] + qb.y * kf[5] + qb.z * kf[6] + qb.w * kf[7];
            }
        }
#pragma unroll
        for (int h = 0; h < NQ; ++h) sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], 1);
        float rb[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            rb[4 * i] = rbv[i].x; rb[4 * i + 1] = rbv[i].y; rb[4 * i + 2] = rbv[i].z; rb[4 * i + 3] = rbv[i].w;
        }
        float p[NQ];
#pragma unroll
        for (int h = 0; h < NQ; ++h) {
            float s = sc[h] * scale;
#pragma unroll
            for (int i = 0; i < 16; ++i) s = fmaf(rs[h][i], rb[i], s);  // rb = 0 beyond the extent
            s = valid ? s : -INFINITY;
            float tm = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
            const float mn = fmaxf(m[h], tm);  // tm is finite: the tile has at least one valid position
            const float corr = __expf(m[h] - mn);
            p[h] = __expf(s - mn);
            l[h] = l[h] * corr + (par ? 0.0f : p[h]);  // per-lane partial (even lanes), summed at the end
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[h][d] *= corr;
            m[h] = mn;
        }
        // values: lane owns dims 4*lane .. 4*lane+3; position jj's probability lives in lane 2*jj
#pragma unroll 4
        for (int jj = 0; jj < TILE; ++jj) {
            const uint2 vr = *reinterpret_cast<const uint2*>(vt + jj * HD + lane * 4);
            const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vr.x));
            const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&vr.y));
#pragma unroll
            for (int h = 0; h < NQ; ++h) {
                const float pj = __shfl_sync(0xffffffffu, p[h], 2 * jj);
                acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
            }
        }
        __syncwarp();
    }

    // merge the 4 warps -> block partial (m, l, o) per head (wo overlays the K tiles: wait for every warp)
    __syncthreads();
#pragma unroll
    for (int h = 0; h < NQ; ++h) {
        const float lsum = warp_sum(l[h]);
        if (lane == 0) { wm[warp][h] = m[h]; wl[warp][h] = lsum; }
#pragma unroll
        for (int d = 0; d < 4; ++d) wo[warp][h][lane * 4 + d] = acc[h][d];
    }
    __syncthreads();
    float* pm = part + ((size_t)kvh * NS + split) * (NQ * (HD + 2));  // [m, l] x NQ then o [NQ][HD]
    float bM[NQ], bL[NQ], bo[NQ];
    {
        const int dim = threadIdx.x;
#pragma unroll
        for (int h = 0; h < NQ; ++h) {
            float M = -INFINITY;
#pragma unroll
            for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w][h]);
            float L = 0.0f, o = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (wm[w][h] == -INFINITY) continue;
                const float e = __expf(wm[w][h] - M);
                L += wl[w][h] * e;
                o += wo[w][h][dim] * e;
            }
            bM[h] = M;
            bL[h] = L;
            bo[h] = o;
        }
    }
    auto finish = [&](int h, float M, float L, float o) {  // sink, normalization, gate, outputs (thread = dim)
        const int qh = kvh * NQ + h, dim = threadIdx.x;
        const float sk = sink[qh];
        const float Mx = fmaxf(M, sk);
        const float e = M == -INFINITY ? 0.0f : __expf(M - Mx);
        const float Lt = __expf(sk - Mx) + L * e;
        const float g = gate[qh * HD + dim];
        const float y = o * e / Lt * (1.0f / (1.0f + __expf(-g)));
        out[qh * HD + dim] = y;
        warp_quant32(y, oq + qh * HD + warp * 32, od + qh * 4 + warp);
    };
    if (nsplit == 1) {  // the only split: finish in place, no global round trip
#pragma unroll
        for (int h = 0; h < NQ; ++h) finish(h, bM[h], bL[h], bo[h]);
        return;
    }
#pragma unroll
    for (int h = 0; h < NQ; ++h) {
        pm[NQ * 2 + h * HD + threadIdx.x] = bo[h];
        if (threadIdx.x == 0) { pm[2 * h] = bM[h]; pm[2 * h + 1] = bL[h]; }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&counters[kvh], 1u) == (unsigned)nsplit - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();

    // last block of this kv head: merge the non-empty splits, add the sink, gate, write out + q8_1
    __shared__ float ml[64][NQ][2];  // NS <= 64
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
            const float e = ms == -INFINITY ? 0.0f : __expf(ms - M);
            L += ml[sp][h][1] * e;
            o += __ldcg(part + ((size_t)kvh * NS + sp) * (NQ * (HD + 2)) + NQ * 2 + h * HD + threadIdx.x) * e;
        }
        finish(h, M, L, o);
    }
    if (threadIdx.x == 0) counters[kvh] = 0;
}
