// Batched Q8_0 matvec for short chains (T <= 8 tokens, speculative verify): every weight row is read once and dotted
// with the q8_1 activations of all T tokens (decode's ink_mv_q8q with a token dimension).

#include "common.cuh"

struct MVSegQB {
    const int8_t* qs;  // int8 [rows][K]
    const __half* d;   // [rows][K/32]
    const int8_t* xq;  // [T][K]
    const float* xd;   // [T][K/32]
    float* y;          // y[t * ldy + row]
    int n;
    int ldy;
};
struct MVQB4 {
    MVSegQB s[4];
    int nseg;
    int K;
    int T;
};

// Grid (ceil(rows / 8)), 256 threads: one warp per row.
INK_EXTERN void __launch_bounds__(256) ink_mv_q8q_b(MVQB4 a) {
    constexpr int TM = 8;
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    int seg = 0;
    while (seg < a.nseg && row >= a.s[seg].n) { row -= a.s[seg].n; ++seg; }
    if (seg >= a.nseg) return;
    const MVSegQB& s = a.s[seg];
    const int K = a.K, T = a.T, lane = threadIdx.x & 31;
    const int8_t* qs = s.qs + (size_t)row * K;
    const __half* d = s.d + (size_t)row * (K / 32);
    float acc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) acc[t] = 0.0f;
    for (int c = lane; c < K / 16; c += 32) {
        const int4 q = __ldg(reinterpret_cast<const int4*>(qs) + c);
        const float wd = __half2float(d[c >> 1]);
#pragma unroll
        for (int t = 0; t < TM; ++t) {
            if (t < T) {
                const int4 x = *reinterpret_cast<const int4*>(s.xq + (size_t)t * K + 16 * c);
                int is = __dp4a(q.x, x.x, 0);
                is = __dp4a(q.y, x.y, is);
                is = __dp4a(q.z, x.z, is);
                is = __dp4a(q.w, x.w, is);
                acc[t] = fmaf((float)is, wd * s.xd[(size_t)t * (K / 32) + (c >> 1)], acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        if (t < T) {
            const float v = warp_sum(acc[t]);
            if (lane == 0) s.y[(size_t)t * s.ldy + row] = v;
        }
    }
}
