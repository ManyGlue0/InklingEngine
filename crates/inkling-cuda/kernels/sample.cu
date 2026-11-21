// On-device token sampling for the autoregressive loop (one block of 1024 threads).
//
// Parameters come from device memory so a captured CUDA graph serves every request. With temperature <= 0 the
// sampler is greedy (lowest index among ties). Otherwise: candidates are the tokens within a probability ratio
// `floor` of the best one (raised until at most CAP remain; min_p raises it too), sorted by logit (ties by index,
// deterministic), truncated by top_k and top_p, and one is drawn with a counter-based RNG (seed, step).
// Writes the token as the next input (step[0]), appends it to hist and advances the position, like
// ink_argmax_next.

struct SampleParams {
    float temperature, top_p, min_p;
    int top_k;
    unsigned long long seed;
};

namespace smp {
constexpr int CAP = 1024, NT = 1024;

__device__ __forceinline__ unsigned long long splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
}  // namespace smp

extern "C" __global__ void __launch_bounds__(1024) ink_sample(const float* __restrict__ x, int n, const SampleParams* __restrict__ P,
                                                              int* __restrict__ step, int* __restrict__ hist) {
    using namespace smp;
    __shared__ float cl[CAP];
    __shared__ int ci[CAP];
    __shared__ float wv[32];
    __shared__ int wi[32];
    __shared__ int ncand;
    __shared__ float s_max;
    __shared__ int s_tok;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SampleParams p = *P;

    // max logit (and its index for the greedy path)
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = tid; i < n; i += NT) {
        const float t = x[i];
        if (t > v) { v = t; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) { wv[warp] = v; wi[warp] = idx; }
    __syncthreads();
    if (warp == 0) {
        v = wv[lane];
        idx = wi[lane];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { s_max = v; s_tok = idx; }
    }
    __syncthreads();
    const float M = s_max;

    if (p.temperature > 0.0f && p.top_k != 1) {
        const float T = p.temperature;
        // candidate floor (probability ratio to the best token), raised until <= CAP candidates
        float lfloor = logf(fmaxf(p.min_p, 1e-6f));
        for (int attempt = 0; attempt < 8; ++attempt) {
            if (tid == 0) ncand = 0;
            __syncthreads();
            const float cut = M + T * lfloor;
            int mine = 0;
            for (int i = tid; i < n; i += NT) mine += x[i] >= cut;
            for (int o = 16; o > 0; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
            if (lane == 0) atomicAdd(&ncand, mine);
            __syncthreads();
            const int c = ncand;
            __syncthreads();
            if (c <= CAP) break;
            lfloor = lfloor * 0.5f;  // e.g. ln 1e-6 -> ln 1e-3 -> ...
        }
        if (tid == 0) ncand = 0;
        __syncthreads();
        const float cut = M + T * lfloor;
        for (int i = tid; i < n; i += NT) {
            const float t = x[i];
            if (t >= cut) {
                const int slot = atomicAdd(&ncand, 1);
                if (slot < CAP) { cl[slot] = t; ci[slot] = i; }
            }
        }
        __syncthreads();
        const int c = min(ncand, CAP);
        for (int i = c + tid; i < CAP; i += NT) { cl[i] = -INFINITY; ci[i] = 0x7fffffff; }
        __syncthreads();
        // bitonic sort, descending by logit, ascending index on ties
        for (int k = 2; k <= CAP; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int i = tid, l = i ^ j;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const bool i_first = cl[i] > cl[l] || (cl[i] == cl[l] && ci[i] < ci[l]);
                    if (desc != i_first) {
                        const float tl = cl[i]; cl[i] = cl[l]; cl[l] = tl;
                        const int ti = ci[i]; ci[i] = ci[l]; ci[l] = ti;
                    }
                }
                __syncthreads();
            }
        }
        // probabilities over the kept candidates, top_k, top_p, draw (one warp, sequential prefix over <= 1024)
        if (warp == 0) {
            const int kmax = p.top_k > 0 ? min(c, p.top_k) : c;
            float total = 0.0f;
            for (int base = 0; base < kmax; base += 32) {
                const int i = base + lane;
                const float e = i < kmax ? __expf((cl[i] - M) / T) : 0.0f;
                if (i < kmax) cl[i] = e;
                total += e;
            }
            for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
            // top_p: smallest prefix with mass >= top_p * total
            const float target_p = (p.top_p > 0.0f && p.top_p < 1.0f ? p.top_p : 1.0f) * total;
            const unsigned long long h = splitmix(p.seed ^ splitmix((unsigned long long)step[1] + 0x51ED27ull));
            const float u01 = (float)(h >> 40) * (1.0f / 16777216.0f);
            int kept = kmax;
            float kept_mass = total;
            {
                float run = 0.0f;
                for (int base = 0; base < kmax; base += 32) {
                    const int i = base + lane;
                    float e = i < kmax ? cl[i] : 0.0f;
                    // inclusive warp scan
                    for (int o = 1; o < 32; o <<= 1) {
                        const float t = __shfl_up_sync(0xffffffffu, e, o);
                        if (lane >= o) e += t;
                    }
                    const float incl = run + e;
                    const unsigned m = __ballot_sync(0xffffffffu, i < kmax && incl >= target_p);
                    if (m) {
                        const int first = __ffs(m) - 1;
                        kept = base + first + 1;
                        kept_mass = __shfl_sync(0xffffffffu, incl, first);
                        break;
                    }
                    run = __shfl_sync(0xffffffffu, incl, 31);
                }
            }
            const float target = u01 * kept_mass;
            int tok = ci[0];
            {
                float run = 0.0f;
                for (int base = 0; base < kept; base += 32) {
                    const int i = base + lane;
                    float e = i < kept ? cl[i] : 0.0f;
                    for (int o = 1; o < 32; o <<= 1) {
                        const float t = __shfl_up_sync(0xffffffffu, e, o);
                        if (lane >= o) e += t;
                    }
                    const float incl = run + e;
                    const unsigned m = __ballot_sync(0xffffffffu, i < kept && incl > target);
                    if (m) {
                        tok = ci[base + __ffs(m) - 1];
                        break;
                    }
                    run = __shfl_sync(0xffffffffu, incl, 31);
                    if (base + 32 >= kept) tok = ci[kept - 1];  // rounding: take the last kept token
                }
            }
            if (lane == 0) s_tok = tok;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int t = s_tok;
        hist[step[2]] = t;
        step[2] += 1;
        step[0] = t;
        step[1] += 1;
    }
}
