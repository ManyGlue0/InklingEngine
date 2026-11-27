// Token mask for constrained decoding (structured output, e.g. response_format json_schema).
//
// Runs right before ink_sample inside the decode graph. `on` and `mask` live in device memory so the captured graph
// serves every request: the host writes them before each step. When *on != 0, every logit whose bit is clear in
// `mask` (one bit per token, 32 tokens per word, token t at word t / 32, bit t % 32) becomes -inf, so the sampler
// can only pick allowed tokens. With *on == 0 the kernel returns at once.

extern "C" __global__ void __launch_bounds__(1024) ink_apply_mask(float* __restrict__ x, int n, const int* __restrict__ on,
                                                                  const unsigned int* __restrict__ mask) {
    if (*on == 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (((mask[i >> 5] >> (i & 31)) & 1u) == 0u) x[i] = -INFINITY;
    }
}
