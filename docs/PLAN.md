# InklingEngine — plan

A from-scratch inference engine for **Inkling 2 Mini** (Thinking Machines Lab, Apache-2.0): Rust host and CUDA C++
kernels, specialized for this one architecture.

## Goals (v1)
1. **Single-stream decode** that beats Unsloth llama.cpp (48 tok/s on an RTX 3090). The target is 100+ tok/s.
2. **Prefill / long prompts** that beat 1350 tok/s at pp512.
3. **256k context in 24 GB**, with a KV format chosen by measurement (f16 / q8 / 6-bit / q4, compared by KLD).
4. **OpenAI-compatible server**: streaming chat completions, chat template, reasoning split, tool calling.
5. **Vision** via mmproj.
6. **Correctness**: numerical match against the reference (per-layer tensor dumps, final-logit KLD and top-1).

Runs on any NVIDIA GPU from sm_75 to sm_120 (Blackwell gets native FP4). Tuning happens on an RTX 3090 (sm_86).

## Baseline we must beat (RTX 3090, UD-Q4_K_XL, 32k ctx, q8_0 KV)
| | llama.cpp (Unsloth b12588) |
|---|---|
| decode | 47–50 tok/s |
| prefill pp512 | ~1350 tok/s |
| GPU time / token | ~18 ms: 19% Q8_0 matvec, 15% NVFP4 matvec, ~60% in ~7,400 tiny kernels |

## Model facts (from GGUF + reference source)
- d_model 2304, 24 weight layers, core layers 6–17 run 3 passes, so 48 layer passes per token.
- 8 full-attention passes (KV cache grows with context) and 40 sliding-window passes (1,536-cell window), plus short-conv state.
- Experts: 128, top-8, expert FFN 1152. NVFP4 in 20 layers (with per-expert F32 scales), Q8_0 in 4.
- Attention/out/head/embeddings in Q8_0. Small control weights (loop gates, conv kernels, clips) in F32/BF16.
- Depth attention ("attention residuals") across blocks, cross-pass loop control (gain/forget, rank 64).
- Vocab 201,024 (200,058 unpadded), final softcap 30.

## Architecture
```
crates/
  inkling-core     GGUF reader, weight repacking, model config, tokenizer, sampler
  inkling-cuda     CUDA FFI (driver API), memory pools, CUDA graph capture, kernel launch wrappers
  inkling-engine   forward pass (decode + prefill), KV/state management, scheduler
  inkling-server   OpenAI HTTP API (axum), chat template (minijinja), tool-call parsing, SSE streaming
  inkling-cli      run / chat / bench / validate
kernels/           .cu sources compiled by nvcc in build.rs to a fatbin (sm_75..sm_120 + PTX), embedded
tools/refdump/     small C++ tool linked against reference llama.cpp: dumps named tensors to .npy
tests/             numerical-match tests against reference dumps
```

## Performance strategy
**Decode (memory/launch bound):**
- One persistent buffer arena, and the whole token step captured as **one CUDA graph** (replayed per token).
- About 10 fused kernels per layer pass instead of about 150:
  mix+norm → (loop control) → fused QKV/R/gate/sink matvec over repacked concatenated weights → conv+clip+norm+tau
  epilogue → flash-decode with on-the-fly relative bias and sink → gate+out-proj → post-conv+residual →
  mix+control+norm → router+top-k → expert gate_up matvec with SwiGLU epilogue → expert down matvec with weighted
  reduce → post-conv+residual.
- Weights are repacked at load time into kernel-friendly layouts (no extra files, no quality change).
- Matvec kernels tuned to reach about 85% of DRAM bandwidth. Greedy sampling and top-k/top-p run on the GPU.
- Later options: overlap expert prefetch with attention; a persistent megakernel per layer.

**Prefill (compute bound):**
- INT8 tensor-core GEMM for Q8_0 weights (activations quantized per row). Grouped expert GEMM for NVFP4
  (dequant in the main loop; native FP4 MMA on sm_120).
- Chunked flash attention with banded relative bias.

**Long context:**
- KV only for the 8 full-attention passes (about 17 KB/token at 8-bit, so 256k ≈ 4.3 GB).
- Candidate KV formats are measured by KLD vs f16 KV at 32k/128k before picking a default.

## Correctness
1. Reference = Unsloth llama.cpp (`reference/`, git-ignored): prebuilt b12588 + source of branch
   `inkling2-loop-legacy-layout` (MIT). It's built from source with a tensor-dump tool.
2. Per-stage unit tests: each fused kernel vs a naive reference on random inputs.
3. Per-layer match: our intermediate tensors vs reference dumps (max abs / rel error).
4. End-to-end: final logits over a fixed prompt set, KLD and top-1 agreement vs reference. Perplexity on a text sample.

## Milestones
- **M0** Repo skeleton: cargo workspace, build.rs + nvcc fatbin, CUDA FFI smoke test, CI-ish `cargo test`.
- **M1** Reference harness: build reference llama.cpp + refdump tool, dump all stages for test prompts.
- **M2** GGUF loader + repack + tokenizer (exact match vs reference tokenization).
- **M3** Correct naive forward pass (simple kernels, decode only), matching reference logits.
- **M4** Fused decode kernels + CUDA graph. First speed milestone vs 48 tok/s.
- **M5** Prefill path (tensor-core GEMMs, chunked attention), long-context KV, KV-format study.
- **M6** OpenAI server, chat template, reasoning split, tool calling.
- **M7** Vision (mmproj).
- **M8** Portability pass (sm_75–sm_120), docs, release.
