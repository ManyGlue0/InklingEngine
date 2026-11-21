# InklingEngine

A from-scratch inference engine for **Inkling 2 Mini** (Thinking Machines Lab) on NVIDIA GPUs: a Rust host, hand-written
CUDA kernels, and nothing else. It runs the Unsloth GGUF directly (weights are repacked on load), serves an
OpenAI-compatible API with reasoning, tool calls and image input, and fits a 256k-token context in 24 GB.

It is built for this one model. That focus is why it is fast: the looped 48-pass schedule, the depth attention,
the short convolutions and the NVFP4 experts each get fused, specialized kernels.

## Performance

RTX 3090 (24 GB), `Inkling-2-Mini-UD-Q4_K_XL.gguf`, single stream:

| | llama.cpp (Unsloth b12588) | **InklingEngine** |
|---|---|---|
| Decode | 47–50 tok/s | **113–117 tok/s** (flat from 256 to 4k tokens of context) |
| Prefill, 512-token prompt | ~1,350 tok/s | **3,100 tok/s** |
| Prefill, 2k / 8k-token prompt | | **4,350 / 4,190 tok/s** |
| Context in 24 GB | 32k uses ~22 GB (q8_0 KV) | **262,144** (6-bit KV) |
| Decode at 32k / 131k / 262k depth | | 97 / 68 / 45 tok/s |
| Image encode, 1280×1280 (1,024 tokens) | | 116 ms |

### How correctness is checked
Logits are compared against the reference llama.cpp implementation on the same GGUF. On an 807-token prompt with
batched prefill:
- median KL divergence is 6e-5;
- top-1 agreement is 97%;
- the mean KL (~0.018) comes almost entirely from router near-ties that flip an expert choice.

The vision encoder's projected embeddings match `llama-mtmd-debug` to about ±0.002, the difference between f16 and
F32 GEMMs.

### KV cache format
The KV format was chosen by measurement, not assumed (`inkling kvstudy`). 6-bit K/V on the 8 full-attention passes
costs only +0.0006 mean KL over 8-bit, while the sliding-window passes keep f16. See [docs/PLAN.md](docs/PLAN.md)
and [docs/MODEL.md](docs/MODEL.md).

## Requirements

- **GPU:** NVIDIA, compute capability 7.5 or newer (Turing / RTX 20xx through Blackwell / RTX 50xx), with enough
  memory for the weights plus the context:

  | context | VRAM used by the engine |
  |---|---|
  | 8k | ~18.3 GiB |
  | 32k | ~18.6 GiB |
  | 128k | ~19.8 GiB |
  | 256k | ~21.4 GiB |
  | + vision (`--mmproj`) | +0.7 GiB (+ up to 0.4 GiB while encoding a large image) |

  A 24 GB card runs the full 256k context; a 20 GB card should fit about 64k (untested). There is no CPU offload.
- **Driver:** a recent NVIDIA driver with CUDA 12.8+ support. Only the driver is needed to run; the toolkit is
  needed only to build.
- **Model files:** from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth): `Inkling-2-Mini-UD-Q4_K_XL.gguf`
  and, for images, `mmproj-F32.gguf`. Only UD-Q4_K_XL has been tested.
- **To build:** Rust 1.85 or newer, and the CUDA Toolkit 12.x (`nvcc` on PATH or `CUDA_PATH` set).

Tested on Windows 11 with an RTX 3090. The code has no Windows-specific parts, but Linux builds have not been tested
yet. The sm_75 code paths were validated by running the compute_75 PTX on the 3090.

## Build

```sh
cargo build --release                      # kernels for the GPUs in this machine only (fast)
# portable binary for every supported architecture (sm_75..sm_120 + PTX for future GPUs):
INKLING_CUDA_ARCHS=75,80,86,89,90,100,120 cargo build --release
```

The binary is `target/release/inkling` (`inkling.exe` on Windows). The kernels are embedded in it.

## Run

```sh
# OpenAI-compatible server (http://127.0.0.1:8080/v1)
inkling serve Inkling-2-Mini-UD-Q4_K_XL.gguf --mmproj mmproj-F32.gguf --ctx 131072

# one-off generation (greedy by default; see "Environment" below for sampling)
inkling generate Inkling-2-Mini-UD-Q4_K_XL.gguf "The three largest moons of Jupiter are" 64
```

Server flags:

| flag | default |
|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` |
| `--ctx N` | `131072` (max 262144 on 24 GB) |
| `--mmproj FILE` | none (enables image input) |
| `--api-key KEY` (or `INKLING_API_KEY`) | none |
| `--model-id NAME` | the file name |
| `--temp`, `--top-p`, `--top-k`, `--min-p` | `0.8`, `0.95`, `40`, `0.05` |
| `--repeat-penalty`, `--repeat-last-n` | `1.0`, `64` |

### API

- `POST /v1/chat/completions`, streaming (SSE) or not. It renders the model's own chat template natively (output
  identical to its Jinja template) and returns:
  - the reasoning in `reasoning_content` and the answer in `content`;
  - tool calls in `tool_calls`, with `finish_reason: "tool_calls"`.
- **Request fields:**
  - `tools` and `tool_choice: "none"`;
  - `reasoning_effort` (`none`/`low`/`medium`/`high`/`max` or a number in 0–0.99; default 0.9);
  - `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `stream_options.include_usage`;
  - sampling: `temperature`, `top_p`, `top_k`, `min_p`;
  - penalties: `presence_penalty`, `frequency_penalty`, `repeat_penalty` (or `repetition_penalty`), `repeat_last_n`.
- **Images:** content parts of type `image_url`, `input_image` or `image`, carrying a `data:` URL or bare base64
  (PNG, JPEG, WebP, GIF, BMP). Remote URLs are not fetched.
  - An image becomes up to 4,096 tokens: one per 40×40 pixels.
  - Larger images are downscaled with the aspect ratio kept.
- `POST /v1/completions`: raw text completion; `prompt` is a string or an array of token ids.
- `GET /v1/models`, `GET /health`.
- **Prompt caching:** a request that extends the previous conversation reuses its state, including images, and the
  reused count is reported in `usage.prompt_tokens_details.cached_tokens`.
- **CORS** is open, so browser UIs work. Requests are served one at a time, in order.

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "reasoning_effort": "low", "stream": true }'
```

Smoke tests: `python tools/server_smoke.py` (tool-call round trip) and `python tools/vision_smoke.py` (image input).

### Environment

| variable | effect |
|---|---|
| `INKLING_KV=f16` | f16 KV cache on the full-attention passes instead of 6-bit (~2.5× the KV memory) |
| `INKLING_TEMP`, `INKLING_TOP_P`, `INKLING_TOP_K`, `INKLING_MIN_P`, `INKLING_SEED` | sampling for `inkling generate` |
| `INKLING_REPEAT_PENALTY`, `INKLING_PRESENCE_PENALTY`, `INKLING_FREQUENCY_PENALTY`, `INKLING_REPEAT_LAST_N` | penalties for `inkling generate` |
| `INKLING_CUDA_ARCHS` | build time: target architectures (default `native`) |

Developer commands:
- **Benchmarks:** `bench`, `benchpp`, `benchctx`.
- **Correctness:** `validate` against reference dumps, `kvstudy`, `cachetest`, `rendertest`, `gemmtest`, `visiontest`.

## How it works

- **Decode.** One CUDA graph per token, with sampling on the GPU. The device runs the autoregressive loop and the
  host only reads tokens from mapped memory. Per pass there are about a dozen fused kernels:
  - depth-attention mix + norm + int8 quantization;
  - a multi-segment Q8 matvec on dp4a;
  - split-K flash-decoding with the relative-position bias and the attention sink computed on the fly;
  - a persistent fused MoE kernel: NVFP4 gate_up → grid barrier with prefetched down weights → down + combine +
    short conv + residual.

  The kernels issue all of their loads up front, so small weights that miss L2 every token do not serialize on
  DRAM latency.
- **Prefill.** Chunks of 2,048 tokens on int8 tensor cores (Q8 weights, per-row-quantized activations), with grouped
  expert GEMMs and device-side routing. Flash attention runs on f16 tensor cores with banded relative bias.
- **Long context.** Only the 8 full-attention passes keep a full KV cache, in 6-bit: 104 bytes per head per position.
  The 40 sliding-window passes keep f16 rings of window + chunk entries.
- **Vision.** Native ViT (23 layers) on f16 tensor cores: 2-D RoPE, flash attention, 2×2 patch merge and MLP
  projector. The embeddings replace the image placeholder positions during prefill.

[docs/MODEL.md](docs/MODEL.md) specifies the forward pass the kernels implement.

## Limitations

- **Single stream:** one request at a time, no batching across requests.
- **One model:** Inkling 2 Mini only, and the whole model must fit in VRAM.
- **Hardware:** NVIDIA GPUs only. Native FP4 tensor cores on Blackwell are not used yet; NVFP4 is decoded in
  software everywhere.
- **Images:** the server does not fetch image URLs.

## Credits

- **Inkling 2 Mini** by [Thinking Machines Lab](https://thinkingmachines.ai) (Apache-2.0).
- **GGUF conversion and quantization** by [Unsloth](https://unsloth.ai).
- **Reference implementation:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), Unsloth's Inkling 2
  branch. It was used as the numerical reference and as the source of the model and preprocessing semantics. No
  llama.cpp code is included.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
