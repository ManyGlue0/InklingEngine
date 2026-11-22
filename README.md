# InklingEngine

A from-scratch inference engine for **Inkling 2 Mini** (Thinking Machines Lab) on NVIDIA GPUs: a Rust host, hand-written
CUDA kernels, and nothing else. It runs the Unsloth GGUF directly (weights are repacked on load), serves an
OpenAI-compatible API with reasoning, tool calls and image input, and fits a 384k-token context in 24 GB. On smaller
GPUs (8 GB and up) it keeps part of the experts in system RAM and runs them on the CPU.

It is built for this one model. That focus is why it is fast: the looped 48-pass schedule, the depth attention,
the short convolutions and the NVFP4 experts each get fused, specialized kernels.

> InklingEngine is an independent inference engine optimized for Thinking Machines Lab's Inkling 2 Mini model. It is
> not affiliated with or endorsed by Thinking Machines Lab.

![inkling serve answering a streaming chat request](docs/screenshot.png)

## Performance

RTX 3090 (24 GB), `Inkling-2-Mini-UD-Q4_K_XL.gguf`, single stream, Windows 11:

| | llama.cpp (Unsloth b12588) | **InklingEngine** |
|---|---|---|
| Decode | 47–50 tok/s | **118–120 tok/s** (flat from 256 to 4k tokens of context) |
| Prefill, 512-token prompt | ~1,350 tok/s | **3,100 tok/s** |
| Prefill, 2k / 8k-token prompt | | **4,150–4,300 / 4,050 tok/s** |
| Context in 24 GB | 32k uses ~22 GB (q8_0 KV) | **262,144** (6-bit KV), **393,216** (4-bit KV) |
| Decode at 32k / 131k / 258k depth (6-bit KV) | | 106 / 88 / 75 tok/s |
| Decode at 323k depth (4-bit KV) | | 77 tok/s |
| Image encode, 1280×1280 (1,024 tokens) | | 116 ms |

On Linux the same card is faster: **135 tok/s** decode and **4,880 tok/s** at a 2k-token prompt (Ubuntu 24.04 under
WSL2). The Windows driver model adds overhead to every CUDA graph launch.

### How correctness is checked
Logits are compared against the reference llama.cpp implementation on the same GGUF. On an 807-token prompt with
batched prefill:
- median KL divergence is 6e-5;
- top-1 agreement is 97%;
- the mean KL (~0.018) comes almost entirely from router near-ties that flip an expert choice.

The vision encoder's projected embeddings match `llama-mtmd-debug` to about ±0.002, the difference between f16 and
F32 GEMMs.

### KV cache types
Only the 8 full-attention passes keep a cache that grows with the context; its format is selectable with `--kv`
(keys and values separately with `--kv-k` / `--kv-v`). The 40 sliding-window passes always keep small f16 rings.

| `--kv` | bytes per token | KL vs f16 cache (8k tokens) | top-1 agreement | context on 24 GB, all on GPU |
|---|---|---|---|---|
| `q4` | 9 KiB | 0.0090 | 96.5% | ~393k |
| `q6` (default) | 13 KiB | 0.0047 | 97.4% | ~262k |
| `q8` | 17 KiB | 0.0041 | 97.6% | ~200k |
| `f16` | 32 KiB | 0 | 100% | ~100k |

Measured with `inkling kvstudy` on 2,048 positions of real text. Any perturbation of the cache, even rounding the f16
values again, already gives KL ≈ 0.004 and ~97.8% top-1: the model amplifies tiny differences through its 48 passes.
So `q8` and `q6` are at that floor and `q4` costs a little. At 32k tokens the gaps shrink (q8 0.0028, q6 0.0034, q4
0.0061). Quantized caches are also faster to read at long context: decode attention runs on int8 tensor cores.

The forward pass is specified in [docs/MODEL.md](docs/MODEL.md).

### Smaller GPUs: CPU offload
When the model and context do not fit in free VRAM, `inkling` keeps the experts of some MoE layers in system RAM
and runs them on the CPU (AVX2, all cores). Every token reads only 8 of each layer's 128 experts, so a layer on the
CPU costs one read of ~34 MB (NVFP4 layers) or ~64 MB (the four Q8 layers) from RAM per visit. Layers are chosen
in this order:
- first the prelude and coda layers, which run once per token;
- the looped core layers last, because they run three times per token.

During prompt processing the offloaded experts stream to the GPU over PCIe in groups, overlapped with compute.

Measured on the RTX 3090 with its memory capped by `INKLING_VRAM_LIMIT` (Ryzen 9 5950X, dual-channel DDR4-3200).
A real smaller card has a slower GPU as well, but most of the time goes to the CPU part:

| VRAM | layers on the CPU | experts in RAM | decode | prefill, 8k prompt |
|---|---|---|---|---|
| 24 GB | 0 | 0 | 118 tok/s | 4,050 tok/s |
| 16 GB | 5–6 | 4.6–5.1 GiB | 59 tok/s | 3,000 tok/s |
| 12 GB | 13 | 8.8 GiB | 34 tok/s | |
| 8 GB | 20–21 | 12.6–13.1 GiB | 19 tok/s | 1,600 tok/s |

Decode speed with offload is set by system RAM bandwidth (~38 GB/s here); faster RAM or more channels help
directly. The planner measures free VRAM *and* free system RAM before loading and refuses, with a message, when
the offloaded experts plus 2 GiB of headroom do not fit in RAM. So an 8 GB card needs about 16 GB of free RAM,
which in practice means a 32 GB machine. `--offload off` disables offloading, `--offload N` forces N layers.

The KV cache always stays in VRAM, so on a small card the context competes with the experts. A smaller `--ctx` (the
server defaults to 131072) or `--kv q4` leaves room for more experts on the GPU and makes decode faster. The table
above used a 4k–16k context.
On a CPU without AVX2 a scalar fallback runs, much slower.

## Requirements

- **GPU:** NVIDIA, compute capability 7.5 or newer (Turing / RTX 20xx through Blackwell / RTX 50xx). With the whole
  model on the GPU (default 6-bit KV cache):

  | context | VRAM used by the engine |
  |---|---|
  | 8k | ~17.4 GiB |
  | 32k | ~17.7 GiB |
  | 128k | ~18.9 GiB |
  | 256k | ~20.6 GiB |
  | 384k (`--kv q4`) | ~20.7 GiB |
  | + vision (`--mmproj`) | +0.7 GiB (+ up to 0.4 GiB while encoding a large image) |

  A 24 GB card runs everything on the GPU. With less VRAM part of the experts move to system RAM automatically (see
  [CPU offload](#smaller-gpus-cpu-offload)); 8 GB is the practical minimum.
  At startup `inkling` checks the free VRAM (device-wide, as `nvidia-smi` reports it) and the free system RAM, and
  refuses to load, with a message, when the model and context do not fit, for example while another instance is
  running. Otherwise Windows would page the overflow into system RAM and stall the machine.
- **System RAM:** with offload, enough free RAM for the offloaded experts plus 2 GiB (about 16 GB free for an 8 GB
  card, 11 GB for 12 GB, 7 GB for 16 GB).
- **Driver:** a recent NVIDIA driver with CUDA 12.8+ support. Only the driver is needed to run; the toolkit is
  needed only to build.
- **Model files:** from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth): `Inkling-2-Mini-UD-Q4_K_XL.gguf`
  and, for images, `mmproj-F32.gguf`. Only UD-Q4_K_XL has been tested.
- **To build:** Rust 1.85 or newer, and the CUDA Toolkit 12.x (`nvcc` on PATH or `CUDA_PATH` set).

Tested with an RTX 3090 on Windows 11 and on Ubuntu 24.04 under WSL2: build, generation, server and image input.
Smaller cards were simulated by capping the 3090's memory. The sm_75 code paths were validated by running the
compute_75 PTX on the 3090; other GPUs have not been tested on real hardware yet.

## Quick start

InklingEngine is one program, `inkling`. Download it (Windows or Linux) or build it from source.

### Windows: download

1. Download `inkling-<version>-windows-x64.zip` from the
   [latest release](https://github.com/ManyGlue0/InklingEngine/releases/latest) and unzip it anywhere.
2. The only prerequisite is an NVIDIA driver; nothing else needs installing.
3. The zip contains `inkling.exe`, which holds the GPU code for every supported NVIDIA generation, plus this README,
   the license files and `tools/chat.py`.
4. `inkling.exe` is a command-line program: open a terminal in that folder (in Explorer, right-click > Open in
   Terminal) and run it as `.\inkling.exe`. Double-clicking it only flashes a window.

### Linux: download

Download `inkling-<version>-linux-x64.zip` from the same release page. It works on native Linux and under WSL2:

```sh
unzip inkling-<version>-linux-x64.zip
cd inkling-<version>-linux-x64
./inkling --version
```

- **Prerequisites:** only the NVIDIA driver (under WSL2, the Windows driver provides the GPU).
- **Distribution:** built on Ubuntu 24.04, so it needs glibc 2.39 or newer (Ubuntu 24.04+, Debian 13+, Fedora 40+).
  On older distributions, build from source.
- **What's inside:** the same files as the Windows zip, with `inkling` in place of `inkling.exe`.

### Any OS: build from source

**1. Install the prerequisites** (see [Requirements](#requirements)):
- an NVIDIA driver;
- the [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads): check with `nvcc --version`;
- [Rust](https://rustup.rs): check with `cargo --version`.

**2. Get the code and install the `inkling` command:**

```sh
git clone https://github.com/ManyGlue0/InklingEngine.git
cd InklingEngine
cargo install --locked --path crates/inkling-cli
```

This compiles everything, CUDA kernels included (a few minutes the first time), and copies `inkling` to Cargo's bin
folder: `~/.cargo/bin`, or `%USERPROFILE%\.cargo\bin` on Windows. Rustup puts that folder on your PATH, so open a new
terminal and check:

```sh
inkling --version
```

If you would rather not install it, run `cargo build --release` instead and call the binary by its path,
`target/release/inkling`.

### Then

**Download the model** from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth):
`Inkling-2-Mini-UD-Q4_K_XL.gguf`, plus `mmproj-F32.gguf` if you want image input.

**Run it** (from an unzipped release, type `.\inkling.exe` on Windows or `./inkling` on Linux instead of `inkling`):

```sh
# OpenAI-compatible server at http://127.0.0.1:8080/v1 (point any OpenAI client or chat UI at it)
inkling serve path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf --mmproj path/to/mmproj-F32.gguf --ctx 131072

# or a one-off generation in the terminal (greedy by default; see "Environment" for sampling)
inkling generate path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf "The three largest moons of Jupiter are" 64
```

Running `inkling` with no arguments lists every command.

### Build options

The kernels are compiled into the binary.
- **Default (`native`):** builds only for the GPUs in the build machine, which is fastest.
- **Portable:** a binary for every supported architecture, sm_75 to sm_120, plus PTX for future GPUs:

```sh
INKLING_CUDA_ARCHS=75,80,86,89,90,100,120 cargo install --locked --path crates/inkling-cli
```

On Windows PowerShell, set the variable first: `$env:INKLING_CUDA_ARCHS="75,80,86,89,90,100,120"`.
`nvcc` must be on PATH, or `CUDA_PATH` set (the CUDA installer does this).

## Server

Server flags:

| flag | default |
|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` |
| `--ctx N` | `131072` (max ~262144 on 24 GB, ~393216 with `--kv q4`) |
| `--kv q4\|q6\|q8\|f16`, `--kv-k`, `--kv-v` | `q6` (KV cache type; also on `generate` and the benchmarks) |
| `--offload auto\|off\|all\|N` | `auto` (MoE layers on the CPU only when VRAM is short) |
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

Try it from a terminal with the bundled client: `python tools/chat.py "Hello!"` (add `--image photo.jpg` for vision).
Smoke tests: `python tools/server_smoke.py` (tool-call round trip) and `python tools/vision_smoke.py` (image input).

### Environment

| variable | effect |
|---|---|
| `INKLING_KV`, `INKLING_KV_K`, `INKLING_KV_V` | KV cache type, as `--kv` / `--kv-k` / `--kv-v` |
| `INKLING_OFFLOAD` | as `--offload` |
| `INKLING_CPU_THREADS` | CPU expert threads (default: physical cores, at most 16) |
| `INKLING_VRAM_LIMIT`, `INKLING_RAM_LIMIT` | plan as if only that many GiB were free (simulate a smaller machine) |
| `INKLING_TEMP`, `INKLING_TOP_P`, `INKLING_TOP_K`, `INKLING_MIN_P`, `INKLING_SEED` | sampling for `inkling generate` |
| `INKLING_REPEAT_PENALTY`, `INKLING_PRESENCE_PENALTY`, `INKLING_FREQUENCY_PENALTY`, `INKLING_REPEAT_LAST_N` | penalties for `inkling generate` |
| `INKLING_SKIP_VRAM_CHECK=1` | start even when the free-VRAM check fails (not recommended) |
| `INKLING_CUDA_ARCHS` | build time: target architectures (default `native`) |

Developer commands:
- **Benchmarks:** `bench`, `benchpp`, `benchctx`.
- **Benchmarks:** `attnbench` (decode attention on a synthetic cache).
- **Correctness:** `validate` against reference dumps, `kvstudy`, `cachetest`, `rendertest`, `gemmtest`, `visiontest`,
  `cpumoetest` (CPU experts against the GPU kernel).
- **Diagnostics:** `vraminfo` prints total and usable VRAM.

## How it works

- **Decode.** One CUDA graph per token, with sampling on the GPU. The device runs the autoregressive loop and the
  host only reads tokens from mapped memory. Per pass there are about a dozen fused kernels:
  - depth-attention mix + norm + int8 quantization;
  - a multi-segment Q8 matvec on dp4a;
  - split-K flash-decoding on tensor cores (int8 scores against the quantized cache, f16 P·V), with the
    relative-position bias and the attention sink computed on the fly;
  - a persistent fused MoE kernel: NVFP4 gate_up → grid barrier with prefetched down weights → down + combine +
    short conv + residual.

  The kernels issue all of their loads up front, so small weights that miss L2 every token do not serialize on
  DRAM latency.
- **Prefill.** Chunks of 2,048 tokens on int8 tensor cores (Q8 weights, per-row-quantized activations), with grouped
  expert GEMMs and device-side routing. Flash attention runs on f16 tensor cores with banded relative bias.
- **Long context.** Only the 8 full-attention passes keep a full KV cache, 4-, 6- or 8-bit (72, 104 or 136 bytes per
  head per position, groups of 32 with an f16 scale) or f16. The 40 sliding-window passes keep f16 rings of window +
  chunk entries. The token embedding table stays in host memory (the GPU gathers one row per token over PCIe).
- **CPU offload.** Offloaded experts are repacked once into pinned RAM. During decode a kernel copies the router
  output into mapped memory, a CUDA host node runs the experts on a thread pool inside the token's graph, and the
  post conv reads the result back. During prefill the same pinned copy streams to the GPU in 16-expert groups on a
  second stream, double-buffered.
- **Vision.** Native ViT (23 layers) on f16 tensor cores: 2-D RoPE, flash attention, 2×2 patch merge and MLP
  projector. The embeddings replace the image placeholder positions during prefill.

[docs/MODEL.md](docs/MODEL.md) specifies the forward pass the kernels implement.

## Limitations

- **Single stream:** one request at a time, no batching across requests.
- **One model:** Inkling 2 Mini only.
- **Offload:** only the MoE experts move to the CPU; the rest of the model (about 3.3 GiB) and the KV cache stay in
  VRAM.
- **Hardware:** NVIDIA GPUs only. Native FP4 tensor cores on Blackwell are not used yet; NVFP4 is decoded in
  software everywhere.
- **Images:** the server does not fetch image URLs.

## Credits

- **Inkling 2 Mini** by [Thinking Machines Lab](https://thinkingmachines.ai) (Apache-2.0).
- **GGUF conversion and quantization** by [Unsloth](https://unsloth.ai).
- **Reference implementation:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), Unsloth's Inkling 2
  branch. It was used as the numerical reference and as the source of the model and preprocessing semantics. No
  llama.cpp code is included; its license notice is reproduced in [NOTICE](NOTICE).
- **Chat template** (`tests/data/chat_template.jinja`, a test oracle) is the model's own, from the Inkling 2 Mini
  release (Apache-2.0).

## License

[MIT](LICENSE). Third-party notices (the
llama.cpp MIT notice, the Inkling 2 chat template used as a test oracle) are in [NOTICE](NOTICE).
