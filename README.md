# InklingEngine

A from-scratch inference engine for **Inkling 2 Mini** (Thinking Machines Lab) on NVIDIA GPUs: a Rust host, hand-written
CUDA kernels, and nothing else. It runs Unsloth's GGUFs directly, from UD-Q4_K_XL (16.4 GiB) down to UD-IQ1_S
(7.2 GiB), with every quantization format decoded natively. It serves an OpenAI-compatible API with reasoning, tool
calls and image input, and fits a 384k-token context in 24 GB. On smaller GPUs (8 GB and up) it keeps part of the
experts in system RAM and runs them on the CPU.

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
Logits are compared against the reference llama.cpp implementation on the same GGUF, for every supported file
(table under [Quantizations](#quantizations)). For UD-Q4_K_XL on an 873-token prompt with batched prefill:
- median KL divergence is 3e-5;
- top-1 agreement is 98.7%;
- the mean KL (0.0037) comes almost entirely from router near-ties that flip an expert choice.

Every weight format is also decoded three ways, by the CUDA kernels, the CPU path and gguf-py's reference
dequantizers, and the results are bit-identical (`inkling codectest` + `tools/codec_oracle.py`).

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

### Quantizations
All UD files of [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (the root folder,
not `Mainline_Support`) run, except UD-Q8_K_XL, which stores weights in BF16. Each file mixes several formats
(NVFP4, Q8_0, the K-quants, the IQ formats, MXFP4 and Unsloth's Q2_0). The engine decodes all of them in its own
kernels, on the GPU and in the CPU offload path. Measured on the RTX 3090:

| file | size | quality vs Q8_K_XL (Unsloth) | KL vs reference, decode / prefill | decode | prefill, 8k prompt |
|---|---|---|---|---|---|
| UD-Q4_K_XL | 16.4 GiB | KLD 0.0107, top-1 98.0% | 0.0039 / 0.0037 | 122 tok/s | 4,150 tok/s |
| UD-Q4_K_M | 15.5 GiB | 0.0108, 97.9% | 0.0039 / 0.0037 | 118 tok/s | 4,080 tok/s |
| UD-IQ4_XS | 14.7 GiB | 0.0109, 97.7% | 0.0038 / 0.0036 | 111 tok/s | 3,530 tok/s |
| UD-Q3_K_XL | 12.7 GiB | 0.0174, 96.3% | 0.0041 / 0.0029 | 123 tok/s | 3,730 tok/s |
| UD-IQ3_XXS | 10.9 GiB | 0.0269, 94.7% | 0.0042 / 0.0044 | 119 tok/s | 3,310 tok/s |
| UD-Q2_K_XL | 8.5 GiB | 0.0595, 91.6% | 0.0054 / 0.0047 | 127 tok/s | 3,370 tok/s |
| UD-IQ2_XXS | 7.9 GiB | 0.0726, 90.6% | 0.0054 / 0.0063 | 128 tok/s | 3,210 tok/s |
| UD-IQ1_M | 7.5 GiB | 0.0867, 90.2% | 0.0053 / 0.0053 | 132 tok/s | 3,400 tok/s |
| UD-IQ1_S | 7.2 GiB | 0.0911, 89.4% | 0.0046 / 0.0038 | 134 tok/s | 3,230 tok/s |

The quality column is Unsloth's measurement of each file against their largest one: it is the cost of the quant
itself. The KL column checks this engine against the reference llama.cpp on the same file: it should stay small for
every row, and it does. Q4_K_M and IQ4_XS are as good as Q4_K_XL and smaller; Q3_K_XL is the best trade-off for
16 GB cards; the 1- and 2-bit files lose noticeably more, and the three smallest fit a 12 GB card entirely.

### Smaller GPUs: CPU offload
When the model and context do not fit in free VRAM, `inkling` keeps the experts of some MoE layers in system RAM
and runs them on the CPU (AVX2, all cores). Every token reads only 8 of each layer's 128 experts, so a layer on the
CPU costs one read of ~34 MB (NVFP4 layers) or ~64 MB (the four Q8 layers) from RAM per visit. Layers are chosen
in this order:
- first the prelude and coda layers, which run once per token;
- the looped core layers last, because they run three times per token.

During prompt processing the offloaded experts stream to the GPU over PCIe in groups, overlapped with compute.

Decode speed in tok/s, measured on the RTX 3090 with its memory capped by `INKLING_VRAM_LIMIT` (Ryzen 9 5950X,
dual-channel DDR4-3200, 4k context). A real smaller card has a slower GPU as well, but most of the time goes to the
CPU part. In brackets: GiB of experts kept in system RAM.

| file | 24 GB | 16 GB | 12 GB | 10 GB | 8 GB |
|---|---|---|---|---|---|
| UD-Q4_K_XL | 122 | 56 (5.1) | 36 (8.8) | 24 (11.0) | 19 (13.1) |
| UD-Q4_K_M | 118 | 62 (3.6) | 37 (7.9) | 26 (9.5) | 21 (11.6) |
| UD-IQ4_XS | 111 | 45 (2.9) | 33 (6.7) | 24 (8.8) | 19 (10.9) |
| UD-Q3_K_XL | 123 | 86 (1.0) | 33 (5.1) | 24 (7.2) | 20 (8.8) |
| UD-IQ3_XXS | 119 | 127 | 43 (3.2) | 29 (5.1) | 16 (7.2) |
| UD-Q2_K_XL | 127 | 136 | 88 (0.8) | 40 (2.8) | 20 (4.9) |
| UD-IQ2_XXS | 128 | 133 | 135 | 47 (2.2) | 23 (4.4) |
| UD-IQ1_M | 132 | 137 | 139 | 49 (1.7) | 25 (3.7) |
| UD-IQ1_S | 134 | 139 | 142 | 50 (1.6) | 25 (3.6) |

Prompt processing with offload: about 3,000 tok/s on 16 GB and 1,600 tok/s on 8 GB for UD-Q4_K_XL (8k prompt).

With NVFP4 and Q8_0 experts, offloaded decode is bound by system RAM bandwidth (~38 GB/s here): faster RAM or more
channels help directly. The IQ formats read fewer bytes but take more CPU work to decode, so on 8 GB the small files
are only a little faster than UD-Q4_K_M. What they change is how much fits: a 12 GB card runs UD-IQ2_XXS, UD-IQ1_M and
UD-IQ1_S entirely on the GPU, and the small files need much less system RAM. The planner measures free VRAM *and* free system RAM before loading and refuses, with a message, when
the offloaded experts plus 2 GiB of headroom do not fit in RAM. With UD-Q4_K_XL an 8 GB card needs about 15 GB of
free RAM, in practice a 32 GB machine; with UD-IQ1_M about 6 GB. Loading itself stays light: each layer's file pages
are released once the layer is uploaded, so a load peaks around 2.5 GB of RAM plus the offloaded experts. `--offload off` disables offloading, `--offload N` forces N layers.

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
- **System RAM:** with offload, enough free RAM for the offloaded experts plus 2 GiB (bracketed numbers in the
  offload table; with UD-Q4_K_XL about 15 GB free for an 8 GB card, 11 GB for 12 GB, 7 GB for 16 GB).
- **Driver:** a recent NVIDIA driver with CUDA 12.8+ support. Only the driver is needed to run; the toolkit is
  needed only to build.
- **Model files:** from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth): any UD file in the root folder
  except UD-Q8_K_XL (see [Quantizations](#quantizations); UD-Q4_K_XL is the default choice, UD-Q3_K_XL for 16 GB
  cards) and, for images, `mmproj-F32.gguf`. All nine were checked against the reference llama.cpp.
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
- **Distribution:** needs glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, Fedora 35+, RHEL 9+). On older
  distributions, build from source.
- **What's inside:** the same files as the Windows zip, with `inkling` in place of `inkling.exe`.
- **WSL2 and CPU offload:**
  - WSL2 gives Linux only half of the PC's RAM by default, which limits how many layers can be offloaded. Raise it
    with `memory=` in `%USERPROFILE%\.wslconfig`.
  - Each handoff between GPU and CPU costs more under WSL2 than on Windows or native Linux. For example, with
    16 GB of VRAM: 40 tok/s under WSL2, 59 tok/s on Windows.

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

### Benchmark

`inkling bench` measures prompt processing and decode speed on your GPU, at the context depths you choose:

```sh
# decode speed on an empty context (under a minute, most of it loading the model)
inkling bench path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf

# prompt and decode speed at 8k and 32k tokens, comparing 4-bit and 6-bit KV caches
inkling bench path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf --fill 0,8k,32k --kv q4,q6

# the full 1M-token context
inkling bench path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf --ctx 1M --fill 0,128k,512k,1M --kv q4
```

The last one on the RTX 3090. The 9 GiB cache of a 1M context leaves too little VRAM for all the experts, so 7 MoE
layers run on the CPU:

```
note: --fill 1M is capped to 1048447 tokens so that the 128 decoded tokens fit in --ctx 1048576
Inkling-2-Mini-UD-Q4_K_XL, context 1M (1048576 tokens), 128 decoded tokens at each depth
CPU offload: 7 of 24 MoE layers run on the CPU (5.6 GiB of experts in system RAM: layers 5,18,19,20,21,22,23; 16 threads, AVX2)

KV q4: 9.0 KiB/token, 9.00 GiB at this context | loaded in 14.3 s, VRAM in use 23.0 / 24.0 GiB (NVIDIA GeForce RTX 3090)
    depth  prompt                                  decode
        0                                          52.7 tok/s
     128k  0→128k: 1960 tok/s in 66.9 s            46.9 tok/s
     512k  128k→512k: 883 tok/s in 445.2 s         41.2 tok/s
       1M  512k→1M: 451 tok/s in 1163.1 s          33.4 tok/s
  CPU experts: 1457 us per layer visit (3612 visits)
```

How it works:
- **`--fill`:** the depths to measure at (default `0`). They are filled in one pass. The prompt is prefilled to the
  first depth and `--gen` tokens are decoded there. The prefill then continues from the cache to the next depth.
  - Each row shows the prompt speed of the segment that ended at that depth and the decode speed there.
  - Prompt speed falls with depth because every new token attends to all the earlier ones. Filling 1M from empty
    takes about 28 minutes on the 3090.
- **`--ctx`:** the context to allocate. Default: just enough for the deepest fill.
  - The context decides how much VRAM the cache takes, and so whether experts move to the CPU. To measure the setup
    you will serve with, pass the same `--ctx` as `inkling serve`.
  - If the deepest fill plus `--gen` does not fit, that depth is shortened by a few tokens, as in the note above.
- **`--kv`:** the KV cache type (default `q6`), `K:V` for separate key and value types (`q8:q4`), or a comma list
  such as `q4,q6`. With a list, the model is reloaded for each entry and a table comparing their decode speeds closes
  the run.
- **`--gen N`:** tokens decoded at each depth (default 128, at most 2,000).
- **`--offload`:** as for `serve`.
- **Sizes:** `k` = 1,024 and `M` = 1,048,576 tokens, so `128k` = 131,072. Plain numbers work too.

The filler text is built into the binary. Speed does not depend on what the text says, so results are comparable
across machines.

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
| `--ctx N` | `131072` (all on the GPU of a 24 GB card: up to ~262144, ~393216 with `--kv q4`; beyond that, up to 1M, some experts move to the CPU) |
| `--kv q4\|q6\|q8\|f16`, `--kv-k`, `--kv-v` | `q6` (KV cache type, `K:V` such as `q8:q4` for separate types; also on `generate` and `bench`) |
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
| `INKLING_CPU_PIN=1` | pin CPU expert threads to physical cores (Windows; faster in isolation, slower in practice) |
| `INKLING_NO_PIN=1` | keep offloaded experts in ordinary instead of pinned memory |
| `INKLING_VRAM_LIMIT`, `INKLING_RAM_LIMIT` | plan as if only that many GiB were free (simulate a smaller machine) |
| `INKLING_TEMP`, `INKLING_TOP_P`, `INKLING_TOP_K`, `INKLING_MIN_P`, `INKLING_SEED` | sampling for `inkling generate` |
| `INKLING_REPEAT_PENALTY`, `INKLING_PRESENCE_PENALTY`, `INKLING_FREQUENCY_PENALTY`, `INKLING_REPEAT_LAST_N` | penalties for `inkling generate` |
| `INKLING_SKIP_VRAM_CHECK=1` | start even when the free-VRAM check fails (not recommended) |
| `INKLING_CUDA_ARCHS` | build time: target architectures (default `native`) |

Developer commands:
- **Benchmarks:** `bench` (see above), `attnbench` (decode attention on a synthetic cache).
- **Correctness:** `validate` against reference dumps, `kvstudy`, `cachetest`, `rendertest`, `gemmtest`, `visiontest`,
  `cpumoetest` (CPU experts against the GPU kernel), `codectest` (every weight format: CUDA vs CPU decoders; then
  `python tools/codec_oracle.py` compares with gguf-py).
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
- **One model:** Inkling 2 Mini only, in the UD GGUFs of Unsloth's root folder (not UD-Q8_K_XL, which uses BF16
  weights, and not the `Mainline_Support` layout).
- **Offload:** only the MoE experts move to the CPU; the rest of the model (about 3.3 GiB) and the KV cache stay in
  VRAM.
- **Hardware:** NVIDIA GPUs only. Native FP4 tensor cores on Blackwell are not used yet; NVFP4 is decoded in
  software everywhere.
- **Images:** the server does not fetch image URLs.

## Credits

- **Inkling 2 Mini** by [Thinking Machines Lab](https://thinkingmachines.ai) (Apache-2.0).
- **GGUF conversion and quantization** by [Unsloth](https://unsloth.ai).
- **Reference implementation:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), Unsloth's Inkling 2
  branch. It was used as the numerical reference and as the source of the model, preprocessing and quantization
  format semantics. The only material taken from it is the codebook tables of the IQ formats (generated from
  `ggml-common.h` by `tools/gen_codec_tables.py`); its license notice is reproduced in [NOTICE](NOTICE).
- **Chat template** (`tests/data/chat_template.jinja`, a test oracle) is the model's own, from the Inkling 2 Mini
  release (Apache-2.0).

## License

[MIT](LICENSE). Third-party notices (the
llama.cpp MIT notice, the Inkling 2 chat template used as a test oracle) are in [NOTICE](NOTICE).
