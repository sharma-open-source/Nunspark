# NunSpark

[![tests](https://github.com/sharma-open-source/Nunspark/actions/workflows/tests.yml/badge.svg)](https://github.com/sharma-open-source/Nunspark/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/nunspark)](https://pypi.org/project/nunspark/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)](https://github.com/ml-explore/mlx)
[![Built on MLX](https://img.shields.io/badge/built%20on-MLX-orange)](https://github.com/ml-explore/mlx)

**Run LLMs that don't fit in your Mac's RAM — at usable speeds.**

Your Mac has 16 GB of unified memory. The model you want needs 16–40 GB of weights. NunSpark
runs it anyway: it packs the model into per-layer (and, for MoE models, per-*expert*) pieces
on disk and streams exactly the weights each token needs through a small resident budget —
then claws the speed back with three levers that only make sense in the disk-bound regime:

- **MoE expert streaming** — a Qwen3-30B-A3B token fires only 8 of 128 experts per layer.
  NunSpark loads *only those*, caches them expert-aware, and prefetches the ones the next
  verify pass will probably fire. Measured on a 16 GB M4: **0.5 → 3.2 tok/s**, reading
  ~100 MB/token instead of ~1 GB.
- **Deep-K speculative decoding** — on disk-bound hardware the draft model's compute is free,
  so speculation can go far deeper (K=16–24) than GPU serving ever would. One weight sweep
  buys up to 7 accepted tokens. Lossless: every emitted token is the target's own argmax
  (deterministic per config); on large real models fp16 numerics under different verify-pass
  shapes can occasionally flip an argmax near-tie (~1/100 tokens, self-healing) versus running
  single-token greedy, so identity is target-verified, not guaranteed byte-for-byte.
- **Byte-budgeted streaming engine** — every forward pass verified bit-identical to full-load
  mlx-lm, fp16 and 4-bit, across 24 registered architectures.

**Measured on a 16 GB M4 MacBook Pro** (16–40 GB models that can't fully load, all lossless):

| Model (4-bit) | Size on disk | tok/s | How |
|---|---|---|---|
| Qwen3-30B-A3B (MoE) | 16 GB | **2.8–3.2** | expert streaming + expert-aware cache + persistent scatter buffers (0.9.x, auto budget 6 GB) |
| Qwen3-30B-A3B (MoE) | 16 GB | **up to 1.54 speculative** | + deep-K spec (M≈7 on reasoning prompts; pre-scatter-fix number — re-benchmark pending) |
| Qwen2.5-32B (dense) | 18 GB | **0.9–1.1** | streaming + deep-K spec |
| Llama-3.3-70B (dense) | 40 GB | **~0.9** | streaming + deep-K spec |
| gpt-oss-120b (117B MoE, 59 GB packed) | 64 GB | **1.65–1.96 greedy** | community-verified (M1 Max, 64 GB, v0.5.0) |
| gpt-oss-120b (117B MoE, 59 GB packed) | 16 GB | 0.11–0.14 (correctness floor) | expert streaming, capacity-bound not defect-bound |

For scale: naive streaming (or `llama.cpp` mmap-thrashing) gives ~0.1–0.2 tok/s on the same
hardware for the 70B, and ~0.5 tok/s for the 30B MoE. Nothing here is a quality tradeoff —
"lossless" means the streamed model produces exactly the tokens the full-RAM model would.

### Which model for your RAM

- **16 GB** → `Qwen3-30B-A3B-4bit`, default `--budget auto` (picks 6 GB — the measured
  optimum, **3.2 tok/s greedy**). **More budget is not faster**: past the memory cliff the
  cache fights the macOS compressor — a measured 4/5/6/8/10 GB sweep on a 16 GB M4 gave
  1.79 / 2.52 / **3.23** / 2.84 / 1.33 tok/s. When in doubt go lower, not higher.
- **32–48 GB** → a dense 70B with speculative decoding, e.g. `Llama-3.3-70B-Instruct-4bit`
  with `--draft mlx-community/Llama-3.2-1B-Instruct-4bit` (community: 4.57 tok/s spec vs 3.39
  greedy on code at 64 GB/budget 48GB; 1.16 spec vs 0.10 greedy on a 32 GB M1 Pro at budget
  16GB). **The draft must share the target's tokenizer** — the default draft (Qwen3-0.6B) does
  not match Llama, and a mismatched draft makes the speculative arm useless (~0 acceptance).
- **64 GB+** → `gpt-oss-120b-4bit` with `--budget 58GB` (1.65–1.96 tok/s greedy).

The 70B and 120B numbers above are community-verified (M1 Max, 64 GB, v0.5.0) — see
[docs/community-results.md](docs/community-results.md) for the full shareable reports.

**Try it in three commands** (needs an Apple Silicon Mac, Python 3.11+, and ~35 GB of free
disk — the HF download and the packed copy each take ~16 GB; the download cache can be
deleted after packing):

```bash
pip install nunspark
nunspark pack mlx-community/Qwen3-30B-A3B-4bit ./packed/qwen3-30b
nunspark generate ./packed/qwen3-30b --max-tokens 200 --metrics \
  --prompt "Explain how a B-tree stays balanced."
```

That's a 30-billion-parameter model generating on a machine that cannot hold it in memory.
Add `--draft-model Qwen/Qwen3-0.6B --num-draft-tokens 24` for the speculative mode.

> This is an experimental research project, not a production inference server. Read
> [requirment.md](requirment.md) for the honest story of what worked and what didn't,
> [report.md](report.md) for the dense-model benchmarks,
> [docs/plan4-m3-gate-summary.md](docs/plan4-m3-gate-summary.md) for the MoE numbers above,
> [docs/plan4-m5-gate-summary.md](docs/plan4-m5-gate-summary.md) for the gpt-oss-120b gate, and
> [docs/community-results.md](docs/community-results.md) for shareable `nunspark bench` reports
> from real machines other than the maintainer's.

---

## Why this exists

The obvious approach — split a model into pieces and load only the weights you need — was
tried first in its plain form and it *works*, but it's not competitive: naive weight-streaming
gives roughly the same throughput as `llama.cpp`'s `mmap` (~0.1–0.2 tok/s for a 70B model on a
16 GB M4). Streaming alone is not a moat. Two findings changed that.

### Finding 1: the disk-bound regime rewards *deep* speculation

Speculative decoding's usual cost (extra compute for the draft model) is hidden behind the
dominant cost of reading weights off disk. That means you can push speculation much deeper
(larger K) than anyone tuning for GPU-bound serving would ever try, and each additional
accepted draft token is nearly free — it saves a full weight-read pass instead of costing
extra latency. Measured acceptance multipliers rose from ~2.5× to ~5× as K grew, while naive
tok/s fell — opposite slopes, which is the signal that this regime rewards deep speculation
differently than GPU serving does. And it's lossless: the target model verifies every draft
token, so every emitted token is the target's own argmax (deterministic per config) -- though
not guaranteed byte-for-byte vs single-token greedy at model scale, since a multi-token verify
pass runs fp16 numerics under different kernel shapes (rare, self-healing near-tie flips; see
docs/plan5-m2-mismatch-investigation.md). See [requirment.md](requirment.md)
for the full analysis and caveats (vocab lock-in between draft/target, etc). The dense-model
benchmarks in [report.md](report.md) confirmed it, and added a second lesson:
**draft–target agreement, not resident cache size, is the primary determinant of throughput.**

### Finding 2: MoE models are the natural fit for streaming

A dense model makes you read *every* weight for *every* token — streaming can only amortize
that. A mixture-of-experts model already routes each token through a small slice of itself:
Qwen3-30B-A3B fires 8 of 128 experts per layer, ~1 GB of expert weights per token out of a
16 GB model. NunSpark packs each expert as its own piece and exploits that sparsity end to end:

- **Load only fired experts.** The router runs first; only the 8 winners per layer are read.
  This holds in the speculative tree-verify path too, which loads the fired *union* (measured
  51–65% of experts at K=24) instead of all 128.
- **Cache experts on their own terms.** Dense layers scan cyclically (MRU is right); experts
  reuse sparsely (LRU is right). A two-region cache with per-layer cores pinned took the
  expert hit rate from 57–64% to 89–92% at the same 8 GB budget — that alone was 0.5 → 1.3–2.1
  tok/s.
- **Prefetch the next verify pass's experts.** Consecutive verify passes fire 74–80%
  overlapping expert sets. Predicted experts stream in a low-priority I/O tier into a staging
  buffer that can never evict the live working set. Speculative decoding on top: up to
  **1.54 tok/s lossless** on reasoning prompts (2.4× the no-prefetch control).
- **Scatter into persistent buffers, never rebuild them.** A per-token time-attribution probe
  (2026-07-17) found streaming decode was *not* disk-bound: 59–70% of every token went to
  rebuilding zero-filled 128-expert buffers each layer (~15 GB of transient writes per token)
  just to host the 8 fired experts. The engine now keeps one persistent buffer set and
  scatter-updates only the fired rows — byte-identical output, and 30B decode on a 16 GB M4
  went **1.6 → 3.2 tok/s** in live sessions (a 3.2× bucket-level speedup in controlled flushed
  probes; see `scripts/results/decode_time_attribution.json`).

Every one of those policies was chosen by measurement — the failed variants and their numbers
are written up in [docs/](docs/) gate summaries.

## Design principles

- **Maximize capability, not raw speed.** The goal is running the biggest model your disk can
  hold, not the fastest possible tok/s. Seconds-per-token is an acceptable outcome; the metric
  that matters is "does it run at all."
- **Built on stock MLX / mlx-lm**, not architecture-specific hacks. NunSpark reuses mlx-lm's
  layer math and registry wherever possible so new model families need minimal glue.
- **Minimize target-weight reads.** Every mechanism (prefetch, caching, speculative decoding,
  prefix cache) exists to reduce how many bytes of the *target* model get pulled off disk per
  generated token — that's the bottleneck everything else is fighting.

## What's implemented

- **Packer** (`nunspark pack`) — converts an mlx-lm-compatible model (local dir or HF repo) into
  per-layer safetensors pieces plus a JSON manifest. Supports 4-bit quantized weights
  (embedding + lm_head triplets included).
- **Streaming engine** (`StreamingEngine`) — loads pieces into a byte-budgeted cache
  (`PieceCache`) with a background prefetch pool that overlaps disk I/O with compute.
  Forward pass is verified bit-identical to full-load mlx-lm, fp16 and 4-bit alike.
- **MoE expert streaming** — for MoE models (Qwen3-MoE, gpt-oss), the packer splits each layer
  into a core piece (attention/norms/router) plus one piece per expert; the engine loads only
  the experts the router fires. Two-region cache (MRU for the dense cyclic scan, LRU for
  sparse expert reuse, cores pinned), fired-union selective loading in tree verify, and
  temporal expert prefetch at verify-pass granularity into an eviction-safe staging buffer.
  All bit-identical to loading every expert.
- **Speculative decoding** — the core reason this is usable:
  - Standard draft-model speculative decoding (`--draft-model`, any mlx_lm-format model that
    shares a tokenizer with the target).
  - EAGLE-style feature-level speculation (`--eagle-drafter`) — no full draft model needed.
  - Tree-based speculative decoding for exploring multiple candidate continuations per step.
  - Gemma3/4 MTP (multi-token-prediction) assistant drafting.
  - Configurable draft depth (`--num-draft-tokens`) and acceptance policy
    (`--accept-top-k`: `1` = lossless, `>1` = "fast mode" with controlled deviation).
- **KV-cache management** — RAM-resident KV cache with optional 4/8-bit quantization
  (`--kv-bits`), and a server-side single-slot prefix cache that reuses the previous request's
  KV state and prefills only the new suffix (`--no-prefix-cache` to disable).
- **OpenAI-compatible server** (`nunspark serve`) — serves a packed model behind a v1-style API.
- **Local web UI** (`nunspark web`) — a FastAPI app for long, document-driven **batch**
  generation (not interactive chat — this engine is seconds-per-token). Upload files, submit a
  batch, get one generation job per file streamed to disk with live progress over SSE.
  Compatible jobs are decoded **together** (up to 8 sharing every weight read — measured
  2.3× total throughput at 8 jobs for ~8% more memory; see the Web UI section).
  Plain-text uploads only (.txt/.md): PDFs and other binaries are rejected with a clear
  error — extract the text first. Prompts are also capped against your machine's unified
  RAM (the KV cache grows ~100 KB/token on a 30B model; an uncapped mega-prompt can
  swap-storm the whole Mac). Set `advanced.max_prompt_tokens` to override the cap.
- **Architecture support** — 24 decoder-only model types are registered
  (`src/nunspark/architectures.py`), including Llama, Mistral, Phi-3, Qwen2, Qwen3 (dense +
  MoE via selective expert streaming), Gemma3/Gemma4 (including heterogeneous attention),
  GLM/GLM-4, OLMo-2, InternLM3, gpt-oss, and more. The current list is queryable in code via
  `nunspark.architectures.supported_model_types()`; `pack` will tell you if a model's
  `model_type` isn't supported. Multimodal models are not supported (text decoders only).
  See the **Supported models** table below for the full list with per-family notes.

Not built / deferred: TurboQuant 2–4 bit KV (blocked on upstream MLX SDPA support).

### Supported models

NunSpark streams any text decoder whose `model_type` is in the registry — 26 types today.
It rides stock mlx_lm layer modules, so any quantization mlx-community publishes for these
families (4-bit, 6-bit, 8-bit, fp16, mixed mxfp4) works as-is. `nunspark pack` fails with a
clear message (and the full supported list) on anything unregistered; multimodal models are
not supported (text decoders only).

| `model_type` | Families / examples | Notes |
|---|---|---|
| `qwen3_moe` | Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Coder-480B | **Selective expert streaming** — the headline path; only fired experts are read per token. Field-verified from 30B (16 GB Mac) to 480B (128 GB, community). |
| `gpt_oss` | gpt-oss-20b, gpt-oss-120b | Selective expert streaming + sliding-window attention. `--kv-bits` unsupported (attention sinks); excluded from web-UI batched decode (falls back to sequential). |
| `glm_moe_dsa`, `deepseek_v32` | GLM-5.2 (744B/39B-active), DeepSeek-V3.2 | Selective expert streaming + MLA latent-KV + DSA sparse-attention indexer. Sequential generate/speculative paths only (batched decode and tree spec refuse cleanly); `--kv-bits` unsupported (latent cache). Verified bit-identical to full-load on CI fixtures; no real-model performance measurements yet — the ~39B-active working set wants a high-RAM Mac, and community reports are welcome. |
| `llama`, `mistral` | Llama 3.x, SmolLM, Mistral 7B | Dense. Llama-3.3-70B field-verified on 32 GB; pair with `--draft mlx-community/Llama-3.2-1B-Instruct-4bit` for speculation (the draft must share the target's tokenizer). |
| `qwen2`, `qwen3` | Qwen2.5 (0.5B–72B), Qwen3 dense (0.6B–32B) | Dense. Qwen3-0.6B is the default bench draft; Qwen2.5-32B field-verified. |
| `gemma3`, `gemma3_text`, `gemma4`, `gemma4_text` | Gemma 3 / Gemma 4 | Heterogeneous (sliding + global) attention layers; excluded from web-UI batched decode. |
| `gemma4_assistant` | Gemma 4 MTP assistant | Used as a feature-level drafter for Gemma 4 targets (multi-token prediction). |
| `apertus`, `ernie4_5`, `glm`, `glm4`, `helium`, `hunyuan_v1_dense`, `internlm3`, `mimo`, `olmo2`, `phi3`, `seed_oss`, `telechat3`, `youtu_llm` | GLM-4, OLMo-2, Phi-3, InternLM3, ERNIE 4.5, Hunyuan, … | Dense, via the same registry path as llama/qwen (stock mlx_lm modules, verified bit-identical to full-load on CI fixtures). Fewer real-model field reports — results welcome in the community benchmark thread. |

The authoritative list is `nunspark.architectures.supported_model_types()`; models with
community-measured numbers are collected in
[docs/community-results.md](docs/community-results.md).

## Requirements

- **Apple Silicon Mac** (this is an MLX / unified-memory project — it does not run on
  non-Apple hardware).
- **Python 3.11+**. The system Python on macOS is commonly 3.9, which is too old — create the
  venv with an explicit version:

  ```bash
  uv venv --python 3.11
  ```

- [`uv`](https://github.com/astral-sh/uv) for dependency management (recommended). A plain
  `uv venv` with no `--python` flag may pick up the system 3.9 interpreter and fail — always
  pass `--python 3.11`.

## Installation

### From PyPI

```bash
pip install nunspark            # or: uv pip install nunspark
```

That's the whole install — you get the `nunspark` CLI (`pack` / `generate` / `serve` / `web`)
and the Python API. Requires Python 3.11+ on an Apple Silicon Mac (the `mlx` dependency has
no wheels for other platforms, so installation fails anywhere else by design).

For the web UI, add the `web` extra:

```bash
pip install "nunspark[web]"
```

### From source

```bash
git clone <this-repo>
cd Nunspark
uv venv --python 3.11
uv sync
```

For the web UI, install the optional `web` extra:

```bash
uv sync --extra web
```

> **Gotcha:** the web dependencies (fastapi, uvicorn, python-multipart) are optional. If you
> run a plain `uv run nunspark web ...` after only `uv sync`, `uv run` re-syncs the environment
> and **drops the extra**, causing `ModuleNotFoundError`. Always pass `--extra web` on the `uv
> run` invocation itself (or install the extra into a stable, non-`uv`-managed venv):
>
> ```bash
> uv run --extra web nunspark web --packed-root ./models --port 8000
> ```

## Quickstart

### 1. Pack a model

```bash
uv run nunspark pack mlx-community/SmolLM2-360M-Instruct-bf16 ./packed/smollm2
```

`model_dir` accepts either a local path or a Hugging Face repo id. Output is a directory of
per-layer safetensors files plus `manifest.json`.

### 2. Generate

```bash
uv run nunspark generate ./packed/smollm2 \
  --prompt "Explain disk-streaming inference in two sentences." \
  --max-tokens 128 \
  --budget 1GB \
  --metrics
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--budget` | Resident weight budget (e.g. `512MB`, `4GB`, or `auto`). Default `auto`: **75% of (total RAM − 8 GB)** — 16 GB → 6 GB, 64 GB → 42 GB, 128 GB → 90 GB. The fixed 8 GB reflects the OS-plus-apps baseline, which doesn't scale with RAM. **More budget is not faster**: past the memory cliff the cache fights the macOS compressor, and the cliff is asymmetric (a measured 4/5/6/8/10 GB sweep on a 16 GB M4 with the 30B gave 1.79 / 2.52 / **3.23** / 2.84 / 1.33 tok/s) — when in doubt go lower, not higher. Community numbers below were run with an explicit `--budget`; pin one yourself for comparable results. |
| `--kv-budget` | Resident KV-cache budget (default: unbounded). |
| `--kv-bits {4,8}` | Quantize the KV cache (default fp16). |
| `--io-threads` / `--warm-window` | Parallel page-cache warming (experimental; measured net-neutral or negative in most configurations)). |
| `--draft-model <path>` | Enable speculative decoding against a smaller draft model (must share a tokenizer with the target). |
| `--eagle-drafter <path>` | Use a trained EAGLE feature-level drafter instead of a full draft model. |
| `--ngram-draft` | Model-free prompt-lookup speculative decoding: drafts `--num-draft-tokens` tokens from the most recent prior occurrence of the context suffix. Zero draft-model cost, tokenizer-exact, lossless. Mutually exclusive with `--draft-model`/`--eagle-drafter`. |
| `--ngram-max` | Longest suffix n-gram tried by `--ngram-draft` (default 3). |
| `--no-ngram-adaptive` | Disable the n-gram drafter's adaptive policy. By default it watches its own acceptance rate, shrinks its proposals when they stop earning, and switches itself off entirely (re-probing cheaply every ~50 steps) when speculation isn't paying — so on workloads where prompt-lookup can't win (most MoE decoding) it costs ≈nothing instead of 2–3× throughput. Pass this flag to pin the fixed `--num-draft-tokens` behavior for benchmarking. |
| `--num-draft-tokens` | Draft tokens proposed per speculative sweep (default 16 — the "deep-K" lever described above). |
| `--accept-top-k` | `1` = lossless speculative decoding; `>1` = fast mode (bounded deviation from the target distribution). |
| `--lookahead` | Experimental cross-layer MoE expert prefetch: on decode passes, replicates the next layer's router on the current residual stream and speculatively stages its predicted experts. Output-identical; prefetch only. Silently no-ops on architectures/cores it can't replicate. Opt-in, off by default. |
| `--metrics` | Print tok/s, peak memory, cache hit/miss, and (if speculative) acceptance-multiplier stats after generation. |

### The headline use case: a 30B MoE model on a 16 GB Mac

This is the configuration behind the numbers at the top — a 16 GB model on a 16 GB machine,
streaming only the experts each token actually fires:

```bash
# 1. Pack the oversized target once. For MoE models this splits every layer into a
#    core piece + one piece per expert (~6200 files for Qwen3-30B-A3B).
uv run nunspark pack mlx-community/Qwen3-30B-A3B-4bit ./packed/qwen3-30b

# 2. Stream it. Greedy decode alone reaches ~3.2 tok/s at the default auto budget
#    (6 GB on a 16 GB machine, 0.9.x):
uv run nunspark generate ./packed/qwen3-30b \
  --prompt "Explain how a B-tree stays balanced." \
  --max-tokens 256 --metrics

# 3. Add deep-K speculation. The draft (Qwen3-0.6B) shares Qwen3's tokenizer, so its
#    proposals are valid target tokens; one 30B verify sweep then buys up to ~7 tokens.
uv run nunspark generate ./packed/qwen3-30b \
  --prompt "Think step by step: a train leaves at 9:14 travelling 83 km/h ..." \
  --draft-model Qwen/Qwen3-0.6B \
  --num-draft-tokens 24 \
  --accept-top-k 1 \
  --max-tokens 256 --metrics
```

With `--metrics` you'll see the acceptance multiplier M and the effective tok/s. Which mode
wins depends on the prompt: speculation shines where the draft agrees with the target
(reasoning chains, prose — measured M up to 7.4 and 1.54 tok/s), while terse low-agreement
prompts can be faster in plain greedy (measured 2.1 tok/s on prose, 1.5 on code — pre-scatter-fix
numbers; 0.9.x greedy is faster still, so the spec-vs-greedy break-even shifts toward greedy
until the spec arms are re-benchmarked). `--accept-top-k 1`
keeps it lossless; raise it for "fast mode" if you'll accept bounded deviation. The same two
commands work for dense models (Qwen2.5-32B, Llama-3.3-70B) — speculation is the main lever
there, since every token reads the full layer stack.

**For sparse MoE targets, speculative decoding measures *slower* than greedy** — a K-token
verify pass fires each position's own experts with low cross-token overlap, so expert-union
I/O grows with K while acceptance doesn't. Use greedy for MoE (Qwen3-30B-A3B, gpt-oss); use
speculative decoding for dense targets. Full measurement and root cause:
[docs/plan4-m5-gate-summary.md](docs/plan4-m5-gate-summary.md).

### 64 GB+: gpt-oss-120b, the largest MoE target

Same shape, bigger model. The packer streams the source weights lazily, so packing only needs
~1.7 GB RAM even though the source download is ~63 GB and the packed output is ~59 GB — 16 GB
machines can pack it, just not run it fast (see the "correctness floor" row above):

```bash
# 1. Pack. Needs ~63 GB free for the HF download + ~59 GB for the packed copy.
uv run nunspark pack mlx-community/gpt-oss-120b-4bit ./packed/gpt-oss-120b

# 2. Generate greedy, budget sized to your RAM (64 GB machine shown):
uv run nunspark generate ./packed/gpt-oss-120b \
  --prompt "Explain how a B-tree stays balanced." \
  --budget 58GB --max-tokens 256 --metrics
```

Community-verified (M1 Max, 64 GB, v0.5.0): 1.65–1.96 tok/s greedy at `--budget 58GB`. See
[docs/community-results.md](docs/community-results.md) for the full per-workload table and
[docs/plan4-m5-gate-summary.md](docs/plan4-m5-gate-summary.md) for the M5 gate writeup.

For an end-to-end script that also downloads/converts the model, verifies streamed output
against a full-load run, and can A/B linear vs tree speculation, see
[scripts/try_real_model.py](scripts/try_real_model.py):

```bash
uv run python scripts/try_real_model.py --model mlx-community/Qwen3-30B-A3B-4bit --repack \
    --draft Qwen/Qwen3-0.6B --ab-spec --temp 0.0 --budget 4GB
```

### 3. Serve an OpenAI-compatible API

```bash
uv run nunspark serve ./packed/smollm2 --port 8080 --draft-model <path-to-draft>
```

Accepts the same budget/KV/speculative flags as `generate`, plus `--model-name` and
`--no-prefix-cache` (disables reusing the previous request's KV state across calls).

### 4. Web UI (batch generation)

```bash
uv run --extra web nunspark web --packed-root ./packed --port 8000
```

Open `http://127.0.0.1:8000`, pick or pack a model, upload files, submit a batch. Each file
becomes one generation job; output streams to `<output_dir>/<name>.out.txt` with a
`.meta.json` sidecar, and progress is pushed live over SSE. This is built for long
document-batch jobs, not interactive chat — the engine runs seconds-per-token.

**Batched decode.** Queued jobs with matching settings (same model, budget, KV options,
temperature, max tokens, no draft model) are automatically decoded together, up to 8 at a
time, sharing every weight read: each layer's core is read once for the whole group and the
group's fired experts are loaded as one union. Measured on Qwen3-30B-A3B at an 8 GB budget,
8 jobs together produce **2.3× the total tokens/s** of running them back-to-back, for ~8%
more peak memory. Batched rows are target-greedy correct but, like all multi-token passes,
not guaranteed byte-identical to a solo run of the same prompt (see the exactness note
above); a job's `.meta.json` records `batched`/`batch_size`, and tokens for batched jobs
arrive in a burst when the group finishes rather than streaming one by one. Set
`batch: false` under advanced settings to force sequential runs.

## Community benchmark

```bash
uv run nunspark bench
```

One command: on first run it packs the reference model (`mlx-community/Qwen3-30B-A3B-4bit`)
into `./packed/` if it isn't already there (~35 GB disk — the download plus the packed
copy — and asks for confirmation unless you pass `--yes`), then runs the three-workload
suite (code / prose / reasoning) greedy and speculative, and prints your machine's chip,
RAM, macOS/nunspark/mlx versions, and a tok/s + acceptance-multiplier + peak-memory table.
Use `--quick` for a fast single-workload greedy-only smoke run, or `--no-spec` to skip the
speculative passes. Pass any local packed dir, local model dir, or other HF repo id as the
positional `model` argument to benchmark something other than the default.

The report prints as a GitHub-markdown block between `BEGIN`/`END SHAREABLE REPORT`
delimiters — copy it into a GitHub issue or discussion on this repo so results across
machines are comparable.

## Development

```bash
uv venv --python 3.11        # not a bare `uv venv` — the system Python is too old
uv sync --extra dev
uv run pytest -q
```

Notes:
- Always invoke tests as plain `uv run pytest ...`. Do **not** pass `--active` — if your shell
  has `VIRTUAL_ENV` set to something else (e.g. a system Python framework), `--active` forces
  that broken environment; plain `uv run` correctly ignores it and uses the project's `.venv`.
- Watch out for `... | tail` masking a failing exit code in chained commands — check the pytest
  summary line explicitly rather than trusting `$?` after a pipe.
- `scripts/` contains standalone probes used during development (`io_warm_probe.py`,
  `spec_accept_probe.py`, `kv_probe.py`, `try_real_model.py`, `train_eagle_drafter.py`, etc.) —
  useful references for benchmarking a real model end-to-end, not part of the package API.

## Project layout

```
src/nunspark/
  packer.py, manifest.py, piece_store.py, piece_cache.py   # model → pieces, and piece I/O/caching
  engine.py, archspec.py, architectures.py, tree_shape.py   # per-layer streaming forward pass, arch registry
  generate.py, generate_optimized.py                        # generation loops (plain + speculative)
  eagle_drafter.py, assistant.py, gemma4_assistant.py        # speculative drafters
  kv_store.py, prefix_cache.py, tree_spec.py                 # KV cache management
  server.py                                                  # OpenAI-compatible API server
  webapp/                                                    # FastAPI batch-generation UI
  cli.py                                                     # `nunspark` entry point (pack/generate/serve/web)
scripts/                                                     # standalone benchmarking/probe scripts
tests/                                                       # pytest suite, mirrors src/nunspark modules
```

## Known limitations

- **Vocab lock-in**: draft and target models must share a tokenizer, which narrows which
  (draft, target) pairs are usable for speculative decoding — fine for major model families,
  awkward for exotic ones.
- **gpt-oss** opts out of quantized KV: its attention-sink mechanism is rejected by quantized
  SDPA, and the engine fails fast at startup rather than silently falling back.
- I/O warming (`--io-threads`/`--warm-window`) was investigated in depth and found to be net-
  neutral-to-negative versus plain `mmap` in most tested configurations — it's exposed as a
  flag for experimentation, not recommended by default.
- The engine decodes one *stream* at a time, but the web UI batches compatible queued jobs
  into a single lock-step decode (up to 8 rows sharing every weight sweep — see the Web UI
  section). `nunspark serve` still processes one request at a time: true continuous batching
  (requests joining a running batch) is not implemented. Sliding-window models (gpt-oss)
  are excluded from batched decode and fall back to sequential runs.

## Acknowledgements

Two independent MoE-streaming runtimes shaped parts of NunSpark; ideas adopted from
them were validated against our own gates before shipping (see `docs/backlog.md`
entries #7 and #13 for the full adoption reviews):

- **[TensorFold](https://github.com/ashhart/TensorFold)** (MIT) — an MoE-streaming
  runtime on MLX. Inspired our garbage-drafter invariant test, the n-gram drafter's
  adaptive proposal shrink, and batched decode with shared per-layer expert unions
  (Plan 5). Their negative result on previous-token expert prefetch independently
  confirmed our own Phase-1 measurement.
- **[Colibrì](https://github.com/JustVugg/colibri)** (Apache 2.0) — a pure-C GLM-5.2
  streaming engine. Their router-predictability measurement corroborated our
  cross-layer lookahead probe (`--lookahead`, Plan 7), and their MTP-drafter
  precision finding is recorded for our future GLM-5.2 drafter work. Their expert
  atlas methodology also informed our expert hot-set analysis.

## License

MIT — see [LICENSE](LICENSE).
