# NunSpark

**Run LLMs that don't fit in your Mac's RAM — at usable speeds.**

Your Mac has 16 GB of unified memory. The model you want needs 16–40 GB of weights. NunSpark
runs it anyway: it packs the model into per-layer (and, for MoE models, per-*expert*) pieces
on disk and streams exactly the weights each token needs through a small resident budget —
then claws the speed back with three levers that only make sense in the disk-bound regime:

- **MoE expert streaming** — a Qwen3-30B-A3B token fires only 8 of 128 experts per layer.
  NunSpark loads *only those*, caches them expert-aware, and prefetches the ones the next
  verify pass will probably fire. Measured on a 16 GB M4: **0.5 → 2.1 tok/s**, reading
  ~100 MB/token instead of ~1 GB.
- **Deep-K speculative decoding** — on disk-bound hardware the draft model's compute is free,
  so speculation can go far deeper (K=16–24) than GPU serving ever would. One weight sweep
  buys up to 7 accepted tokens. Lossless: output is bit-identical to running the target alone.
- **Byte-budgeted streaming engine** — every forward pass verified bit-identical to full-load
  mlx-lm, fp16 and 4-bit, across 24 registered architectures.

**Measured on a 16 GB M4 MacBook Pro** (16–40 GB models that can't fully load, all lossless):

| Model (4-bit) | Size on disk | tok/s | How |
|---|---|---|---|
| Qwen3-30B-A3B (MoE) | 16 GB | **1.5–2.1** | expert streaming + expert-aware cache |
| Qwen3-30B-A3B (MoE) | 16 GB | **up to 1.54 speculative** | + deep-K spec (M≈7 on reasoning prompts) |
| Qwen2.5-32B (dense) | 18 GB | **0.9–1.1** | streaming + deep-K spec |
| Llama-3.3-70B (dense) | 40 GB | **~0.9** | streaming + deep-K spec |

For scale: naive streaming (or `llama.cpp` mmap-thrashing) gives ~0.1–0.2 tok/s on the same
hardware for the 70B, and ~0.5 tok/s for the 30B MoE. Nothing here is a quality tradeoff —
"lossless" means the streamed model produces exactly the tokens the full-RAM model would.

**Try it in three commands** (needs an Apple Silicon Mac, Python 3.11+, and ~35 GB of free
disk — the HF download and the packed copy each take ~16 GB; the download cache can be
deleted after packing):

```bash
pip install nunspark
nunspark pack mlx-community/Qwen3-30B-A3B-4bit ./packed/qwen3-30b
nunspark generate ./packed/qwen3-30b --budget 8GB --max-tokens 200 --metrics \
  --prompt "Explain how a B-tree stays balanced."
```

That's a 30-billion-parameter model generating on a machine that cannot hold it in memory.
Add `--draft-model Qwen/Qwen3-0.6B --num-draft-tokens 24` for the speculative mode.

> This is an experimental research project, not a production inference server. Read
> [requirment.md](requirment.md) for the honest story of what worked and what didn't,
> [report.md](report.md) for the dense-model benchmarks, and
> [docs/plan4-m3-gate-summary.md](docs/plan4-m3-gate-summary.md) for the MoE numbers above.

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
token, so output is identical to running the target alone. See [requirment.md](requirment.md)
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
- **Architecture support** — 24 decoder-only model types are registered
  (`src/nunspark/architectures.py`), including Llama, Mistral, Phi-3, Qwen2, Qwen3 (dense +
  MoE via selective expert streaming), Gemma3/Gemma4 (including heterogeneous attention),
  GLM/GLM-4, OLMo-2, InternLM3, gpt-oss, and more. The current list is queryable in code via
  `nunspark.architectures.supported_model_types()`; `pack` will tell you if a model's
  `model_type` isn't supported. Multimodal models are not supported (text decoders only).

Not built / deferred: TurboQuant 2–4 bit KV (blocked on upstream MLX SDPA support).

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
| `--budget` | Resident weight budget (e.g. `512MB`, `4GB`). Lower = more disk reads, less RAM. |
| `--kv-budget` | Resident KV-cache budget (default: unbounded). |
| `--kv-bits {4,8}` | Quantize the KV cache (default fp16). |
| `--io-threads` / `--warm-window` | Parallel page-cache warming (experimental; measured net-neutral or negative in most configurations)). |
| `--draft-model <path>` | Enable speculative decoding against a smaller draft model (must share a tokenizer with the target). |
| `--eagle-drafter <path>` | Use a trained EAGLE feature-level drafter instead of a full draft model. |
| `--num-draft-tokens` | Draft tokens proposed per speculative sweep (default 16 — the "deep-K" lever described above). |
| `--accept-top-k` | `1` = lossless speculative decoding; `>1` = fast mode (bounded deviation from the target distribution). |
| `--metrics` | Print tok/s, peak memory, cache hit/miss, and (if speculative) acceptance-multiplier stats after generation. |

### The headline use case: a 30B MoE model on a 16 GB Mac

This is the configuration behind the numbers at the top — a 16 GB model on a 16 GB machine,
streaming only the experts each token actually fires:

```bash
# 1. Pack the oversized target once. For MoE models this splits every layer into a
#    core piece + one piece per expert (~6200 files for Qwen3-30B-A3B).
uv run nunspark pack mlx-community/Qwen3-30B-A3B-4bit ./packed/qwen3-30b

# 2. Stream it. Greedy decode alone reaches 1.5–2.1 tok/s at an 8 GB budget:
uv run nunspark generate ./packed/qwen3-30b \
  --prompt "Explain how a B-tree stays balanced." \
  --budget 8GB --max-tokens 256 --metrics

# 3. Add deep-K speculation. The draft (Qwen3-0.6B) shares Qwen3's tokenizer, so its
#    proposals are valid target tokens; one 30B verify sweep then buys up to ~7 tokens.
uv run nunspark generate ./packed/qwen3-30b \
  --prompt "Think step by step: a train leaves at 9:14 travelling 83 km/h ..." \
  --draft-model Qwen/Qwen3-0.6B \
  --num-draft-tokens 24 \
  --accept-top-k 1 \
  --budget 8GB --max-tokens 256 --metrics
```

With `--metrics` you'll see the acceptance multiplier M and the effective tok/s. Which mode
wins depends on the prompt: speculation shines where the draft agrees with the target
(reasoning chains, prose — measured M up to 7.4 and 1.54 tok/s), while terse low-agreement
prompts can be faster in plain greedy (2.1 tok/s on prose, 1.5 on code). `--accept-top-k 1`
keeps it lossless; raise it for "fast mode" if you'll accept bounded deviation. The same two
commands work for dense models (Qwen2.5-32B, Llama-3.3-70B) — speculation is the main lever
there, since every token reads the full layer stack.

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
- This is a single-stream engine: the server and web UI both process one generation at a time
  (no concurrent-request batching) since the whole point is disk-bound streaming, not
  throughput-oriented serving.

## License

MIT — see [LICENSE](LICENSE).
