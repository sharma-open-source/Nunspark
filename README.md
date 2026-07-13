# NunSpark

**Run oversized LLMs on Apple Silicon by streaming weights from disk, layer by layer.**

NunSpark targets a simple, unglamorous problem: your Mac has 16 GB of unified memory and you
want to run a model that needs 40+ GB of weights. Instead of refusing to load, NunSpark packs
a model into one file per transformer layer ("piece"), and streams each piece from disk into a
small resident budget just before it's needed, discarding it right after. Combined with
disk-aware speculative decoding, this makes large (13–30B, 4-bit) models usable — a few tokens
per second instead of not-at-all — on machines that could never hold the model in RAM.

> This is an experimental research project, not a production inference server. Read
> [requirment.md](requirment.md) for the honest story of what worked and what didn't, and
> [report.md](report.md) for measured benchmarks on a real 16 GB M4 (Qwen2.5-32B and
> Llama-3.3-70B, 4-bit).

---

## Why this exists

The obvious approach — split a model into pieces and load only the weights you need — was
tried first in its plain form and it *works*, but it's not competitive: naive weight-streaming
gives roughly the same throughput as `llama.cpp`'s `mmap` (~0.1–0.2 tok/s for a 70B model on a
16 GB M4). Streaming alone is not a moat.

The actual finding: in the **disk-bound regime**, speculative decoding's usual cost (extra
compute for the draft model) is hidden behind the dominant cost of reading weights off disk.
That means you can push speculation much deeper (larger K) than anyone tuning for GPU-bound
serving would ever try, and each additional accepted draft token is nearly free — it saves a
full weight-read pass instead of costing extra latency. Measured acceptance multipliers rose
from ~2.5× to ~5× as K grew, while naive tok/s fell — opposite slopes, which is the signal
that this regime rewards deep speculation differently than GPU serving does.

| Target (4-bit) | Naive stream | × M(~5) | × M(~3, conservative) |
|-----------------|--------------|---------|------------------------|
| 70B             | 0.08 tok/s   | 0.40    | 0.24 tok/s             |
| 30B             | 0.17 tok/s   | 0.85    | 0.51 tok/s             |
| 26B             | 0.20 tok/s   | 1.00    | 0.60 tok/s             |

Speculative decoding is lossless (identical output distribution to running the target alone),
so this is a straight 3–5× speedup, not a quality tradeoff. See
[requirment.md](requirment.md) and
[docs/superpowers/specs/2026-06-05-nunspark-speculative-streaming-design.md](docs/superpowers/specs/2026-06-05-nunspark-speculative-streaming-design.md)
for the full analysis and caveats (vocab lock-in between draft/target, why the win concentrates
in the 13–30B band rather than the hero 70B number, etc).

**Measured results** ([report.md](report.md)) confirm the thesis on a real 16 GB M4: a streamed
Qwen2.5-32B-4bit reaches **~0.9–1.1 tok/s losslessly** (and ~2.6–2.8 tok/s in approximate
top-k mode), and a streamed Llama-3.3-70B-4bit stays usable at **~0.9 tok/s** — both while
keeping only ~1.1–1.4 GB of weights resident out of 18–40 GB models. The headline finding
from those runs: **draft–target agreement, not resident cache size, is the primary determinant
of throughput.**

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
- **Streaming engine** (`StreamingEngine`) — loads one piece at a time into a byte-budgeted LRU
  cache (`PieceCache`) with a background prefetch worker that overlaps disk I/O with compute.
  Forward pass is verified bit-identical to full-load mlx-lm, fp16 and 4-bit alike.
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

Not built / deferred: MoE expert-streaming beyond the existing selective-expert loading;
TurboQuant 2–4 bit KV (blocked on upstream MLX SDPA support).

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
| `--io-threads` / `--warm-window` | Parallel page-cache warming (experimental; measured net-neutral or negative in most configurations — see [docs/superpowers/specs/2026-06-06-nunspark-parallel-io-warming-design.md](docs/superpowers/specs/2026-06-06-nunspark-parallel-io-warming-design.md)). |
| `--draft-model <path>` | Enable speculative decoding against a smaller draft model (must share a tokenizer with the target). |
| `--eagle-drafter <path>` | Use a trained EAGLE feature-level drafter instead of a full draft model. |
| `--num-draft-tokens` | Draft tokens proposed per speculative sweep (default 16 — the "deep-K" lever described above). |
| `--accept-top-k` | `1` = lossless speculative decoding; `>1` = fast mode (bounded deviation from the target distribution). |
| `--metrics` | Print tok/s, peak memory, cache hit/miss, and (if speculative) acceptance-multiplier stats after generation. |

### The headline use case: a mid-tier model that doesn't fit in RAM

This is the reason the project exists — run a 4-bit ~30B model on 16 GB by streaming it, and
recover usable throughput with a small draft model that shares the target's tokenizer:

```bash
# 1. Pack the oversized target once (weights live on disk as per-layer pieces).
uv run nunspark pack mlx-community/Qwen3-30B-A3B-4bit ./packed/qwen3-30b

# 2. Stream it with speculative decoding. The draft (Qwen3-0.6B) shares Qwen3's
#    tokenizer, so its proposals are valid target tokens. --budget caps resident
#    weights well below the full model size; --num-draft-tokens is the "deep-K" lever.
uv run nunspark generate ./packed/qwen3-30b \
  --prompt "Write a Python function that streams a large file line by line." \
  --draft-model Qwen/Qwen3-0.6B \
  --num-draft-tokens 16 \
  --accept-top-k 1 \
  --budget 4GB \
  --max-tokens 256 \
  --metrics
```

With `--metrics` you'll see the acceptance multiplier and the effective-vs-naive tok/s — that
multiplier (measured ~3–5× on easy prompts) is the whole finding. `--accept-top-k 1` keeps it
lossless; raise it for "fast mode" if you'll accept bounded deviation.

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
docs/superpowers/{specs,plans}/                              # design specs and implementation plans, one per feature
scripts/                                                     # standalone benchmarking/probe scripts
tests/                                                       # pytest suite, mirrors src/nunspark modules
```

Each shipped feature has a paired spec + plan doc under `docs/superpowers/` — check there first
for the rationale behind a specific module (e.g. why the prefix cache is single-slot, how KV
quantization interacts with tree speculative decoding, gpt-oss's quantized-KV opt-out).

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
