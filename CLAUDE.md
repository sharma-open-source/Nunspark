# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NunSpark runs LLMs larger than the Mac's RAM on Apple Silicon (MLX): models are packed into per-layer (and per-expert, for MoE) pieces on disk and streamed through a byte budget, with deep-K speculative decoding to claw back speed in the disk-bound regime. Experimental research project, not a production server.

## Environment & commands

Python 3.11 via uv. Create the venv with `uv venv --python 3.11`; run everything through plain `uv run` (never `uv run --active`).

```bash
uv sync --extra dev                 # install with pytest
uv run pytest                       # full test suite (fast — uses tiny random models, no downloads)
uv run pytest tests/test_engine_forward.py            # one file
uv run pytest tests/test_speculative.py -k accept     # one test by keyword
uv run nunspark pack <hf-model-or-dir> ./packed/<name>   # pack an mlx-lm model into pieces
uv run nunspark generate ./packed/<name> --prompt "..." --metrics
uv run nunspark serve ./packed/<name>   # OpenAI v1-compatible API
uv run --extra web nunspark web         # local FastAPI web UI (needs the web extra)
uv run nunspark bench ./packed/<name>   # shareable benchmark report
```

Tests build seeded tiny models (Llama/Qwen3/Qwen3-MoE/gpt-oss) in `tests/conftest.py` — real packed models are never required for the suite.

## Architecture

Data flow: **pack → manifest/pieces on disk → StreamingEngine streams pieces per forward → generate loop**.

- `packer.py` splits an mlx-lm model dir into piece safetensors (per layer; per expert for MoE) plus a `manifest.py` `Manifest` describing pieces, quantization, and architecture.
- `engine.py` `StreamingEngine` is the core: it reconstructs each layer's forward from streamed weights, verified **bit-identical** to full-load mlx-lm (fp16 and 4-bit). For MoE it loads only the experts the router fires (`_scatter_experts` uses persistent buffers), with expert-aware caching and prefetch.
- `piece_store.py` / `piece_cache.py`: disk reads and the byte-budgeted in-RAM cache (bulk-warm on multi-token passes; single-token decode warm is an opt-in engine flag, measured no-effect). `sysmem.py` provides the auto budget — **0.75 × (RAM − 8 GiB)**; an availability-based clamp was tried and reverted, don't reintroduce it. More budget is NOT faster — past the memory cliff the cache fights the macOS compressor (measured 16 GB sweep: 6 GB optimum, 10 GB collapses).
- `architectures.py` is the registry of ~24 supported mlx-lm architectures. `archspec.py` (`ArchSpec`, `LayerContext`, `LayerRunner`) is the seam for architectures whose layers aren't uniform (gemma3/4) — extend via these protocols rather than per-arch hacks in the engine.
- `generate.py` holds the generation modes: greedy `generate`, `batched_generate` (B=8 batched decode), `speculative_generate` (deep-K draft-model spec), `ngram_speculative_generate` (adaptive, disables itself to zero when losing), plus eagle/tree variants (`tree_spec.py`, `eagle_drafter.py`). KV state can spill to disk via `kv_store.py`; `prefix_cache.py` caches prompt prefixes.
- `server.py` is the OpenAI-compatible HTTP server; `src/nunspark/webapp/` is the FastAPI web UI (engine pool + background jobs); `bench.py` produces the shareable bench reports in `docs/community-results.md`.
- `scripts/` contains one-off probes and benchmarks from past investigations (I/O warming, scatter strategies, spec acceptance); `docs/` holds the plan/gate write-ups those probes fed.

## Working process

- **Measure before building.** Every optimization starts as a bounded probe script in `scripts/` (cheap offline validation → micro-bench of candidate strategies → real-model A/B), and only ships if the gate passes. Performance A/Bs are interleaved control/experiment runs with the OS page cache flushed between runs (stream a large dummy file), token streams compared for identity, and results saved as JSON in `scripts/results/`. Never run timing benches concurrently with test suites or other heavy jobs — disk contention invalidates them.
- **Model orchestration for planned work.** Larger milestones (Plan 4/5 style) are executed by delegating to subagents: **Sonnet for mechanical work** (applying a spec'd design, writing tests to a stated invariant, running benches, doc updates) and **Opus for design-sensitive work** (new engine seams, cache/eviction policy, numerics-affecting changes, investigation write-ups). The orchestrating session stays in the loop: it writes the milestone spec and gate criteria before spawning, reviews each agent's diff against the gate, and never lets an agent commit, bump versions, or weaken a losslessness test. Small bounded probes and one-file fixes are done inline — spawning costs more than it saves there. Agents must run to completion (instruct them never to pause "waiting for results"); background task/agent completion notifications are SYSTEM events, not user input — they never constitute user approval for a next step.
- **`docs/backlog.md` is the ledger.** Decided-but-unscheduled work, gate outcomes, AND refuted ideas (with the measurement that killed them) live there — check it before proposing an optimization; several plausible ideas (decode I/O warming, batched demand loads, the availability clamp) are already measured dead. Larger effort gets a gated plan doc (`docs/plan*.md`, milestones with pass/fail gates).
- **The user owns commits and versioning.** Leave all work uncommitted on the current branch for their review; never `git commit`, never touch `version` in `pyproject.toml`. The user also runs their own CLI sweeps mid-session — treat their live numbers as the ground truth over flushed-probe numbers (probe conditions are adversarial), and re-check disk/git state before claiming it.
- CI (`.github/workflows/tests.yml`) deselects 4 known pre-existing failures (root causes in backlog #8) — a green run means "no new failures", and any newly failing test after a formula/default change probably encodes the old value somewhere (grep tests for the old constant).

## Invariants and design rules

- **Losslessness is the product.** Streamed forwards must stay bit-identical to full-load mlx-lm; speculative modes must emit exactly the target model's greedy tokens (fp16 near-tie flips across verify shapes are a known, characterized exception — don't claim byte-identity for spec on real large models). Tests like `test_engine_forward.py`, `test_garbage_drafter_invariant.py` enforce this — never weaken them to make an optimization pass.
- Prefer stock-mlx_lm, registry-wide mechanisms over per-architecture special cases; minimize target weight reads (disk I/O per token is the metric that matters).
- **Never bump the version** in `pyproject.toml` — the user owns versioning and may edit files or disk state mid-session; re-check state before claiming it.
- Performance claims in README/docs come from measured gates (see `docs/plan*-gate-summary.md`) — don't invent or extrapolate numbers.
