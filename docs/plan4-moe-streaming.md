# Plan 4 — MoE Expert Streaming: from "works" to "winner"

**Thesis.** Dense streaming is pinned at `SSD_bandwidth / model_bytes` per verify sweep.
MoE models (Qwen3-30B-A3B, GPT-OSS-20B/120B) activate a small fraction of weights per
token, so bytes-per-token collapses — *if* the runtime caches, prefetches, and verifies
at expert granularity. The selective-MoE foundation already exists (packer splits
core + per-expert pieces; `_moe_layer_forward` loads only fired experts). This plan
closes the four gaps between "selective loading works" and "GPT-OSS-120B is usable on
a 16 GB MacBook":

| # | Gap | Where |
|---|-----|-------|
| G1 | Expert reads are serial: router runs, *then* fired experts load — no overlap with compute or prefetch | `engine.py:_moe_layer_forward` |
| G2 | MRU eviction is tuned for the dense cyclic scan; expert pieces (sparse, skewed reuse) are explicitly not handled | `piece_cache.py:_evict_locked` |
| G3 | `tree_forward` reassembles ALL experts per MoE layer — spec-decode trees and selective MoE don't compose | `engine.py:tree_forward` |
| G4 | No measurement or headline result: expert locality never profiled; GPT-OSS-120B never run | — |

**Orchestration model.** Fable orchestrates; implementers are subagents:
- **Sonnet** — well-specified mechanical work: instrumentation, scripts, tests, plumbing,
  benchmark runs, docs.
- **Opus** — design-sensitive work: cache-policy changes under the `PieceCache` lock,
  prefetch pipeline restructuring, anything touching the verify loop's correctness.

Every milestone ends with a **gate**: a measured number or passing test the orchestrator
checks before opening the next milestone. No milestone starts until its inputs' gates pass.
Working branch: `version2.0` (current) or a `plan4-*` branch off it — decide at M1 start.

**Standing rules for all implementer prompts**
- Python 3.11 via `uv venv --python 3.11`; run tests with plain `uv run` (never `--active`).
- Bit-identical output is the invariant for engine changes: selective paths must match
  load-everything paths exactly (existing tests assert this — keep them green).
- Stock-mlx_lm compatibility over arch hacks (registry-wide, per design principles).
- Each task lands as a small reviewed diff with a test; benchmarks write JSON to
  `scripts/results/` so runs are comparable across milestones.

---

## M1 — Measure expert locality (the data that drives everything)

*Implementer: Sonnet. No engine behavior changes — instrumentation + analysis only.*

1. **Expert-trace instrumentation.** Add an opt-in trace hook to `_moe_layer_forward`
   (env var or engine kwarg): per token, per layer, record the fired expert set.
   Write compact JSONL. Zero overhead when off.
2. **Cache counters split by piece class.** `PieceCache` currently counts global
   hits/misses. Split counters by piece kind (dense layer / MoE core / expert) —
   classify from the pid prefix, no API break.
3. **Trace-analysis script** (`scripts/analyze_expert_trace.py`) computing, per layer
   and overall: expert firing frequency distribution (how skewed?), token-to-token
   Jaccard overlap of fired sets, reuse distance histogram, working-set-vs-cache-size
   curve (simulate cache sizes offline from the trace — cheap, no reruns), and
   fired-union size as a function of verify-window length K and tree size B×d
   (this decides G3's worth before writing any engine code).
4. **Baseline runs.** Qwen3-30B-A3B-4bit with the existing engine: greedy decode and
   spec decode (Qwen3-0.6B draft, K=24), ~500 tokens each over 3 diverse prompts
   (code / prose / reasoning — reuse report.md workloads). Record tok/s,
   bytes-read-per-token (core vs expert), hit rates, peak memory.

**Gate G-M1:** a short markdown summary with the locality numbers, and a stated
decision for each downstream milestone: expected hit-rate ceiling for M2, expected
overlap win for M3, fired-union curve verdict for M4. If token-to-token expert overlap
is high (papers suggest 60–90% for these models), M2/M3 proceed as designed; if it's
low, M3's temporal prefetch is demoted and router-probe prefetch (M3c) promoted.

---

## M2 — Expert-aware cache policy

*Implementer: Opus (lock-protected eviction logic; concurrency-sensitive).*

1. **Two-region budget in `PieceCache`.** Keep MRU exactly as-is for dense/core pieces
   (it's correct for the cyclic scan). Add an expert region governed by
   frequency-weighted eviction (LFU with decay, or segmented-LRU — Opus picks based on
   M1's reuse-distance histogram and justifies in the PR). Single byte budget with a
   configurable split (`--expert-cache-frac`, default from M1's working-set curve).
2. **Hot-expert pinning (optional, data-permitting).** If M1 shows a stable hot set
   (e.g. 10% of experts serve 50% of firings), pre-warm and pin them at engine start.
3. **Bench vs M1 baseline**, same prompts/settings. Sweep 2–3 budget splits.

**Gate G-M2:** expert-piece hit rate improves materially over M1 baseline at equal
total budget, tok/s does not regress on dense models (run one dense regression bench),
and all existing tests pass. Target from M1's simulation; if the measured win is <5%
tok/s, record why and move on — M3 is the bigger lever.

---

## M3 — Kill the router stall: expert prefetch

*Implementer: Opus. The novel systems contribution; staged from safe to speculative.*

Today the per-layer timeline is: load core → attention → router → **stall: load fired
experts** → expert FFN. Three escalating attacks, each gated by measurement:

- **M3a — Temporal prefetch.** When the window-prefetcher warms layer L+1's core piece,
  also enqueue the experts that layer L+1 fired for the *previous* token (or the union
  over the previous verify window). Uses only existing trace state; no model math.
  Cheapest possible version of the idea and, if M1's overlap number is high, captures
  most of the value.
- **M3b — Priority-aware prefetch queue.** Expert prefetches must not delay the core
  pieces the scan needs unconditionally: make the prefetch queue two-level
  (cores first, speculative experts fill remaining I/O slack). Also ensure a mid-flight
  speculative expert load never blocks `get()` for a demand piece.
- **M3c — Router-probe lookahead (research-y; only if M3a's residual stall is big).**
  At layer L, after computing h, probe layer L+1's router with the *pre-attention*
  hidden state (router weights are in the already-tiny core piece — prefetching L+1's
  core early makes them available). Measure probe-vs-actual top-k overlap offline from
  M1 traces first; implement only if overlap ≥ ~70%. Probed experts join the
  speculative queue tier. A miss costs nothing but wasted read bandwidth — correctness
  is untouched because the real router still decides.

**Gate G-M3:** per-layer stall time (instrument: wall time between router output and
expert weights resident) drops by ≥50% vs M1 baseline, tok/s improves end-to-end,
bit-identical output vs no-prefetch run. Report wasted-prefetch bytes (speculative
reads never used) — must stay under ~20% of expert bytes read.

---

## M4 — Selective experts in batched/tree verification

*Implementer: Sonnet, with the M1 fired-union data as the spec. Conditional milestone.*

Only if M1's fired-union curve shows tree/window verification fires well under all
experts (e.g. union < 60% for the tree shapes actually used):

1. Rework the `tree_forward` MoE path: run core + router on the batch, take the union
   of fired experts across all paths/positions, `_scatter_experts` that union only.
   (The window-verify path through `forward()` already does this naturally — verify,
   add a test asserting it.)
2. Test: tree verify output bit-identical to the load-all path on a small MoE model.
3. Bench spec-decode (linear K sweep and one tree shape) on Qwen3-30B-A3B vs M3.

**Gate G-M4:** bit-identical tree verification; expert bytes per tree sweep reduced in
line with M1's union prediction; spec-decode tok/s ≥ M3's greedy tok/s × acceptance-derived
expectation.

---

## M5 — The headline: GPT-OSS-120B on 16 GB

*Implementer: Sonnet for packing/benchmark plumbing; Opus on call for whatever breaks.*

1. **Preflight (orchestrator + user):** ~65 GB free disk for the packed 120B
   (MXFP4 ≈ 60 GB) plus HF download staging — confirm before starting; consider an
   external SSD for the packed dir (also a striping data point).
2. **Pack & smoke-test GPT-OSS-20B first** (~11 GB) — cheap end-to-end validation of
   the gpt_oss selective path with M2+M3 active, on-machine.
3. **Draft-model selection experiment.** Spec decode needs a shared tokenizer; GPT-OSS
   has no small sibling. Candidates, in order: (a) prompt-lookup / n-gram drafting
   (tokenizer-free, zero memory — likely the winner for code/RAG prompts),
   (b) GPT-OSS-20B as a *streamed* draft (shares the SSD — measure whether it pays),
   (c) greedy no-spec as the honest floor. Bench each on 2 prompts before the full run.
4. **Pack GPT-OSS-120B, full benchmark suite:** tok/s (greedy + best draft mode),
   bytes/token, expert hit rate, peak unified memory, resident-weights figure.
   The demo artifact: one chart — 120B answering at N tok/s, memory graph ≈ 3–4 GB.
5. **Qwen3-30B-A3B final numbers** with everything on, against report.md's dense-32B
   baselines (same prompts).

**Gate G-M5:** GPT-OSS-120B generates coherent output end-to-end on the 16 GB M4 with
peak memory < 8 GB and tok/s ≥ 1.0 greedy (stretch: ≥ 3 with drafting). Qwen3-30B-A3B
beats the dense-32B exact-mode number (1.13 tok/s) by ≥ 3×.

---

## M6 — Consolidate: report, web UI, docs

*Implementer: Sonnet.*

1. Extend `report.md` (or a `report-moe.md`) with the M1–M5 measurements: the
   bytes-per-token story, cache/prefetch ablations (each milestone's gate numbers form
   the ablation table for free), the 120B headline.
2. Wire MoE models through the web UI path (`engine_pool` budget knobs for the expert
   cache split); verify `nunspark web` serves Qwen3-30B-A3B.
3. README: move MoE from "not built / deferred" to a first-class section with the
   pack/generate commands and the headline chart.

**Gate G-M6:** README quickstart for Qwen3-30B-A3B works copy-paste on a clean checkout.

---

## Sequencing & risk

```
M1 ──► M2 ──► M3a ──► M3b ──► (M3c?) ──► M5 ──► M6
        └────► M4 (conditional, parallel to M3b/M3c)
```

- M1 is cheap and de-risks everything: every later design choice cites an M1 number.
- M2 and M3 are the engine's real work; M3a alone may capture most of the win.
- M4 runs in parallel once M1's union data is in (Sonnet lane while Opus is on M3).
- Biggest risks: (a) expert temporal locality lower than literature suggests → M3c
  becomes load-bearing; (b) disk space / download time for the 120B → M5 preflight;
  (c) no good 120B draft → prompt-lookup fallback is the mitigation, and greedy-only
  is still a headline if tok/s ≥ 1.
