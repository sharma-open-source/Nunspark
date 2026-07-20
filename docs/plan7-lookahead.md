# Plan 7 — Online router-lookahead expert prefetch (backlog #9 / #13-C1)

Status: COMPLETE 2026-07-20. M1 + M2 shipped (flag opt-in); M3 gate ran and
**FAILED as configured** — `--lookahead` stays opt-in/off. See "M3 outcome"
at the end of this doc and scripts/results/lookahead_ab.json.

## Motivation (all measured)

- Decode expert misses are serial demand-faults at ~0.45 GB/s effective; the
  same bytes read as known-in-advance bulk reads hit 0.73–1.36 GB/s
  (backlog #1 warm_bulk measurements). Within-layer parallelism is dead —
  ~0.36 misses/layer, nothing to batch (backlog #9, decode_bulkwarm A/B).
  The only queue depth available at decode is CROSS-LAYER.
- Offline go/no-go PASSED (scripts/results/router_lookahead_probe.json,
  Qwen3-30B): layer L+1's router evaluated on layer L's output hidden state
  recalls the true top-8 at 98.3% inside a top-12 window (90.4% fully
  contained); k=2 top-12 still 96.0%. Independently corroborated on GLM-5.2's
  256-expert geometry by Colibrì (71.6% one layer ahead).
- Upside bound (backlog #10 attribution): expert demand-load wait is 17–30%
  of decode post-scatter-fix; at 6 GB/30B ≈ 41.6 misses × 3.0 ms ≈ 125
  ms/token of hideable stall. Realistic win 1.1–1.5×, not more. The gate is
  sized accordingly.

## Design constraints (fixed by the orchestrator — do not relax)

1. **Losslessness untouched.** Prediction is read-only math on the residual
   stream; the layer's own routing/compute path is byte-for-byte unchanged.
   Token streams with the feature ON must be identical to OFF, always.
2. **Opt-in flag until the M3 gate passes:** `StreamingEngine(...,
   lookahead_prefetch=False)`. Default flips only on a passed gate.
3. **Never block the compute path.** The prediction for layer L+k needs
   L+k's core piece (post_attention_layernorm weight + router weights). If
   that core is not ALREADY resident, skip the prediction for that layer —
   a non-blocking peek on PieceCache, never a blocking get(). No new
   threads for the prediction math itself.
4. **Reuse the M3a speculative staging tier** (`cache.prefetch(pids,
   speculative=True)`): predicted pieces land in staging, capped, dedup'd
   against resident/in-flight, waste-counted by the existing counters. No
   new cache regions. The existing epoch/grace semantics already cover the
   same-pass consumption window.
5. **Decode passes only** (batch_tokens == 1). Multi-token passes keep the
   shipped temporal prefetch + warm_bulk path unchanged. On decode passes
   the lookahead REPLACES the temporal spec issue (a strictly stronger
   prior); the temporal `_fired_history` recording stays as-is for
   multi-token consumers.
6. **Registry-wide with graceful no-op.** Prediction replicates the router
   forward (norm + gate matmul + spec.moe_route top-k) from the core piece
   dict, per arch. Must work for qwen3_moe and gpt_oss (the bench models).
   Any arch whose router cannot be replicated from the core dict (bias
   layouts, deepseek MoEGate correction bias, etc.) must silently skip —
   never guess, never crash, never alter output.
7. Defaults from the probe: lookahead depth k=1, widen top-12. Both are
   engine parameters (`lookahead_depth`, `lookahead_topn`) so M3 can sweep
   without code changes.
8. House rules bind all agents: never `git commit`, never touch `version`
   in pyproject.toml, never weaken a losslessness test, plain `uv run`.

## Milestones

### M1 — engine seam (Opus; design-sensitive)

Deliverables:
- `PieceCache.peek(pid)` (or equivalent): non-blocking resident lookup that
  never counts as hit/miss, never reorders LRU/MRU, never triggers a load.
- Prediction + issue in the decode path of `_moe_attn_and_mix` (or a helper
  called from it), honoring constraints 1–7. Quantized and fp16 router
  weights both handled (probe's `_predict_topn` is the reference for the
  math; note it currently handles qwen3-style cores only).
- Engine counters: `lookahead_issued`, `lookahead_skipped_core_missing`
  (staging used/wasted counters already exist).
- Tests (tiny seeded models, no downloads):
  a. Bit-identity: greedy token stream flag-ON == flag-OFF on the tiny MoE
     models (qwen3_moe AND gpt_oss AND glm_moe_dsa — the last should
     exercise the graceful-skip path or work, either is a pass, but which
     one happened must be asserted explicitly).
  b. Non-blocking: a lookahead with the next core evicted issues nothing
     and increments the skip counter.
  c. Prediction correctness: on a tiny model with the true fired set known,
     the predicted top-N contains what the probe math says it contains
     (guards against silent math drift vs `_predict_topn`).

Gate M1: all new tests pass + full suite green (355+ passed, same 4 known
failures) + orchestrator diff review confirms constraints 1–7.

### M2 — plumbing (Sonnet; mechanical)

- CLI: `--lookahead` on `nunspark generate` / `bench` / `serve` (+ web UI
  advanced flag), threaded to the engine; default off.
- `--metrics` / bench report lines for the new counters.
- Doc touches: README flag row (NO perf claims), backlog cross-links.

Gate M2: suite green; flags reach the engine (test asserting plumb-through).

### M3 — real-model A/B gate (orchestrator, inline)

Interleaved flushed A/B on packed/qwen3-30b at 6 GB budget (the measured
16 GB optimum), 150–200 greedy decode tokens, page cache flushed between
runs, no concurrent heavy jobs; plus one TTFT check (prompt prefill must not
regress — lookahead is decode-only so any change is a bug).

Gate M3 (to flip the default ON):
- Token streams byte-identical control vs lookahead in every run.
- Median decode tok/s improvement >= 10%; no run slower than control by
  more than noise (>5%).
- staging waste bytes bounded (< 15% of bytes issued — probe says ~4/12
  predictions are extra; most should dedup against resident experts).

FAIL handling: keep the flag opt-in, record the numbers in backlog #9, done
— the feature is still lossless and harmless off.

## Non-goals

- k>=2 pipelining, predicted-set warm_bulk fusion, GLM-5.2 real-model
  numbers, batched-decode integration — all only after M3 reads out.

## M3 outcome (measured 2026-07-20 — gate FAIL, flag stays opt-in)

Qwen3-30B, 6 GB budget, 150 greedy decode tokens, 3 interleaved flushed
pairs per configuration (scripts/lookahead_ab_probe.py,
scripts/results/lookahead_ab.json). Token streams byte-identical to control
in all 6 pairs — losslessness held everywhere.

- **topn=12 (probe-recall optimum): FAIL.** Median 2.171 → 2.255 (+3.9%),
  two of three pairs SLOWER (−9.5%, −6.5%). Cause: 2.6× read amplification
  (92 → 238 MB/token, 19.9 GB staged-then-wasted bytes; only 42% of
  speculative loads ever demanded). The 4 never-fired extra pieces per layer
  plus wrong guesses saturate the single materialize worker and starve
  demand loads.
- **topn=8: closer, still FAIL.** Median 2.561 → 2.777 (+8.4% < 10%), one
  pair −8.0% (> the 5% noise bound). Speculative used 76%, amplification
  1.28×, waste 3.5 GB.
- **The mechanism itself works:** expert stall dropped ~25–30% in every
  lookahead run of both sweeps (e.g. 17.1 → 12.0 s per 150 tokens), and the
  lookahead arm's run-to-run spread is visibly tighter than control's.
  The problem is that control variance at 6 GB (2.40–3.02 tok/s across six
  flushed runs) is the same order as the win, and the residual read
  amplification pays back most of the hidden stall.

Refinement candidates if this is ever revisited (record here, do not build
without a new gate): (a) issue only predictions ABSENT from cache whose
predicted rank is inside the true top-k count (kill the top-N overhang
entirely — at topn=8 waste is pure misprediction, 24%); (b) stop counting
speculative issues into `misses` so accounting stays comparable; (c) a
second materialize worker (or warm-only path) for the speculative tier so
spec loads can never head-of-line-block a demand miss; (d) gate on measured
per-run expert-stall fraction — the feature can only win its stall share
(17–30% of decode), so machines/budgets with higher stall shares (bigger
models, colder caches) should see more than this 16 GB testbed can.
User live sweeps with `--lookahead` on other budgets/models remain the
ground truth over these probe numbers.
