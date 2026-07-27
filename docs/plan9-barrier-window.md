# Plan 9 — Windowed per-layer eval (recover the layer_sync barrier) (backlog #16 verify → Plan 9)

Status: M0 FAILED the ≥10% default-on gate 2026-07-27 (clean run: best W=3 =
+7.61% median vs W=1; streams identical, zero churn). Decision (user, 2026-07-27):
SHIP OPT-IN. **M1-lite IMPLEMENTED 2026-07-27** — `StreamingEngine(eval_window=1)`
default (= today, exact), W>1 drains every W-th layer with a wire-proximity memory
guard; CLI `--eval-window`, serve plumbing, bit-identity tests. Default stays W=1
(off); no default-on flip (M0 under bar). Scoped 2026-07-27 off the #16
barrier-verification probes.

## Motivation (all measured) and the central catch

The #16 verification (backlog #16, scripts/results/barrier_cost_probe.json +
barrier_headroom_probe.json) established that decode's barrier buckets are REAL
host↔device round-trips, not drained compute — but only PARTLY recoverable:

- Probe 1: sync FLOOR = 34.4 ms/token = 61% of the 55.9 ms router+layer_sync
  bucket (116.6 mx.eval/token @ 0.219 ms + 48 tolist @ 0.181 ms). The round-trips
  are real.
- Probe 2 (clean, Llama-1B resident, bit-identical A/B via the `_fully_resident`
  toggle): the per-layer `mx.eval(h)` barrier is worth **0.268 ms/layer** ≈ one
  eval round-trip. Projected to the 30B (×48) = **~12.9 ms/token ≈ 8.7%
  end-to-end** — of which #16 had attributed 28 ms to layer_sync, so only ~half
  that bucket is recoverable; the rest is compute draining at the barrier.

**THE CATCH (mechanism dig, 2026-07-27):** that 8.7% is a RESIDENCY CEILING, not a
streaming-realizable number. The per-layer `mx.eval(h)` is MEMORY-LOAD-BEARING:
MLX refcounts a lazy graph's inputs, so deferring the eval keeps every layer's
weights alive in the pending graph, and `_evict_locked` + `mx.clear_cache`
(piece_cache.py) cannot reclaim a buffer the graph still references. The barrier
is precisely what lets layer N's weights drop before layer N+1 loads — which is
why it is skipped ONLY under `_fully_resident` (when the whole model fits and
nothing is evicted anyway; that is the regime Probe 2 measured on the 1B). On the
STREAMING 30B (10 GB budget on 16 GB), dropping the barrier accumulates the whole
token's weight set → exceeds budget → macOS-compressor churn (#15) → net SLOWER.

So the lever is not "drop the barrier"; it is "drain LESS OFTEN without letting the
live weight set exceed budget" — a WINDOWED pipeline: keep W layers' graphs in
flight (their weights pinned by refcount), eval when the window advances. Recovers
`(W-1)/W · 12.9 ms`, but peak live weights ≈ W layers, so W is bounded by memory
slack under the wire (10 GB budget on a 11.84 GB wire ≈ ~1.8 GB slack; a 30B-4bit
layer's core+8 experts is order hundreds of MB, so W is small — 2–4). Realistic
recover therefore ~4–7% end-to-end — quite possibly UNDER the 10% gate. Whether
ANY W clears the gate WITHOUT triggering churn is the entire question, and it is
cheap to answer before building anything.

**Key enabler:** MLX refcounting already pins graph-referenced buffers, so a
windowed-eval PROTOTYPE (eval every W-th layer instead of every layer) is correct
by construction — bit-identical, no explicit pin API needed to MEASURE the
tradeoff. Explicit eviction-pinning is only worth building if windowed-eval clears
the gate but the naive refcount version leaves memory on the table.

## Design constraints (fixed — do not relax)

1. **Losslessness is the product.** Windowed eval is scheduling-only (mx.eval
   never changes numerics — the engine's own _sync_layer docstring); output must
   be bit-identical to W=1 (today's per-layer eval), fp16 and 4-bit, every arch.
   Never weaken test_engine_forward / test_garbage_drafter_invariant.
2. **Peak memory is the hard constraint, not a nice-to-have.** The win is bounded
   by how many layers' weights fit the slack under the wire. Any W that pushes
   compression out of the clean band (#15: clean wired 30B compresses ~15–20 GB;
   a run >30 GB or a tok/s collapse to ~3 is churn) is INVALID, not faster — the
   #15 validity check is mandatory on every timed run.
3. **Opt-in until the gate passes.** Prototype via probe monkeypatch (M0, no
   engine change); if it ships, `StreamingEngine(eval_window=1)` default = today's
   behavior, flipped only on a passed real-model gate.
4. **Refcount pinning is implicit and sufficient to MEASURE.** Do not build an
   explicit cache pin/unpin API for M0/M1 — the pending graph already holds the
   weights; the cache simply can't free them until the window drains. Explicit
   pinning is a separate, later milestone gated on M2 showing the naive version
   under-delivers.
5. House rules bind all agents: never `git commit`, never touch `version` in
   pyproject.toml, plain `uv run`, leave work uncommitted for review.

## Milestones (each has a pass/fail gate)

**M0 — windowed-eval realizability probe (go/no-go; the whole plan hinges here).**
`scripts/windowed_eval_probe.py`: monkeypatch `_sync_layer` to eval only every
W-th layer (a token-scoped counter; the trailing partial window is drained by the
logits/sample eval), on the REAL streaming 30B @ 10 GB wired. Sweep W ∈ {1,2,3,4}.
For each W: flushed ABBA-interleaved vs W=1, 200 greedy tokens, token streams
byte-identical across ALL runs, per-run `compressed_gb_during_run` (#15 validity —
drop/re-run any run whose compression leaves the clean band). Report tok/s(W),
median delta vs W=1, and peak/compression per W.
**GATE: some W delivers ≥10% median tok/s over W=1 AND stays in the clean
compression band (no churn) AND streams bit-identical. If the best in-band W is
<10%, Plan 9 STOPS here — recorded dead in #16 with the W-sweep (the barrier is
real but not streaming-realizable to the bar), and the residency-only ceiling is
noted as unreachable under streaming.** Run: `uv run python
scripts/windowed_eval_probe.py ./packed/qwen3-30b`.

**FIRST PASS 2026-07-27 — PROMISING but the gate call is warmup-biased; probe
hardened, clean re-run pending.** W-sweep {1,2,3,4}, streams identical, **zero
compression at every W (the memory-load-bearing risk did NOT bind at W≤4 on the
10 GB budget)** — a clean peak at **W=3**. As-run it reported W=3 = +14.08% vs W=1
(7.60→8.67 tok/s), which the win-exceeds-the-8.7%-ceiling makes mechanistic sense
for: windowing also recovers CROSS-LAYER OVERLAP (layer N GPU compute overlapping
N+1's router-tolist + scatter + fetch) that the isolated-barrier 1B Probe 2 could
not see. BUT the two W=1 runs were 7.147 (run 1) vs 8.048 (run 8): run 1 was the
global cold start (purge no-op'd → only run 1 read cold disk), and the symmetric
order put W=1 at both ends, dragging its mean down and INFLATING every delta.
Warm-only, best W ≈ +7–8% — borderline UNDER the gate. So the true figure is
8–14% and undecided. FIX (applied): the probe now discards a warmup run first
(no timed run is the cold start), does REPS=3 runs/W, and aggregates by MEDIAN.
**Re-run before proceeding to M1** — the gate turns on whether the clean number is
≥10%.

**CLEAN RESULT 2026-07-27 — GATE FAILED.** Medians of 3: W=1 8.025, W=2 8.363
(+4.2%), **W=3 8.636 (+7.61%, best)**, W=4 8.426 (+5.0%); streams identical; zero
compression at every W (the memory-load-bearing risk never bound at W≤4 on 10 GB).
The median correctly rejected another cold-start outlier (raw W=1 run 1 = 5.13
tok/s). **Best in-band W=3 = +7.61% < the ≥10% bar → Plan 9 STOPS as a default-on
plan.** This bookends the verification cleanly: Probe 2's isolated ceiling 8.7% →
streaming-realizable 7.6% (just under it; the cross-layer-overlap bonus was real
but only enough to nearly reach, not exceed, the isolated ceiling). The
first-pass +14% was the warmup artifact. NOTE: +7.6% is real, lossless, ~5-line,
churn-free at W≤4 — a legit OPT-IN candidate (below), just not a default-on win.

**M1-lite — opt-in engine implementation behind `eval_window`. IMPLEMENTED
2026-07-27 (uncommitted).** M0 failed the default-on gate but the user chose to
ship the +7.6% opt-in. `_sync_layer` now evals every W-th layer (`_sync_counter %
_eval_window == 0`), W=1 = today's per-layer eval EXACTLY (early-returns before any
counter logic, byte-untouched). Because the barrier is memory-load-bearing, a
WIRE-PROXIMITY GUARD (`_eval_window_mem_ceiling` = 0.90·wired_limit) forces a drain
when `_active_memory()` nears the wire — windowing can never push a tighter budget
/ bigger model into churn (the M0 no-churn result was at W≤4/10 GB; the guard
generalizes it). `_active_memory()` reads MLX's active-memory counter (cheap, not
a device sync; 0-fallback disables the guard on APIs that lack it). One change
covers decode + prefill + tree (all route through `_sync_layer`). `eval_window` in
prefetch_stats. CLI `--eval-window W` on generate + serve (→ run_server →
build_server). Tests: tests/test_eval_window.py — W=3 bit-identical to W=1 across
prefill+decode on qwen3_moe + gpt_oss + DENSE llama × fp16/4-bit, stat-reports-W,
and default-is-1. test_cli serve-dispatch dicts updated (`eval_window: 1`).
DEFAULT STAYS W=1 (off) — no default-on flip, since M0 is under the 10% bar.
**GATE (pending run): full suite green (same 4 known #8 failures); test_eval_window
bit-identical.** Run: `uv run pytest tests/test_eval_window.py` then `uv run pytest`.

**M2 — real-model flushed A/B gate.**
Qwen3-30B @ 10 GB wired, 200 greedy tokens, ABBA control(W=1)/experiment(blessed
W), page cache flushed between runs, per-run compression as #15 validity, streams
byte-identical. Re-run the #16 attribution probe under the window to confirm
layer_sync actually shrank. **GATE: ≥10% median decode tok/s AND no pair worse
than −5% (the house gate shape). Pass → default-on + docs. Fail → dead in #16.**

**M3 — (optional stack) explicit eviction-pinning.**
Only if M2 passes but profiling shows the refcount-implicit version leaving memory
unreclaimed (peak higher than W layers warrants, forcing a smaller W than slack
allows). Add a dynamic cache pin/unpin so weights are released the instant a
layer drains rather than when the window rolls. Same gate. Cache-lifetime change
(design-sensitive — Opus), spec separately if reached.

## Notes

- Re-derive the memory slack per machine/budget before assuming a W: on a tighter
  budget or bigger model, W collapses to 1 and Plan 9 is inapplicable there.
  Report slack + achieved W in every result.
- router_sync (the tolist) stays crossed off — irreducible; selective streaming
  must read routing on host to fetch (#9 lookahead died avoiding it). This plan
  touches only layer_sync.
- If M0 dies, the honest conclusion is "decode on this hardware is near the
  compute+memory floor; the remaining decode levers are model/quant choices (a
  smaller or lower-bit model), not engine scheduling." Record and stop chasing
  per-token decode scheduling.
