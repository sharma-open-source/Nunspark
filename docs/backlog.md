# Backlog — queued optimizations and follow-ups

Ideas we have decided are worth doing but are not scheduled into a plan yet.
Each entry records the motivation and enough design detail to start cold.

## 1. Prefill / time-to-first-token optimization (DONE 2026-07-15 — shipped)

**Outcome.** Implemented as `PieceCache.warm_bulk`: right after a layer's router
fires on a multi-token pass, the complete fired-expert piece list is raw-read by
an 8-thread pool to populate the OS page cache while `_scatter_experts`'s serial
`get()` loop runs — converting single-threaded mmap demand-faults (~360–510 MB/s
measured) into queue-deep reads (0.73–1.36 GB/s measured; SSD sequential
reference 2.4 GB/s). Gated on multi-token passes only, so single-token greedy
decode is untouched (Phase-1's "warming is net-negative for decode" verdict
still honored). Measured on Qwen3-30B-A3B, 273-token prompt, 8 GB budget,
interleaved A/B with page-cache flushes between runs
(scripts/results/prefill_bulkwarm_ab.json, probe: scripts/prefill_probe.py):
prefill **21.6–30.8 s → 8.1–15.0 s** (~2× median, up to 3.8×), identical bytes,
misses, and first token (lossless); 200-token greedy decode after the change
1.62 tok/s vs 1.65 before, identical cache counters. Expected to transfer
directly to gpt-oss-120b (same serial-fault bottleneck, 57–95 s TTFT) —
community verification pending.

Original entry follows.

**Addendum (2026-07-15): prompt-length memory safety.** The "not worth doing:
chunked prefill" verdict below was I/O-scoped and stands; chunked prefill was
nevertheless shipped as a MEMORY-safety fix after a field incident: a binary PDF
uploaded to the web UI decoded to a garbage mega-prompt, whose single-pass
prefill (unbounded activation + KV memory) swap-stormed a 16 GB Mac until the
macOS watchdog killed WindowServer. Shipped: (a) `generate._prefill` — prompt
processed in `--prefill-chunk`-sized windows (default 1024), each window's
graph evaluated before the next, bit-identical incl. gpt-oss sliding-window
rotation (tests/test_chunked_prefill.py); gemma4-MTP left unchunked (its
assistant needs whole-prompt `target_kv_states`). Measured cost of chunking on
Qwen3-30B (273 tokens, chunk=96, 3 windows): 2.6× expert re-reads across
windows — the trade is deliberate; prompts <= chunk are single-pass and free.
(b) Web UI guards in runner.py: binary/PDF uploads rejected by magic
bytes/NUL/replacement-ratio; prompt tokens capped against unified RAM from a
per-token KV estimate (override: `advanced.max_prompt_tokens`), failing loudly
instead of truncating.

**Problem.** 120B prefill is 57–95 s for ~100-token prompts on 16 GB. During
prefill every prompt token fires its own experts, so the per-layer union touches a
large fraction of all 128 experts — tens of GB of reads before the first token.
The reads go through mmap demand-faulting at ~300 MB/s effective (Phase 1 I/O
measurement); 30 GB / 300 MB/s ≈ 100 s explains the whole TTFT.

**Key insight.** The Phase 1 verdict "I/O warming is not the lever" applies to
*decode* (random, unpredictable access). Prefill is the opposite: after layer L's
router runs we know the complete list of expert pieces layer L needs *before
touching any of them* — a known-in-advance bulk read pattern.

**Plan (bounded experiment):**
1. Instrument prefill: per-layer wall time split into router / expert-read / compute.
2. After the router, issue the layer's full fired-expert piece list as one batched,
   deep-queue read — `madvise(MADV_WILLNEED)` on the union, or explicit `preadv`
   into the PieceCache — instead of fault-as-you-go. Target: SSD sequential-class
   bandwidth (multi-GB/s) instead of 300 MB/s.
3. Measure TTFT on gpt-oss-120b and Qwen3-30B. Realistic upside: 57–95 s → 15–30 s.

**Not worth doing:** chunked prefill (caps peak memory, does not reduce total unique
reads); cross-layer expert prediction (router output not knowable early).

**Stacks with:** persistent prompt-prefix KV cache (below).

## 2. Persistent prompt-prefix KV cache (MEDIUM)

Chat-style use re-sends an identical system prompt / harmony preamble every request.
The web UI already reuses a single in-memory prefix slot; extend it to save/load the
prefix KV to disk (mlx_lm `save_prompt_cache` precedent) so later sessions skip
prefill of the shared prefix entirely → near-zero TTFT on reuse. Cheap; stacks
with backlog #1.

## 3. Reproduce the Llama-3.3-70B community spec anomaly (MEDIUM)

Community M1 Max/64 GB report (docs/community-results.md): code spec 4.57 vs 3.39
greedy (M=14.29) but prose/reasoning spec *slower* than greedy despite M=4.00/7.69.
For a dense model a K-token verify pass costs ~one greedy token of weight I/O, so
M=4 should win. Also unexplained: M values that high with the default Qwen3-0.6B
draft, whose tokenizer differs from Llama-3.3's. Reproduce locally on a smaller
dense model (or rent time on a big machine), check SpecStats accounting, and check
whether the tokenizer-mismatch warning fired.

## 4. Migrate remaining spec paths off `kv.truncate` rollback (LOW — latent)

`eagle_speculative_generate`, `adaptive_speculative_generate`, and
`gemma4_mtp_speculative_generate` still roll back with `kv.truncate`, which violates
the RotatingKVCache contract once rotated (`is_trimmable()` False). Not reachable
from `nunspark bench`; only affects sliding-window targets on those paths. Migrate
to `engine.verify_forward` + `commit_verified` like `speculative_generate` /
`ngram_speculative_generate` (0.5.0). gemma4-MTP also slices `target_kv_states` and
needs its own design pass.

## 5. Adaptive spec-K (or auto-disable) for MoE targets (LOW)

Both community (gpt-oss-120b, K=24, M≈1.2) and local (K=16, M=1.30) data show the
MoE verify-pass expert-union tax makes long sweeps lose: per-pass expert union grows
~linearly with K while acceptance does not. Options: default `--no-spec` for MoE
manifests, cap K for MoE (K<=4–8), or adapt K online from measured M and MB/token.
DATA COMPLETE (m5_120b_ngram_bench.json): K=16 n-gram spec loses 2–4x on all three
workloads (M 1.15–1.39, bytes/token 3–5x greedy, expert hit halves). Recommendation:
`nunspark bench` should default the spec arm OFF (or to a small K) when the manifest
has experts, and print why. Small-K sweep (K=2–4) is the remaining open question —
at M~1.3 with K=3 the union tax shrinks ~5x, could break even.

**Refinement (2026-07-16, community Qwen3-235B-A22B on M3 Ultra 96 GB):** first MoE
spec WIN — reasoning 0.58 vs 0.30 greedy (M=11.11) with the default Qwen3-0.6B draft,
which shares the Qwen3 tokenizer. Same run shows M≈4 is break-even (code M=4.00 and
prose M=3.85 both ~par with greedy) despite a 2–3x MB/token union tax. So the rule is
conditional, not blanket: default spec OFF for MoE when the draft tokenizer mismatches
the target (or for n-gram at large K); keep it available — and maybe ON — when
tokenizer-matched, since high-M workloads (reasoning) win ~2x. An adaptive policy
could watch measured M for a few sweeps and disable spec if M stays below ~K/6.

**Small-K question CLOSED (2026-07-24, wired baseline — GATE FAIL at every K;
scripts/spec_breakeven_probe.py, scripts/results/spec_breakeven_wired.json).**
Qwen3-30B + matched Qwen3-0.6B draft, greedy @ 10 GB vs spec @ 9 GB (the draft
must fit under the 11.84 GB wire), K = 4/8/16, 300 tokens, flushed interleaved
rounds. Against the clean wired greedy baseline (7.96; today's 8 prior controls
7.98–8.15): **K=4 −22% best-run (6.19), K=8 −47%, K=16 −59%.** The failure is
structural on a streamed MoE: measured verify-pass cost = 4.6×/6.7×/8.2× a
greedy token at K=4/8/16 (union tax — 12–22 expert misses/token vs greedy's
4.6, 30–56 MB/token vs 11.7) while M = 2.54/2.89/3.33 (deterministic across
rounds; acceptance 39/24/15%). Break-even needs M ≈ the pass-cost ratio, i.e.
near-100% acceptance at K=4 — and the union tax grows with K as fast as M's
ceiling, so NO K escapes. Memory compounds it: spec peaked 11.63 GB MLX memory
at a 9 GB budget (draft + verify activations ≈ +1.5 GB over greedy), hugging
the wire and compressing 60–80 GB/run. Recommendation stands and is now fully
measured for 16 GB streamed MoE: **greedy is the mode; README updated** (spec
row + 16 GB guidance). The M3-Ultra-class conditional win (huge RAM, high-M
reasoning) is untouched. The probe's greedy median is contaminated by a 4th
#15-style ambient-churn sighting (its LAST run: 3.06 tok/s, 130 GB compressed,
byte-identical cache work) — verdict is robust either way (even vs the
contaminated median, K=4 is −3.9%).

## 6. Bench hygiene: matched-draft defaults, draft pre-download, warm-up (MEDIUM)

From the M1 Pro 32 GB 19-run study + follow-up comment (docs/community-results.md,
2026-07-16):

- **Matched-draft default (highest value).** `nunspark bench <llama-model>` pairs the
  default Qwen3-0.6B draft with a Llama target; bench.py warns "acceptance will likely
  be ~0" but runs anyway, so people publish ~0.10 tok/s in both columns and conclude
  deep-K speculation doesn't work. The commenter swapped in
  `mlx-community/Llama-3.2-1B-Instruct-4bit` (0.7 GB) and got M=14.29 / +1097% on code.
  Fix: a small target-family → draft map (Qwen3 → Qwen3-0.6B, Llama-3.x →
  Llama-3.2-1B-Instruct-4bit, no-match → `--ngram` or skip spec arm with a clear
  message) instead of one hard-coded default; refuse the model-draft spec arm on
  tokenizer mismatch unless `--force-draft`.
- **Pre-download the draft before timing.** The first-ever spec run downloads the draft
  mid-run and posts the worst numbers of any run in the study. Resolve/download the
  draft during setup, before any timers start.
- **Warm-up.** First ~2 runs after a cold start read ~15% low; a 5-minute idle gap cost
  the spec arm 20% (draft + pages evicted). Options: one untimed warm-up pass before
  the timed suite (or a `--warmup N` flag), and a note in the report when the run was
  cold. The study also showed apps-open vs closed is pure noise — no need to tell
  users to quit apps.
- Determinism cross-check worth keeping: M is byte-identical across runs at temp=0, so
  M diverging across machines/runs on the same model+draft+K indicates a real bug, not
  noise.

## 8. Fix the 4 known test failures (LOW, test-infra)

Root cause found 2026-07-17 while setting up CI: the three tests/test_cli.py failures
(pack_then_generate, kv_budget_arg_accepted, io_warmer_flags_bit_identical) are NOT
missing optional deps — transformers' "install sentencepiece or tiktoken" error is
misleading (both installed, still fails). The tiny_model_dir fixture ships only a slow
tokenizer that current transformers can no longer convert to a fast one; fix = have the
fixture write a `tokenizer.json` (tokenizers-library serialization) like the other
fixtures do. The fourth (webapp test_run_generation_bad_output_dir_does_not_raise) is a
real unfixed behavior. All four are deselected in .github/workflows/tests.yml — remove
the deselects when fixed.

## 9. Decode demand-parallel warm — MEASURED 2026-07-17, no effect (do not revisit as-proposed)

Community suggestion: a `get_many` batched demand load (thread-pool the post-router
expert misses) at single-token decode, claiming 8 sequential misses/layer → ~256 ms
serial I/O per token. Tried: `StreamingEngine(decode_bulk_warm=True)` (opt-in flag,
default off) extends the existing `warm_bulk` to single-token passes; interleaved A/B
with page-cache flushes (scripts/decode_bulkwarm_probe.py,
scripts/results/decode_bulkwarm_ab.json). Result: control 0.767/1.079 vs warm
0.891/0.854 tok/s — a wash inside run noise, token streams identical. Root cause the
premise is wrong at our coverage: 30B@8GB decode misses ~17.4/token over 48 layers =
**~0.36 misses per layer** — a missing layer almost always misses exactly ONE piece, so
within-layer parallelism has nothing to parallelize. Consistent with Phase-1's "warming
is net-negative for decode". The only way to get queue depth at decode is CROSS-LAYER:
run layers L+1..L+3's (core-resident, tiny) router gates on layer L's hidden state,
prefetch the predicted top-set into the M3a staging buffer — same-token residual-stream
lookahead, much stronger prior than the shipped previous-token temporal prefetch
(Jaccard ~0.30). Cheap offline go/no-go: instrument one forward pass, measure how often
true top-8 at L+k lands in the L-state-predicted top-12 (k=1..3).

**Go/no-go MEASURED — strong GO** (scripts/router_lookahead_probe.py,
scripts/results/router_lookahead_probe.json; Qwen3-30B, 63 decode passes, ~2.9k
layer-pairs per k): k=1 top-12 mean recall **98.3%** (90.4% of passes fully
contain the true top-8; top-16 → 99.2%/95.9%); k=2 top-12 still 96.0%; recall
*improves* with depth (early/mid/late thirds 96.0/99.3/99.7%). Independently
corroborated by Colibrì's GLM-5.2 measurement (71.6% one layer ahead on the
256-expert geometry — see #13). Next step is the online half: run layer L+1's
(core-resident, tiny) router on L's output inside the decode loop, feed the
predicted top-N into the M3a staging tier, and A/B decode tok/s at fixed budget
— the win mechanism is converting decode's serial demand-faults (~0.45 GB/s)
into known-in-advance bulk reads (0.73–1.36 GB/s measured in warm_bulk).

**Plan 7 status (CLOSED 2026-07-20).** M1 (engine seam:
`StreamingEngine(lookahead_prefetch=...)`, counters, bit-identity +
non-blocking + prediction-correctness tests) and M2 (CLI/bench/serve
`--lookahead` plumbing, metrics lines, README flag row) shipped. M3 A/B gate
**FAILED as configured** (scripts/results/lookahead_ab.json; Qwen3-30B, 6 GB,
3 flushed interleaved pairs per config): topn=12 median +3.9% with 2/3 pairs
slower (2.6× read amplification, 42% speculative-load use); topn=8 median
2.561 → 2.777 (+8.4% < the 10% gate), 1/3 pairs −8%, 76% use, 1.28×
amplification. Token streams byte-identical in all 6 pairs. Expert stall
reliably fell 25–30% in every lookahead run — the prediction works; the
residual waste plus 6 GB control variance (2.40–3.02 tok/s) eats the win.
`--lookahead` stays opt-in/off. Refinement candidates recorded in
docs/plan7-lookahead.md ("M3 outcome"); user live sweeps with `--lookahead`
outrank these probe numbers as ground truth.

## 10. Decode is NOT disk-bound: fix _scatter_experts full-buffer rebuild + budget cliff (FIX SHIPPED 2026-07-17 — budget sweep still open)

**Outcome.** Persistent full-size scatter buffers (no re-zero; invalidated on
_make_slot; strategy B of scripts/results/scatter_strategy_microbench.json) shipped in
_scatter_experts. Post-fix, same flushed probes: **6 GB 1.342 → 4.235 tok/s (3.16×**,
scatter bucket 524 → 73 ms/token); 8 GB 0.745 → 1.267 (1.70×, still compressor-
limited). 150-token decode stream byte-identical to the pre-fix control; bitwise
stale-row regression test in tests/test_scatter_persistent_bufs.py; full suite 355
passed / same 4 known failures. **Project decode best on 16 GB: 2.5-2.8 tok/s live (user sweep), 4.2 in flushed
controlled probes, vs the previous 1.6-1.65 headline.**

**Budget formula RESOLVED same day (two iterations).** The user's live greedy sweep
at temp 0 (4/5/6/8/10 GB = 1.79 / 2.52 / **3.23** / 2.84 / 1.33 tok/s, 200 tokens)
put the 16 GB optimum at 6 GB — an earlier 6-vs-8 comparison was confounded by
--temp 1.0 on the 6 GB arm. Final formula in sysmem.py: `auto = 0.75 * (RAM - 8 GB)`,
floor 2 GB — the FIXED 8 GB models the OS-plus-apps baseline (absolute, not
proportional), and one formula now fits all calibration machines: 16 -> 6 (measured
optimum), 64 -> 42 (community ran 44-58 fine), 128 -> 90 (community ran 90 fine).
NOTE — availability clamp TRIED AND REVERTED: a first iteration clamped auto to
`available - 1 GB` using macOS `memory_pressure -Q` free-percentage; in the user's
real session it starved the budget to the 2 GB floor on a machine that ran fine at
6 GB moments later — the kernel free-pct is too volatile mid-session to size a cache
that macOS will happily make room for. Auto is a pure function of TOTAL RAM
(deterministic, testable); loaded machines pass an explicit smaller --budget. Tests
in tests/test_sysmem.py; README updated (headline 2.8-3.2 tok/s, scatter-fix bullet,
16 GB model guide + --budget flag row with the sweep, quickstarts now use auto).
REMAINING: re-run the standard bench suite + report.md / community-results
comparisons on 0.9.x. ~~Re-check the spec-decode arms~~ **DONE 2026-07-24 on the
wired baseline: spec loses at every K (best K=4 −22% vs greedy 7.96) — closed
under #5's small-K entry (spec_breakeven_wired.json).**
Original finding follows.

Original entry (pre-fix measurements):

Attribution probe 2026-07-17 (scripts/decode_time_attribution_probe.py,
scripts/results/decode_time_attribution.json), 30B, 100 greedy tokens, flushed:

- **Expert demand-load wait is only 17–30% of decode.** The dominant bucket is
  `_scatter_experts` EXCLUDING the disk wait: **789 ms/token at 8 GB (59%), 524 ms at
  6 GB (70%)**. Structural cause: every MoE layer of every token builds FULL-SIZE
  zero-filled buffers for all 128 experts (~312 MB/layer, ~15 GB of writes/token over
  48 layers) to host ~8 fired experts, then `mx.eval`s it. The zero rows are never
  gathered by the switch matmul — they are pure waste. Fix candidates: (a) compact
  fired-only buffers with inds remapped to 0..k-1 (16x fewer bytes; needs a numerics
  gate — cross-shape fp16 caution), (b) persistent slot buffers, scatter only the
  fired rows, never re-zero (stale unfired rows are harmless because never gathered;
  depends on MLX in-place/donation behavior for setitem-scatter). Upside ~2-3x decode.
- **The 16 GB auto-budget (8 GB) is past the memory cliff:** 6 GB ran **1.80x faster**
  (1.342 vs 0.745 tok/s) despite 2x the misses (41.6 vs 21.3/token); per-miss stall
  18.7 ms vs 3.0 ms — at 8 GB the loads fight the macOS compressor. Sweep 5-7 GB and
  revisit the auto formula for 16 GB machines (interacts with (a): smaller transient
  buffers may move the cliff).
- Host syncs (router tolist + per-layer eval) are ~10% — a native rewrite is NOT
  supported by the data; the bottleneck is our buffer strategy, fixable in Python/MLX.
- Also fixes/covers: the 40% run-to-run bench variance at 8 GB (pressure-dependent),
  and the same scatter tax inside every spec verify pass and prefill window.

## 7. TensorFold adoptions — PROMOTED to Plan 5 (docs/plan5-tensorfold-adoptions.md)

TensorFold (github.com/ashhart/TensorFold, MIT) — an independent MoE-streaming runtime
on MLX — yielded three adoptions, planned and gated in Plan 5: (A1) garbage-drafter
invariant test (a never-accepted drafter must reproduce greedy byte-for-byte; covers
the untested verify_forward/commit_verified path), (A2) adaptive proposal shrink for
the n-gram drafter (directly attacks #5's worst case: their data shows 15% acceptance
at no slowdown), (A3) batched decode sharing per-layer expert unions across requests
(their headline: 3.4 → 49 total tok/s at batch 16, near-flat memory). Their negative
result — previous-token expert prefetch is a net slowdown (~43% consecutive-token
overlap) — independently confirms our Phase-1 verdict; do not revisit. Their KV
checkpoint w/ suffix-only prefill validates #2; when building #2, copy their
`usage.prompt_tokens_details.cached_tokens` reporting.

## 11. GLM-5.2 / deepseek_v32 follow-ups (Plan 6 M0–M4 shipped 2026-07-20)

Streaming support for `glm_moe_dsa` (GLM-5.2) and `deepseek_v32` landed —
see docs/plan6-glm52.md for gates. Deliberately deferred:

- **Batched decode + tree spec for Paired-cache archs.** Both refuse with a
  clear ValueError today (`_reject_rotating`-style gate; `tree_forward`
  raises). Enabling them means teaching the batched row bookkeeping and
  ephemeral batched-cache seeding about CacheList children. Do only if a
  real GLM-5.2/DSV3.2 batched use case shows up.
- **GLM-5.2 MTP layer as an EAGLE-style drafter.** The checkpoint ships one
  nextn layer (dropped at pack time by mlx-lm sanitize). Same shape as the
  gemma4_assistant story; the eagle_drafter/tree_spec seams exist but tree
  spec is gated off for Paired archs (above), so this needs that first.
  Pre-paid measurement from Colibrì (github.com/JustVugg/colibri, which runs
  GLM-5.2 MTP spec in production): 2.2–2.8 tokens/forward when it pays, and
  **the MTP head must stay int8 — int4 heads collapse to 0–4% acceptance**.
  When we pack the nextn layer, keep it at 8-bit regardless of the base quant.
- **Real-model numbers.** No perf claims exist anywhere for these archs and
  none may be added until measured (community bench or a high-RAM Mac run).
  Active set is ~39B params/token: expect usable speeds only where most of
  the 8/256-expert working set stays RAM-resident; 16 GB Macs will crawl.
- **IndexShare fidelity note.** mlx-lm's implementation runs the indexer
  per-layer and ignores GLM-5.2's `index_topk_freq`/`index_skip_topk_offset`
  IndexShare scheduling fields. NunSpark matches mlx-lm bit-for-bit (that is
  our losslessness contract), so any fidelity question past 2048-token
  contexts is upstream's to resolve; re-check when mlx-lm updates.

## 13. Colibrì adoptions (assessed 2026-07-20 — github.com/JustVugg/colibri, Apache 2.0)

Colibrì is a pure-C GLM-5.2 streaming engine (744B on ~25 GB RAM, token-exact
vs a transformers oracle) — the closest independent comparable to NunSpark.
TensorFold-style adoption review (cf. #7):

- **(C1) Router-lookahead prefetch — validates our #9, PROMOTE.** Their PILOT
  thread prefetches the next layer's experts from routing that is "measurably
  71.6% predictable one layer ahead" on GLM-5.2's 256-expert geometry. Our own
  offline probe (see #9 update) measures 98.3% top-12 recall at k=1 on
  Qwen3-30B. Two independent codebases, two models, same conclusion: the
  residual stream changes slowly enough to prefetch one layer ahead. #9's
  online A/B ran as Plan 7 — shipped lossless + opt-in, but the perf gate
  FAILED (see #9 Plan 7 status): prediction is accurate and stall drops, yet
  read amplification + bench variance eat the win on the 16 GB testbed.
- **(C2) Persistent learned expert pinning — offline go/no-go MEASURED
  2026-07-20: cross-workload transfer is dead; same-workload only.** They
  record per-workload expert usage (`.coli_usage`) and pin the hottest experts
  across sessions; lossless (cache policy only), and we already have the
  unused `pinned` seam in PieceCache. Offline probe over the existing M1
  Qwen3-30B decode traces (scripts/expert_hotset_offline_probe.py,
  scripts/results/expert_hotset_offline.json; 192k expert-loads per workload,
  5646 (layer,expert) pairs): within one workload the distribution IS
  concentrated (top-1% of pairs = ~10% of loads, top-10% = ~46-52%, i.e.
  5-10x uniform) — but a hot set ranked on one workload barely beats uniform
  on another (top-1% pins → 0.0-1.3% coverage vs 1% uniform; top-10% →
  13-19% vs 10%). The hottest experts are workload-specific, which matches
  their own atlas finding (experts are topic-specialized). So: do NOT build
  generic cross-session pinning. The surviving narrow case is same-workload
  cold-start warming (resume yesterday's coding session → prefetch its top
  pins during model load, before the first token) — park it unless a
  cold-start TTFB complaint shows up; in-session, LRU already harvests the
  same concentration. Caveat: one prompt per workload in these traces
  (their atlas doc's autocorrelation trap), so "same workload tomorrow"
  likely transfers better than the cross-workload floor measured here.
- **(C3) MTP int8 constraint** — recorded under #11 (drafter head must be
  int8; int4 collapses acceptance to 0–4%).
- **REJECTED: cache-aware routing (`CACHE_ROUTE`).** They optionally swap
  cached experts into the executed set within a rank window (top-J guaranteed,
  window M preferred-if-cached). This changes which experts run — lossy output
  by construction, violating our bit-identity contract. Even Colibrì ships it
  off-by-default and labels it experimental. Do not re-propose; recorded here
  so the idea has a grave.
- Already have: batch-union expert reads (Plan 5 batched decode), MLA
  compressed KV (deepseek_v32/glm_moe_dsa land it for free).
- Their expert-atlas methodology doc (c/tools/expert_atlas/README.md) lists
  measurement traps worth honoring in the C2 probe: top-p sampling hides ~38%
  of distinct active experts (probe at temp 0), speculative drafts inflate
  usage counts (probe with spec off), cumulative usage logs capture prior runs
  (clear between arms), and per-prompt autocorrelation (validate across
  prompts, not within one).

## 15. Startup bulk prewarm — MEASURED 2026-07-24, GATE FAIL (redundant with prefill warm_bulk; do not revisit as-proposed)

**Idea (from #14 follow-ups / #13 C2's parked "cold-start warming").** The
first cold run at the wired 10 GB budget looked ramp-dominated (live: 2.68
tok/s over 200 tokens), so sequential-read the cache full at engine load
(dense/core first, then experts round-robin across layers, capped at the
budget) before the first token — a one-shot known-in-advance bulk read.

**Probe** (scripts/startup_prewarm_probe.py,
scripts/results/startup_prewarm_ab.json; M4 16 GB, Qwen3-30B, 10 GB budget,
200 greedy tokens, 3 ABBA pairs, both arms wired/shipped defaults, page cache
flushed before every run, per-token window speeds, token streams
byte-identical):

- **Gate metric (end-to-end wall incl. prewarm): control 30.82 s vs prewarm
  31.99 s median (−3.8%)**; pairs +47.8/−12.2/−3.3% — fails both clauses of
  the ≥10%-median / no-pair-worse-than-5% gate.
- **Mechanism: the prewarm is redundant.** Decode-segment misses were
  byte-identical in all 6 runs across BOTH arms (1167 expert misses /
  2.885 GB / 14.8 MB-per-token — deterministic greedy). The 42-token
  prefill's existing warm_bulk (#1) already bulk-fills the cache and page
  cache at 2–3 GB/s; prewarm-arm prefill was 3.9 s vs control 4.0 s — there
  was nothing left for the 3.2 s / 10 GB / ~3 GB/s prewarm to win, so it
  costs its own runtime back.
- What it did buy (why a variant could be argued later, on tail latency
  only): a softer ramp (first-50-token window median 7.11 vs 5.69 tok/s,
  expert stall 3.56 vs 4.48 s) and a tight worst case (arm spread
  31.8–33.5 s vs control 28.5–64.1 s).
- **The real cold-start story changed.** The one catastrophic run
  (control-run1: 64 s, prefill 14.9 s, first-50 2.59 tok/s) is the only run
  matching the live first-run complaint, and its signature is COMPRESSOR
  churn — 35.7 GB compressed / 2.3M compressions, ~2× every other run — in
  the first process after a long idle (the #10 hygiene note's known
  first-runs-read-low effect). It is NOT page-cache coldness: controls 2/3
  paged in the same ~13 GB from disk (845–897k pageins) and ran fast
  (28.5/30.8 s). Two consequences: (a) the house flush manufactures cold
  DISK state but not the true first-run-of-the-day state, so that state is
  unprobed (n=1 per arm); (b) #14's "first live run was cache fill at
  demand-fault speed" reading was incomplete — the fill is normally absorbed
  by prefill warm_bulk at bulk bandwidth; it is the compressor fight that
  makes the first run slow.

**Verdict: do not ship a startup prewarm as-designed.** If the
first-run-of-the-day slowness recurs live, the follow-up is characterizing
that state (compressor churn in the first process after idle — e.g. does a
second immediate run always fix it? does a small warm-up generation at serve
startup?), not more bulk reading. Any insurance-style prewarm revival must be
argued on tail latency with a probe that can actually manufacture the true
cold state.

**Third sighting (2026-07-24, lookahead wired re-run):** the first run of
that session — a control — hit 3.81 tok/s with 19.66 s stall and 39.4 GB
compressed vs 7.98–8.15 for every later control (identical config). Same
signature every time: FIRST child process of a session, ~2× the compressor
traffic, no excess disk reads. The pattern is now consistent enough to
probe deliberately (idle-gap-controlled A/B: does a ~30-token throwaway
generation at process start, or an immediate second run, always restore
full speed?).

**Fourth sighting (2026-07-24, spec break-even probe) — and it is NOT only
a first-run effect:** the LAST run of that session (a greedy control) hit
3.06 tok/s with 130 GB compressed (4.3× its clean twin) on byte-identical
cache work (1382 misses / 3.416 GB both runs), after six spec runs that
each compressed 60–80 GB. So the trigger looks like ambient compressor
state (accumulated or inherited), not process order per se. Contaminated
that probe's greedy median (verdict robust anyway). Any future probe on
this machine should treat per-run `compressed_gb_during_run` as a validity
check (clean runs: 15–20 GB) and re-run outliers.

## 14. Wired-memory limit: the budget cliff is a wiring artifact (MEASURED 2026-07-24 — GATE PASS, ship it)

**Finding.** NunSpark never called `mx.set_wired_limit()`, so the entire piece
cache lived in pageable anonymous memory — exactly what the macOS compressor
eats. (mlx-lm's own generate loop sets the wired limit to
`device_info()["max_recommended_working_set_size"]` before decoding; our
generate loop didn't.) A/B probe (scripts/wired_limit_probe.py,
scripts/results/wired_limit_ab.json; M4 16 GB, Qwen3-30B, 150 greedy tokens,
fresh child process per run, ABBA-interleaved, flushed, vm_stat deltas per
run, wired limit 11.84 GB = device max_recommended):

- Median tok/s control vs wired: 6 GB 3.26 → 4.59 (+41%); 8 GB 5.30 → 6.40
  (+21%); 10 GB 2.67 → **7.98 (+199%)**. Wired never lost any pairing; token
  streams byte-identical in all 12 runs (wiring is memory policy only).
- **The #10 budget cliff inverts**: wired scaling is monotonic with budget
  (4.59 / 6.40 / 7.98 at 6/8/10 GB). "More budget is not faster" was true
  only because the extra budget was being compressed out from under the
  engine — control RSS at a 10 GB budget was 4.9–5.5 GB (the compressor held
  ~half the cache) vs 10.6 GB wired. Control runs compressed 37–57 GB of
  memory per ~30–60 s decode (10 GB control: ~0.5M swapins + ~0.58M
  swapouts); wired runs 8–13 GB.
- New 16 GB best: **8.0 tok/s greedy** (2.5× the 3.2 headline), and wired
  runs are near-deterministic (spread 0.3–0.8% vs control's 2.20–4.32 at
  6 GB) — which also explains the 40% bench variance recorded in #10 and
  the variance that ate the Plan-7 lookahead gate (#9); lookahead may
  deserve a re-run on a wired baseline.

**Follow-ups, in order:**
1. ~~Wired-mode budget re-sweep at 9/10/11 GB~~ **DONE 2026-07-24**
   (scripts/results/wired_limit_ab_v2.json): the 16 GB wired optimum is a
   **10 GB budget** — wired medians 6.69 / 7.79 / 6.36 at 9/10/11 GB; all
   four wired 10 GB runs across both sweeps sit in 7.785–8.006 tok/s.
   11 GB regresses and destabilizes for a measured reason: peak working set
   11.99 GB exceeds the 11.84 GB wired limit, so the tail past the wire is
   compressed again (33.8 GB compressed, 299k swapouts in the slow run).
   **Rule: budget + ~1 GB engine overhead must stay under
   max_recommended_working_set_size** — the wired-regime auto formula should
   be ≈ `max_recommended − 2 GB` (16 GB: 11.84 − 2 ≈ 10, the measured
   optimum), not a fraction of total RAM. Also observed: unwired control
   spans 3× run-to-run (2.05–6.38 at 9 GB — the compressor sometimes stays
   away entirely), so wiring buys determinism as well as speed.
2. ~~Live-session verification~~ **DONE 2026-07-24: 6.92 tok/s live over
   800 tokens at 10 GB** (second run, warm page cache — near-steady-state
   from token 1), vs 3.23 pre-wiring live best and 1.33 pre-wiring live at
   the same 10 GB budget. Matches the 7.8–8.0 flushed probes given live
   conditions. The first-run details below stand as the cold-start
   characterization.
   **First live run (2026-07-24, 200 tokens, 17-token prompt): 2.68 tok/s —
   2.0× the unwired live baseline at the same budget/length (1.33), but
   ramp-dominated:** 5146 misses ≈ 9 GB ≈ the entire 9.52 GB resident peak,
   i.e. the whole run was cache fill at single-token demand-fault speed,
   with the user observing the expected slow-start/fast-finish. Below the
   old 6 GB live number (3.23) at this length — **the optimum budget is now
   generation-length-dependent** (fill cost vs steady-state advantage).
   Longer-run live datapoint pending. This is also C2's parked "cold-start
   warming" complaint materializing: a startup bulk prewarm (sequential-read
   the cache full at load time, ~2 GB/s, instead of demand-faulting it over
   the first ~150 decode tokens at ~0.45 GB/s) would kill the ramp — at a
   10/16 coverage ratio even a uniform fill gives a ~62% hit floor from
   token 1, and unlike the dead per-token decode warming (#9/Phase-1) it is
   a one-shot known-in-advance bulk read, the same mechanism that made
   prefill warm_bulk (#1) a 2× win. **(Measured 2026-07-24: GATE FAIL — see
   #15. Prefill warm_bulk already does this fill; the live first-run
   slowness is compressor churn, not page-cache coldness.)**
3. ~~Ship set_wired_limit~~ **SHIPPED 2026-07-24** (uncommitted):
   `sysmem.wire_memory_limit()` sets the limit to the device
   max_recommended_working_set_size; `StreamingEngine(wire_limit=True)`
   default-on before any allocation (all construction sites inherit);
   `--no-wire` opt-out on generate/serve/bench; tests/test_wire_limit.py.
   The probe's child passes `wire_limit=False` explicitly so its control arm
   stays honest under the new default.
4. ~~Re-derive the auto-budget formula~~ **DONE 2026-07-24** (uncommitted):
   sysmem.py auto is now `0.75 × RAM − 2 GiB` (16 → 10, 64 → 46, 128 → 94)
   — still a pure function of total RAM (availability-clamp lesson honored);
   the 2 GiB headroom keeps budget + ~1 GB engine overhead under the wired
   limit (~0.75 × RAM). tests/test_sysmem.py + test_cli.py constants
   updated (old: 6/42/90); README headline, model guide, `--budget` row,
   quickstart, and CLAUDE.md rewritten — "more budget is not faster" is
   retired with attribution to the wiring artifact. 64/128 GB values sit
   inside community-verified unwired ranges (44–58 / 90 GB ran fine);
   wired re-verification from community machines welcome.
5. Re-run candidates on the new wired baseline: ~~the Plan-7 lookahead M3
   gate~~ **DONE 2026-07-24 — FAIL again, FINAL** (top8 +1.5%, top12 −5.9%
   median at 10 GB wired; with variance gone there was no gate-sized win to
   mask — expert stall is only ~14% of decode at this baseline, so any
   prefetch win is bounded to low single digits. Details in
   docs/plan7-lookahead.md "M3 wired re-run" +
   scripts/results/lookahead_wired_ab.json; do not re-gate on this hardware
   class — check the stall share first on any new machine/model). ~~The
   spec-vs-greedy break-evens (#5, #10)~~ **DONE 2026-07-24 — spec loses at
   every K on the wired baseline (best K=4 −22%); closed under #5's small-K
   entry (spec_breakeven_wired.json). All #14 follow-ups are now closed.**

## 12. DeepSeek-V4-Flash (`deepseek_v4`) — BLOCKED on upstream mlx-lm (assessed 2026-07-20)

**Verdict: cannot ship yet without violating the losslessness contract.**
DeepSeek-V4-Flash (284B total / 13B active, `model_type: "deepseek_v4"`) has
NO implementation in any released mlx-lm (0.31.3 is latest, checked PyPI
2026-07-20). Support lives only in open PR ml-explore/mlx-lm#1189
(32+ commits, still churning), whose own thread documents cache divergence
between single-token and batched decode, Metal-kernel numerical instability,
and a RoPE frequency bug. Our product invariant is "bit-identical to
full-load mlx-lm"; with no stable reference, bit-identity is unverifiable —
and vendoring the PR (gemma4_assistant-style) would enshrine its bugs as our
reference. Community MLX conversions exist on HF but fail to LOAD upstream
(`KeyError: 'deepseek_v4'`) — they were converted, not validated.

**Unblock trigger:** PR #1189 merges and ships in an mlx-lm release → bump
the pin, then run the Plan-6 playbook (tiny seeded model, gated milestones).

**Contract-fit analysis (done now so the plan can start cold).** Config
(deepseek-ai/DeepSeek-V4-Flash): 43 layers, hidden 4096, 256 routed + 1
shared expert top-6, first 3 MoE layers hash-routed (`num_hash_layers`,
static tid2eid table), hybrid CSA/HCA attention with per-layer
`compress_ratios` alternating [0, 4, 128], indexer (`index_topk` 512,
sliding_window 128), yarn rope + separate `compress_rope_theta`, mHC
hyper-connections (`hc_mult` 4, Sinkhorn-normalized comb), 1 MTP layer,
fp8 block weights with fp4 experts. Streaming-contract breaks, hardest first:

1. **mHC hyper-connections replace residuals.** The engine's outer loop
   assumes ONE [B,L,D] stream with residual-inside-block; mHC threads
   hc_mult=4 hidden copies across layers with learned pre/post/comb mixing.
   Needs: expand-at-embed, collapse-before-final-norm, and a LayerRunner +
   LayerContext carrying the 4-copy state (gemma4 seam precedent — bigger,
   but the seam exists). This is the single largest engine assumption ever
   touched; budget it accordingly.
2. **Heterogeneous compressed caches.** Per-layer compress_ratios ⇒ mixed
   cache classes (whatever the PR ships: sliding + compressed-KV + indexer),
   NOT plain KVCache. Extends the Paired/cache_plan seam from Plan 6; spill,
   clone/recording, and offset probes all need the new kinds or clean
   refusals.
3. **Hash-routed early MoE.** Deterministic tid2eid routing for layers 0-2:
   selective streaming actually gets EASIER (fired set known from token ids,
   perfect prefetch), but the router seam must bypass gate logits entirely —
   a new `moe_route`-adjacent hook that consumes token ids, not activations.
4. **Packer:** fp8-block + fp4-expert checkpoints (mlx conversion handles
   dequant upstream, as deepseek_v32 sanitize does); MTP layer drop; expert
   split follows the existing switch-mlp path if the PR uses SwitchGLU.

**Do NOT** pre-implement against the PR branch "to be ready" — the PR's own
instability means any pre-built parity target is sand. Re-check this entry
when bumping mlx-lm for any other reason.
