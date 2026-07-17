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
comparisons on 0.9.x, and re-check the spec-decode arms (verify passes shared the
same scatter tax, so M-vs-win thresholds shift toward greedy).
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
