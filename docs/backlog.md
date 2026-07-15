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
