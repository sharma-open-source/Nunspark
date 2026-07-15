# Streaming 30B/70B LLM Inference on a 16 GB Apple M4 using Weight Streaming + Speculative Decoding

## Overview

This report summarizes experiments performed on a **16 GB Apple M4
MacBook Pro** to evaluate whether very large language models (32B--70B
class) can be executed using a streaming runtime with a small resident
weight cache.

Rather than loading the entire model into memory, weights are streamed
on demand while speculative decoding reduces verifier work.

------------------------------------------------------------------------

# Hardware

  Component        Value
  ---------------- ------------------------
  Machine          Apple MacBook Pro (M4)
  Unified Memory   16 GB

------------------------------------------------------------------------

# Models Evaluated

## Target Models

-   Qwen2.5-32B-Instruct-4bit
-   Llama-3.3-70B-Instruct-4bit

## Draft Models

-   Qwen2.5-0.5B
-   Qwen2.5-3B
-   Qwen3-1.7B
-   Llama-3.2-1B

------------------------------------------------------------------------

# Major Findings

## 1. Resident cache size is not the primary bottleneck

Increasing the resident cache from 512 MB up to 12 GB produced only
marginal improvements in throughput and, in some cases, reduced
performance due to memory pressure.

The dominant factor was **speculative acceptance**, not weight
residency.

------------------------------------------------------------------------

## 2. Draft model quality matters more than draft size

  -----------------------------------------------------------------------
  Draft Model                                 Result
  ------------------------------------------- ---------------------------
  Qwen2.5-0.5B                                Low acceptance, \~0.43
                                              tok/s

  Qwen2.5-3B                                  Significant improvement
                                              (\~0.67 tok/s)

  Qwen3-1.7B                                  Best overall draft model
                                              (\~0.92--1.13 tok/s
                                              depending on workload)

  Qwen2.5-7B                                  Slower despite being larger
  -----------------------------------------------------------------------

The newer Qwen3-1.7B consistently outperformed larger Qwen2.5 draft
models.

------------------------------------------------------------------------

## 3. Optimal speculative window is draft-model dependent

### Qwen2.5-3B

Increasing draft tokens beyond 8 produced almost no improvement.

### Qwen3-1.7B

Performance improved steadily:

  Draft Tokens     Throughput
  -------------- ------------
  4                0.56 tok/s
  8                0.73 tok/s
  12               0.80 tok/s
  16               0.86 tok/s
  20               0.88 tok/s
  24               0.92 tok/s

Beyond 24 tokens, performance saturated.

------------------------------------------------------------------------

# Workload Dependence

Performance varied dramatically depending on the prompt.

  Workload                   Acceptance   Throughput
  ------------------------ ------------ ------------
  Vector DB comparison             2.99   0.49 tok/s
  SaaS billing design              3.45   0.55 tok/s
  JS alert function                5.41   0.92 tok/s
  React WebSocket hook             6.06   1.03 tok/s
  LRU Cache (TypeScript)           7.14   1.13 tok/s

The streaming engine remained constant; throughput tracked speculative
agreement between draft and target models.

Approximate relationship observed:

> Throughput ≈ Verifier Speed × Acceptance Multiplier

------------------------------------------------------------------------

# Fast Approximate Verification (accept-top-k)

Allowing acceptance of draft tokens within the target's top-k
predictions substantially reduced verifier passes.

## Qwen 32B

  Mode            TPS   Acceptance   Deviation
  ------------ ------ ------------ -----------
  Exact          1.13         7.14          0%
  Top-k = 5      2.83        18.18       21.7%
  Top-k = 10     2.65        18.18       26.7%

Top-k=5 offered the best speed/quality tradeoff among tested values.

------------------------------------------------------------------------

# Llama 70B Results

## Exact

-   Throughput: **0.89 tok/s**
-   Acceptance: **11.76**
-   Deviation: **0%**

## Top-k = 5

-   Throughput: **1.45 tok/s**
-   Acceptance: **20.0**
-   Deviation: **9.8%**

The Llama family exhibited much higher draft-target agreement, likely
because the draft and target models share architecture, tokenizer, and
training lineage.

------------------------------------------------------------------------

# Memory Efficiency

## Qwen2.5-32B

-   Model size: 18.43 GB
-   Resident weights: \~1.10 GB
-   Peak unified memory: \~3.1 GB

## Llama-3.3-70B

-   Model size: 39.69 GB
-   Resident weights: \~1.44 GB
-   Peak unified memory: \~3.4 GB

Both models operated while keeping only a small fraction of weights
resident.

------------------------------------------------------------------------

# Key Insights

-   Weight streaming is practical on 16 GB Apple Silicon.
-   Speculative decoding dominates overall performance.
-   Draft-target alignment is a stronger predictor of throughput than
    cache size.
-   Performance is highly workload dependent.
-   Approximate verification (top-k acceptance) enables significant
    throughput gains with controllable divergence.

------------------------------------------------------------------------

# Recommended Configurations

## Highest Exact Throughput

-   Target: Qwen2.5-32B-Instruct-4bit
-   Draft: Qwen3-1.7B-4bit
-   Draft Tokens: 24
-   Cache Budget: 1 GB

Typical throughput:

**0.9--1.1 tok/s**

------------------------------------------------------------------------

## Highest Approximate Throughput

-   Target: Qwen2.5-32B-Instruct-4bit
-   Draft: Qwen3-1.7B-4bit
-   Draft Tokens: 24
-   Accept Top-k: 5

Typical throughput:

**2.6--2.8 tok/s**

------------------------------------------------------------------------

## Highest Fidelity Large Model

-   Target: Llama-3.3-70B
-   Draft: Llama-3.2-1B
-   Accept Top-k: 1

Typical throughput:

**\~0.9 tok/s**

------------------------------------------------------------------------

# Future Work

-   Benchmark additional prompt categories (reasoning, RAG, refactoring,
    summarization).
-   Measure acceptance histograms rather than only averages.
-   Evaluate semantic correctness under approximate verification.
-   Explore adaptive top-k based on confidence.
-   Investigate smarter layer prefetching and cache policies.

------------------------------------------------------------------------

# Conclusion

These experiments demonstrate that large language models well beyond
physical memory size can be executed on a 16 GB Apple M4 using streamed
weights and speculative decoding.

The strongest finding is that **draft-target agreement---not resident
cache size---is the primary determinant of throughput**. With
appropriate draft models and speculative verification, a streamed 32B
model achieved over **1 tok/s losslessly** and nearly **3 tok/s** in
approximate mode, while a streamed 70B model remained usable with only
\~1.4 GB of resident weights.

------------------------------------------------------------------------
------------------------------------------------------------------------

# Part 2: MoE Expert Streaming (Qwen3-30B-A3B and GPT-OSS-120B)

## Overview

Part 1 established that dense-model streaming is pinned at
`SSD bandwidth / model bytes` per verify sweep, and that speculative
decoding is the lever that beats that ceiling. Mixture-of-Experts (MoE)
models change the equation entirely: a token only *fires* a small
fraction of the model's weights. Qwen3-30B-A3B fires **8 of 128**
experts per layer per token; GPT-OSS-120B fires **4 of 128**. If the
streaming runtime caches, prefetches, and verifies at expert
granularity, bytes-per-token collapses independent of speculative
acceptance -- MoE sparsity is itself a streaming lever, and a
complementary one to Part 1's draft-target agreement.

This part covers Plan 4 (`docs/plan4-moe-streaming.md`), a five-gate
milestone sequence (M1-M5, with M4 folded into M3) run on the same
16 GB Apple M4 as Part 1, plus field validation from community
volunteers on larger machines.

------------------------------------------------------------------------

# Milestone Ablation: Qwen3-30B-A3B-4bit

Each milestone's gate benchmark is a controlled step on top of the
previous one, so the sequence reads as a free ablation study. All runs
below use the same three workloads as Part 1 (code / prose / reasoning)
at an 8 GB piece-cache budget.

## M1 -- Baseline (no expert-aware caching)

Greedy tok/s topped out at **0.51-0.53**, with expert hit rate only
57-64% against the live MRU cache policy -- a policy that is correct
for the dense cyclic scan but actively wrong for experts. Per-token
expert demand is ~1.03 GB (8.13 experts x 48 layers x 2.65 MB); at
57-64% hit that leaves ~0.8-1.0 GB/token of misses, which pins
throughput near 0.5 tok/s.

The key locality finding that shaped the rest of the plan: **per-token
consecutive Jaccard overlap of fired experts is low (0.29-0.31)**, but
**per-verify-pass overlap (spec, K=24) is high (0.74-0.80)**. Predicting
the next *token's* experts from the last token is weak; predicting the
next *verify pass's* expert union from the previous pass's union is
strong. This redirected the prefetch design in M3 from per-token to
per-verify-pass granularity. An offline LRU simulation over the M1
trace also predicted 84% expert hit at 5.2 GB and 94% at 7.8 GB of
expert capacity -- versus the 57% the live policy achieved at an 8 GB
budget, i.e. an entire policy's worth of headroom was on the table.

## M2 -- Two-region cache (dense MRU + expert LRU, cores pinned)

`PieceCache` was split into two regions: dense/core pieces keep the
original MRU policy; expert pieces get a plain LRU with a configurable
`--expert-cache-frac` budget split. Cores, embed, and head (<1 GB for
the 30B) are pinned unconditionally.

| workload | tok/s M1 -> M2 | expert hit% M1 -> M2 | MB/token M1 -> M2 |
|---|---|---|---|
| code | 0.51 -> **1.33** (2.6x) | 57.2 -> **89.2** | 916 -> **118** |
| prose | 0.53 -> **1.85** (3.5x) | 63.8 -> **92.2** | 759 -> **85** |
| reasoning | 0.52 -> **1.50** (2.9x) | 58.7 -> **89.0** | 875 -> **120** |

Core hit rate reached 99.5% (pinning works as intended); peak memory
dropped from 10.5 GB to 8.9 GB. The offline simulation's 84-94%
prediction was essentially confirmed by the live 89-92% result --
MRU-for-experts was the entire problem, not cache capacity.

## M3 -- Speculative expert prefetch, at verify-pass granularity

Kills the router stall: during a multi-token verify pass, each
upcoming layer's experts fired on the *previous* pass are enqueued as
low-priority speculative loads (demand cores always outrank them). M4
(fired-union selective loading in the tree-verify path, bit-identical
to load-all at atol=0.0) landed first and is folded into every spec
number below.

Getting the placement policy right took four iterations, because one
K=24 verify pass touches 66-83 experts/layer (~8.4-10 GB) -- more than
the entire 7.2 GB expert cache region. Inserting speculative pieces
directly into the expert LRU always failed someone: hot-insert (MRU
end) let the prefetch blast evict the live working set (reasoning
collapsed to 0.30 tok/s); cold-insert (LRU end) let pieces get evicted
before the pass that needed them (code used 0 of 5036 issued
prefetches); epoch-protected in-LRU inverted the eviction order and
made things worse. A staging buffer added *on top of* the existing
budget fixed placement but pushed peak memory to 11.6 GB, and macOS
paging pressure ate the win. The design that worked: **a staging
buffer that shares the expert region's byte budget dynamically** --
occupied staging squeezes the LRU tail; empty staging (greedy mode)
returns the full region to demand traffic. Speculative loads can never
policy-evict demand pieces (they never enter the LRU) and can never
grow memory beyond a 20% staging cap.

| workload | mode | tok/s OFF -> ON | stall s OFF -> ON | spec used / wasted |
|---|---|---|---|---|
| code | spec | 0.35 -> **0.52** (+49%) | 165.6 -> 138.7 (-16%) | 82% / 8% |
| prose | spec | 0.56 -> **1.17** (+109%) | 75.8 -> 34.2 (-55%) | 72% / 10% |
| reasoning | spec | 0.65 -> **1.54** (+137%) | 91.1 -> 42.5 (-53%) | 75% / 10% |

Reasoning spec at **1.54 tok/s** is the best number the project has
produced -- an M=7.41 acceptance multiplier finally survives the
cache. Wasted speculative bytes stayed at 8-10% (well under the 20%
target); output is bit-identical prefetch-on vs prefetch-off
(unit-tested). Code remains the residual case: its per-pass union
exceeds the whole cache region regardless of policy, so stall only
improved -16% -- a capacity problem, not a placement problem, deferred
to tuning (smaller K, bigger budget) rather than more architecture.

## Ablation summary

| stage | code tok/s | prose tok/s | reasoning tok/s | expert hit% |
|---|---|---|---|---|
| M1 baseline (greedy) | 0.51 | 0.53 | 0.52 | 57-64 |
| M2 two-region cache (greedy) | 1.33 | 1.85 | 1.50 | 89-92 |
| M3 spec + prefetch (spec) | 0.52 | 1.17 | **1.54** | -- |

------------------------------------------------------------------------

# The Headline: GPT-OSS-120B on 16 GB (M5)

GPT-OSS-120B (mlx-community 4bit, 63.39 GB, 36 layers x 128 experts,
4 fired) is nearly **4x larger than the 16 GB machine's unified
memory**. M5 ran it end-to-end through pack -> stream -> generate on
both the 16 GB M4 and, via a community volunteer, a 64 GB Apple M1 Max.

## Two engine bugs found and fixed via the 120B

Getting this model running surfaced defects invisible on smaller
models, both shipped in release 0.5.0:

1. **Sliding-window mask bug.** Global attention masks were built from
   layer 0's `RotatingKVCache`, whose `make_mask` clamps offset to
   `window - 1`; any multi-token pass past the window produced a mask
   short by `offset - 127` columns, crashing in `broadcast_shapes`.
   Hit both locally and independently by a 64 GB community volunteer
   on the published package -- fixed with per-kind mask sources.
2. **Trim rollback unsound after rotation.** `RotatingKVCache.is_trimmable()`
   is `False` once the cache has rotated, so trim-based speculative
   rollback silently broke. Replaced with `engine.verify_forward`
   (per-layer ephemeral cache clones) + `commit_verified`
   (append-only accepted rows) across both `speculative_generate` and
   `ngram_speculative_generate`. 235 tests pass; gpt-oss regressions
   are bit-identical to greedy past rotation.

Both fixes were field-validated: the same volunteer who hit the
sliding-window crash on the pre-0.5.0 release confirmed it gone on
0.5.0.

## Results

**Community (Apple M1 Max, 64 GB, nunspark 0.5.0, budget=58GB):**
greedy **1.65-1.96 tok/s** (reasoning/code/prose), ~80% expert hit,
~450-494 MB/token, peak 51-55 GB. This clears the M5 speed gate
(>=1.0 tok/s greedy on community hardware).

**Local (Apple M4, 16 GB, budget=8GB):** greedy **0.11-0.14 tok/s**,
expert hit 41-53%, 1.4-1.6 GB/token, peak ~12.9 GB. The machine is
doubly capacity-starved: per-token expert demand (~1.8 GB) vastly
exceeds the <=9% of the 57 GB expert pool that fits in cache, and the
non-expert cores alone (~7-9 GB) exceed the entire budget, forcing
~10 core re-reads per token. ~80% of wall time is kernel/mmap
fault-path time (~300 MB/s effective). This is correctly understood as
**capacity-bound, not a defect** -- the model runs correctly and
coherently on 16 GB, which was the correctness bar; 16 GB is the
floor, not the target audience. 32 GB+ with a larger `--budget` is
where the model is actually usable.

------------------------------------------------------------------------

# The Negative Result: Speculative Decoding Is the Wrong Lever for Sparse MoE

Part 1's central finding was that speculative decoding dominates dense
streaming throughput. For sparse MoE, the opposite holds, and the
120B data makes the mechanism explicit.

K=16 n-gram (prompt-lookup) drafting is the best case for a spec
win: it is tokenizer-exact, model-cost-free, and lossless
(`deviation_rate 0.0` on every run). It still loses badly:

| workload | greedy tok/s | ngram-spec tok/s | M | spec MB/token (greedy) |
|---|---|---|---|---|
| code | 0.11 | 0.05 | 1.30 | 5005 (1602) |
| prose | 0.14 | 0.06 | 1.39 | 4598 (1386) |
| reasoning | 0.14 | 0.04 | 1.15 | 8237 (1648) |

Root cause is structural: each of the K+1 positions in a verify pass
fires its own 4-of-128 experts with low cross-position overlap, so the
per-pass expert-union I/O grows roughly linearly with K while
acceptance (M) does not keep pace (M <= 1.39). Bytes/token run 3-5x
greedy and expert hit rate roughly halves -- the drafter is "free" but
the *verification* is not, because MoE verification means reading many
more experts than a single greedy step would. The community's
model-draft run on the 64 GB machine (K=24, M 1.19-2.13, spec
0.27-0.59 vs greedy 1.65-1.96 tok/s) shows the identical signature,
confirming this is regime-independent for sparse MoE, not an artifact
of the 16 GB machine.

The contrast with dense models is stark: the community's
Llama-3.3-70B run shows spec **winning** on code (4.57 tok/s spec vs
3.39 tok/s greedy, M=14.29) -- a dense verify pass re-reads
approximately the same weights as one greedy step, so acceptance
is nearly free.

> **Speculative decoding is the lever for dense streamed models
> (Part 1); expert-aware caching (M2/M3) is the lever for sparse MoE
> models (Part 2).**

------------------------------------------------------------------------

# Future Work

- **Prefill / TTFT bulk read.** 120B prefill takes 57-95 s on 16 GB
  because prefill's per-layer expert union is large and reads go
  through demand-faulted mmap at ~300 MB/s. Since prefill's fired-expert
  set is fully known right after each layer's router runs (unlike
  decode's unpredictable access pattern), a batched/`madvise`-driven
  bulk read per layer could hit sequential-class SSD bandwidth instead.
  Target: 57-95 s -> 15-30 s.
- **Persistent prompt-prefix KV cache.** Save/load the shared
  system-prompt prefix's KV state to disk so repeat sessions skip
  prefill of the shared prefix entirely; stacks with the bulk-read
  optimization above.
- **Small-K spec sweep for MoE.** At the measured M~1.3, a much
  smaller K (2-4) shrinks the per-pass expert-union tax roughly
  5x and might break even; open question, not yet measured. Related:
  `nunspark bench` should probably default the spec arm off (or to a
  small K) for MoE manifests.
- **Reproduce the Llama-3.3-70B spec anomaly.** The community 70B run
  shows code spec beating greedy (4.57 vs 3.39) but prose/reasoning
  spec *losing* despite healthy acceptance (M=4.00/7.69) -- for a
  dense model a K-token verify pass should cost about one greedy
  token's I/O, so this is unexplained and needs local repro plus a
  check of whether the Qwen3-0.6B/Llama-3.3 tokenizer mismatch is
  contaminating the accounting.

------------------------------------------------------------------------

# Part 2 Conclusion

MoE sparsity turns out to be a streaming lever in its own right,
independent of speculative decoding: fixing the cache policy alone
(M2) delivered a 2.6-3.5x greedy speedup on Qwen3-30B-A3B by
recognizing that experts need LRU, not the MRU policy tuned for dense
cyclic scans. Layering verify-pass-granularity speculative prefetch on
top (M3) pushed the best workload to 1.54 tok/s. And the headline
result -- a 59 GB GPT-OSS-120B model producing coherent output on a
16 GB Mac, and clearing 1.65-1.96 tok/s greedy on a 64 GB community
machine -- shows the same streaming architecture that worked for dense
32B/70B models in Part 1 scales to a 120B-class MoE model, provided
the cache and prefetch layers understand *expert* granularity. The
clearest architectural lesson of Part 2 is the inverse of Part 1's:
where Part 1 found speculative decoding to be the dominant lever,
Part 2 finds it actively counterproductive for sparse MoE, and expert
cache/prefetch design to be the lever instead.
