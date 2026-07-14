# Plan 4 — M3 Gate Summary: Speculative Expert Prefetch (+ M4 selective tree verify)

Change under test: M3a temporal expert prefetch — during a multi-token verify
pass, enqueue the experts each upcoming layer fired on the *previous* pass as
low-priority speculative loads (two-tier prefetch queue; demand cores always
outrank them). Getting the *placement policy* for those speculative pieces
right took four iterations; all six arms below are the same engine, same
prompts (code/prose/reasoning), 200 tokens, 8 GB budget, greedy + spec
(Qwen3-0.6B draft, K=24). M4 (fired-union selective expert loading in
`tree_forward`, bit-identity proven at atol=0.0 two independent ways) landed
before the spec arms, so every spec number includes it.

## The placement-policy knife-edge (why four iterations)

One K=24 verify pass touches 66–83 experts/layer ≈ 8.4–10 GB of expert weight —
more than the whole 7.2 GB expert region. Any policy that inserts speculative
pieces *into* the expert LRU therefore fails someone:

| spec tok/s | OFF | hot-insert | cold-insert | one-pass protect | staging (add-on) | **staging (shared)** |
|---|---|---|---|---|---|---|
| code | 0.35 | 0.89 | 0.45 | 0.32 | 0.62 | 0.52 |
| prose | 0.56 | 0.81 | 0.76 | 0.52 | 0.84 | **1.17** |
| reasoning | 0.65 | 0.30 | 0.91 | 0.66 | 0.24 | **1.54** |

- **hot-insert** (MRU end): prefetch blast evicts the live working set →
  reasoning collapses (0.30).
- **cold-insert** (LRU end): pieces evicted before the pass that needs them →
  code used 0 of 5036 issued (0.45).
- **one-pass protect** (epoch-protected in-LRU): protection inverts the value
  order — the *demand* set is evicted first; worst of all arms. Also exposed a
  consume-side gating hole: the first greedy token after prefill issued ~3.3k
  prefetches (~8.8 GB) from the prefill's giant union.
- **staging, added on top of budget**: speculative pieces in a separate capped
  buffer, outside the LRU entirely + issuance gated on the *current* pass being
  multi-token. Fixed placement and the blast — but the extra 1.4 GB pushed spec
  peak to 11.6 GB and macOS memory pressure ate the win (reasoning 0.24 with
  *unchanged* hit% and stall — pure paging cost).
- **staging, shared budget (final)**: staging shares the expert region's byte
  budget dynamically (occupied staging squeezes the LRU tail; empty staging —
  greedy, prefetch off — returns the full region). Peak back to 10.3 GB.

Final policy: speculative loads can never *policy*-evict demand (they never
enter the LRU) and can never grow memory (they squeeze at most the 20% staging
cap); unconsumed guesses expire after one pass of grace.

## Gate benchmark (final arm vs no-prefetch control, same day)

| workload | mode | tok/s OFF → ON | stall s OFF → ON | spec used / wasted |
|---|---|---|---|---|
| code | greedy | 1.52 → 1.50 | 42.4 → 62.9 | — (greedy issues 0) |
| code | spec | 0.35 → **0.52** (+49%) | 165.6 → 138.7 (−16%) | 82% / 8% |
| prose | greedy | 1.77 → 2.09 | 35.6 → 19.3 | — |
| prose | spec | 0.56 → **1.17** (+109%) | 75.8 → 34.2 (−55%) | 72% / 10% |
| reasoning | greedy | 1.15 → 1.80 | 39.4 → 27.6 | — |
| reasoning | spec | 0.65 → **1.54** (+137%) | 91.1 → 42.5 (−53%) | 75% / 10% |

Peak memory 10.1–10.3 GB everywhere. Greedy bytes/token bit-identical to the
control (same 117.6/85.2/120.2 MB — issuance is exactly zero on single-token
passes). Output bit-identity ON vs OFF is unit-proven
(`test_expert_prefetch_output_bit_identical_on_vs_off`). Greedy run-to-run
variance remains ±15–20%; same-day spec A/B is the trustworthy comparison.

## Verdict

G-M3 **PASS**. Criteria: spec tok/s up end-to-end on all three workloads
(+49% / +109% / +137%); stall −50%+ on two of three; wasted speculative bytes
8–10% (<20% target); bit-identical output; no greedy regression; no memory
growth. Reasoning spec at **1.54 tok/s** is the best number the project has
produced — the M=7.41 acceptance multiplier finally survives the cache.

Residual: code spec (tight prompt, deep K) still thrashes — its per-pass union
exceeds the region no matter the policy; stall only −16%. M3c (router-probe
lookahead) would not fix a *capacity* problem, so it stays demoted. The
plausible levers are smaller K for code-like prompts or a bigger budget; both
are tuning, not architecture — deferred to M5 benchmarking.

Next: M5 (GPT-OSS-120B headline). Preflight: needs ~65 GB free disk of ~70
available — confirm before packing.
