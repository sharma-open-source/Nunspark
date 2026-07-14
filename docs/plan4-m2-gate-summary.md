# Plan 4 — M2 Gate Summary: Two-Region Expert Cache

Change: `PieceCache` split into two regions — dense/core pieces keep the original
MRU (cyclic-scan) eviction; expert pieces get plain LRU with an `expert_frac`
budget split. All core pieces + embed/head are pinned unconditionally at engine
construction (computed from the manifest, <1 GB for the 30B). Plumbed as
`--expert-cache-frac`. Tests: 225 passed (8 pre-existing tiktoken env failures).

## Gate benchmark (greedy, 200 tokens, 8 GB budget — vs M1 500-token baseline)

| workload | tok/s M1 → M2 | expert hit% M1 → M2 | MB/token M1 → M2 |
|---|---|---|---|
| code | 0.51 → **1.33** (2.6×) | 57.2 → **89.2** | 916 → **118** |
| prose | 0.53 → **1.85** (3.5×) | 63.8 → **92.2** | 759 → **85** |
| reasoning | 0.52 → **1.50** (2.9×) | 58.7 → **89.0** | 875 → **120** |

Core hit rate 99.5% (pinning works). Peak memory 8.9 GB (was 10.5).
The offline sim predicted 84–94% expert hit at this budget; live result 89–92% —
the model was right, MRU-for-experts was the entire problem.

## Verdict

G-M2 **PASS** (target was "material improvement"; got 2.6–3.5× tok/s, ~8× fewer
bytes/token). Remaining miss traffic is ~85–120 MB/token, still serial after the
router per layer → M3a (verify-pass-granularity temporal prefetch) is now the
binding constraint, exactly as the plan sequenced.
