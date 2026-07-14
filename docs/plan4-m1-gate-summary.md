# Plan 4 — M1 Gate Summary: Expert Locality on Qwen3-30B-A3B-4bit

Baselines: 3 prompts (code/prose/reasoning) × (greedy, spec K=24 w/ Qwen3-0.6B),
500 tokens each, 8 GB piece-cache budget, packed model = 48 layers ×
(10.8 MB core + 128 × 2.65 MB experts), embed 175 MB. Raw data:
`scripts/results/m1_baseline.json`, traces + per-trace analyses in `scripts/results/`.

## Baseline performance

| workload | mode | tok/s | acceptance M | expert hit% | bytes/token |
|---|---|---|---|---|---|
| code | greedy | 0.51 | — | 57.2% | 916 MB |
| code | spec | 0.50 | 3.85 | 56.9% | 769 MB |
| prose | greedy | 0.53 | — | 63.8% | 759 MB |
| prose | spec | 0.33 | 2.30 | 62.8% | 1013 MB |
| reasoning | greedy | 0.52 | — | 58.7% | 875 MB |
| reasoning | spec | 0.86 | 6.85 | 57.5% | 463 MB |

Peak unified memory ~10.5–10.8 GB in all runs. Per-token expert demand is
8.13 experts × 48 layers × 2.65 MB ≈ **1.03 GB/token**; at the measured 57–64%
hit rate that yields the ~0.8–1.0 GB/token of misses that pins throughput at ~0.5 tok/s.

## Locality findings → milestone decisions

**M2 (expert-aware cache): GO — the biggest single win on the board.**
The offline sim (LRU over the real trace) reaches **84% hit at 5.2 GB and 94% at
7.8 GB of expert capacity** — versus the 57% the live MRU policy achieves at an
8 GB budget. The MRU policy (correct for the dense cyclic scan) is actively wrong
for experts. LFU-with-decay ≈ LRU in the sim (within 0.3% everywhere), so a simple
segmented-LRU expert region is sufficient — no frequency machinery needed.
Global expert skew is mild (top-10% experts = 13–17% of firings) but per-layer skew
is strong (top-10% share 25–71% by layer): pinning, if used, must be per-layer.
Cores + embed + head total < 1 GB → pin them all unconditionally.
Projected: expert misses drop from ~900 MB/tok to ~100–200 MB/tok at the same
budget → **~3–6× greedy tok/s**, before M3 overlap.

**M3a (temporal expert prefetch): GO — for the spec path, which is the one that matters.**
Per-token (greedy) consecutive Jaccard is low (0.29–0.31): predicting the *next
token's* experts from the last token is weak. But per-verify-pass (spec) consecutive
overlap is **0.74–0.80** (consistent across all three spec traces): the union fired
by one K=24 verify pass strongly predicts
the next pass's union. M3a should prefetch at verify-pass granularity (previous
pass's per-layer fired sets), not per token. M3c (router-probe) stays demoted unless
M3a's residual stall is large.

**M4 (selective experts in tree/window verify): GO.**
Fired-union at window K=25 is **43–50 experts/layer (34–39%) on greedy traces** and
66–83 (51–65%) per spec verify pass — far below all 128. `tree_forward`'s
load-all-experts path wastes 2–3× the necessary expert bytes. The window path
through `forward()` is already selective (verified by trace batch_tokens).

**Spec-decode note:** acceptance still dominates variance (M=2.3 prose → spec *slower*
than greedy; M=6.85 reasoning → 1.7× greedy). Draft choice/tuning remains a lever on
top of everything above, consistent with report.md.

## Gate verdict

G-M1 **PASS**. Order of attack confirmed: M2 first (largest, simplest win),
M3a second (verify-pass-granularity prefetch), M4 in parallel (Sonnet-lane),
M3c only if M3a leaves a big residual stall.
