# Plan 4 — M5 gate summary: GPT-OSS-120B

**Verdict: PASS** — correctness on 16 GB, speed gate (>=1.0 tok/s greedy) met on
community hardware (M1 Max 64 GB: 1.65–1.96 tok/s), with the 16 GB local result
(0.11–0.14 tok/s) understood and documented as capacity-bound, not a defect.

## What M5 had to show

Run mlx-community/gpt-oss-120b-4bit (63.39 GB, 36 layers x 128 experts, 4 fired)
end-to-end through the pack -> stream -> generate path, correct output, and find
what speed is achievable and why.

## Blockers found and fixed on the way

1. **Mixed quantization** — real gpt-oss checkpoints are MXFP4 experts + affine
   8-bit per-module overrides (182 of them). Engine previously assumed one uniform
   quant config. Fixed with a per-path quant resolver (mlx_lm class_predicate
   style) in embed/head/slot construction.
2. **Eager pack OOM** — packer materialized all 59 GB -> macOS jetsam kill on
   16 GB (also hit by the maintainer's manual repack). Fixed: `mlx_lm.load(...,
   lazy=True)` (0.84 GB RSS for the whole model) + per-layer free during save;
   pack peaks ~1.7 GB.
3. **Sliding-window mask bug** — global masks were built from layer-0's
   RotatingKVCache, whose `make_mask` clamps offset to window-1; any multi-token
   pass past the window produced a mask short by (offset-127) columns ->
   `broadcast_shapes` crash. Hit locally AND by a 64 GB community volunteer on the
   published package. Fixed: per-kind mask sources.
4. **Trim rollback unsound after rotation** — `RotatingKVCache.is_trimmable()` is
   False once rotated. Replaced trim-based speculative rollback with
   `engine.verify_forward` (per-layer ephemeral cache clones + recording caches) +
   `commit_verified` (append only accepted rows). Both `speculative_generate` and
   `ngram_speculative_generate` migrated. 235 tests pass; gpt-oss regressions are
   bit-identical to greedy past rotation. Shipped in 0.5.0.

## Results

### Community (Apple M1 Max, 64 GB, nunspark 0.5.0, budget=58GB)

Greedy 1.88 / 1.96 / 1.65 tok/s (code/prose/reasoning), expert hit ~80%,
~450–494 MB/token, peak ~51–55 GB. **Gate met.** Full table:
docs/community-results.md.

### Local (Apple M4, 16 GB, budget=8GB) — scripts/results/m5_120b_greedy_bench.json

Greedy 0.11 / 0.14 / 0.14 tok/s, expert hit 41–53%, 1.4–1.6 GB/token, peak
~12.9 GB. Doubly capacity-starved: (a) expert demand ~1.8 GB/token vs <=9% of the
57 GB expert pool cacheable; (b) non-expert cores (~7–9 GB) alone exceed the whole
budget -> ~10 core re-reads/token (1,061 core misses/run). ~80% of wall time is
sys (mmap fault path ~300 MB/s effective). A 5 GB budget outruns 8 GB (0.17 tok/s)
by easing paging pressure. 16 GB is the correctness floor, not the audience;
32 GB+ with `--budget 24GB+` is the target tier.

### Speculative decoding on sparse MoE: measured dead end

scripts/results/m5_120b_ngram_bench.json (K=16 n-gram/prompt-lookup drafter —
zero model cost, tokenizer-exact, lossless; deviation_rate 0.0 on all runs):

| workload | greedy tok/s | ngram-spec tok/s | M | spec MB/token (greedy) |
|---|---|---|---|---|
| code | 0.11 | 0.05 | 1.30 | 5005 (1602) |
| prose | 0.14 | 0.06 | 1.39 | 4598 (1386) |
| reasoning | 0.14 | 0.04 | 1.15 | 8237 (1648) |

Root cause is structural, not the drafter: each of the K+1 positions in a verify
pass fires its own 4-of-128 experts with low cross-token overlap, so per-pass
expert-union I/O grows ~linearly with K while acceptance does not (M <= 1.39).
Bytes/token 3–5x greedy; expert hit rate halves. The community's model-draft run
(K=24, M 1.19–2.13, spec 0.27–0.59 vs greedy 1.65–1.96) shows the same signature
on a 64 GB machine — this is regime-independent for sparse MoE.

Contrast: dense Llama-3.3-70B community run shows spec WINNING on code (4.57 vs
3.39 greedy) — a dense verify pass re-reads the same weights as one greedy token,
so acceptance is nearly free. **Speculative decoding is the lever for dense
streamed models; selective expert caching (M2/M3) is the lever for MoE.**
Follow-ups: small-K sweep for MoE + bench defaulting spec off for MoE manifests
(docs/backlog.md #5); prefill/TTFT bulk-read optimization (backlog #1).

## Gate decision

- Correctness: PASS (coherent harmony-format output on 16 GB; bit-exact spec paths).
- Speed >=1.0 tok/s greedy: PASS on 64 GB community hardware (1.65–1.96 tok/s);
  documented-unreachable on 16 GB for capacity reasons with the math to prove it.
- Robustness: two engine bugs affecting ALL sliding-window models found via 120B
  and fixed + regression-tested; validated in the field by the 0.5.0 release.

M5 closed. Next: M6 (report.md extension, web UI MoE wiring, README gate).
