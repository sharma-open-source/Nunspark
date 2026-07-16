# Community benchmark results

Consolidated `nunspark bench` shareable reports from community volunteers, plus the
maintainer's local reference numbers. Newest first within each model. All community
runs use published PyPI releases (version noted per report).

Reading the table: `M` is the speculative acceptance multiplier (tokens advanced per
verify sweep); `expert hit%` / `MB/token` only apply to MoE models; dense models
show `—`.

---

## mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit (MoE, 62L x 160 experts, 8 fired)

### Apple M5 Max, 128 GB RAM, macOS 26.5 — nunspark 0.7.0, mlx 0.32.0 (2026-07-16)

| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |
|---|---|---|---|---|---|---|
| code | greedy | 0.09 | — | 63.4 | 5325 | 98.58 |
| code | spec | 0.11 | 5.00 | 38.9 | 29406 | 99.20 |
| prose | greedy | 0.17 | — | 82.0 | 2655 | 98.59 |
| prose | spec | 0.07 | 2.17 | 49.8 | 29654 | 99.19 |
| reasoning | greedy | 0.12 | — | 74.3 | 3835 | 98.61 |
| reasoning | spec | 0.08 | 2.86 | 39.6 | 39039 | 99.26 |

settings: budget=90GB, max_tokens<=100, speculative=on, K=24 (default Qwen3-0.6B draft)

Notes:
- Largest model run to date: ~270 GB pack on 128 GB RAM (~2.1x over-commit) —
  **runs correctly but is capacity-bound**, 0.09–0.17 tok/s greedy at 2.7–5.3
  GB/token of expert reads. Same structural regime as gpt-oss-120b on 16 GB:
  the 90 GB budget covers roughly a third of the expert pool, so miss volume
  dominates.
- Spec arm reconfirms the MoE verify-union tax at the largest scale yet
  (backlog #5): K=24 blows expert reads to 29–39 GB/token and halves the hit
  rate (82→50, 74→40, 63→39). Code's M=5.00 still only manages 0.11 vs 0.09
  greedy — the union tax eats the entire acceptance win. Third independent
  dataset (local 30B, community 120B, now 480B) all pointing at defaulting the
  spec arm OFF (or small-K `--ngram`) for MoE manifests.
- Interesting inversion vs smaller MoEs: prose is the *fastest* workload here
  (highest expert reuse, 82% hit), while code is slowest — expert diversity on
  code prompts is brutal at 160 experts/layer.

## mlx-community/gpt-oss-120b-MXFP4-Q8 (MoE, 36L x 128 experts, 4 fired)

### Apple M5 Max, 128 GB RAM, macOS 26.5 — nunspark 0.7.0, mlx 0.32.0 (2026-07-16)

| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |
|---|---|---|---|---|---|---|
| code | greedy | 0.84 | — | 67.2 | 749 | 28.76 |
| code | spec | 0.51 | 1.12 | 81.5 | 1927 | 28.91 |
| prose | greedy | 1.31 | — | 74.5 | 587 | 28.77 |
| prose | spec | 0.61 | 1.52 | 69.9 | 2474 | 28.92 |
| reasoning | greedy | 0.97 | — | 69.6 | 704 | 28.79 |
| reasoning | spec | 0.50 | 1.01 | 86.0 | 1335 | 28.94 |

settings: budget=24GB, max_tokens<=100, speculative=on, K=24 (default Qwen3-0.6B draft)

Notes:
- Same volunteer as the 480B run above. Different checkpoint variant than the M1 Max
  run below (MXFP4-**Q8** vs 4bit — larger expert bytes per miss).
- Budget deliberately small (24 GB on a 128 GB machine): greedy 0.84–1.31 tok/s at
  67–75% expert hit, vs the 64 GB volunteer's 1.65–1.96 at ~80% hit with budget=58GB.
  Consistent picture: on this model tok/s tracks expert cache coverage almost
  directly — a 58GB+ budget on this machine should meet or beat the M1 Max numbers.
- Spec arm loses again (fourth dataset): M=1.01–1.52 (tokenizer mismatch → mostly
  bonus-token) while verify-union reads run 1.3–2.5 GB/token. Reinforces backlog #5.

| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |
|---|---|---|---|---|---|---|
| code | greedy | 1.88 | — | 80.2 | 478 | 53.90 |
| code | spec | 0.32 | 1.19 | 95.9 | 568 | 59.14 |
| prose | greedy | 1.96 | — | 81.4 | 450 | 51.04 |
| prose | spec | 0.59 | 2.13 | 91.5 | 521 | 58.55 |
| reasoning | greedy | 1.65 | — | 79.8 | 494 | 55.49 |
| reasoning | spec | 0.27 | 1.30 | 95.4 | 555 | 59.18 |

settings: budget=58GB, max_tokens<=100, speculative=on, K=24 (default Qwen3-0.6B draft)

Notes:
- **Headline: 1.65–1.96 tok/s greedy for a 59 GB model on 64 GB RAM.** M5 target
  (>=1.0 tok/s on community hardware) met.
- Confirms the 0.5.0 sliding-window mask fix in the field — the same volunteer hit
  the `broadcast_shapes (25,152)/(1,64,25,154)` crash on the previous release.
- Spec arm slower than greedy, two compounding causes: (1) default Qwen3-0.6B draft
  does not share the gpt-oss tokenizer, so M≈bonus-token only; (2) MoE verify-pass
  expert-union tax — each of K=24 positions fires its own 4/128 experts with low
  overlap (568 vs 478 MB/token, peak 59 GB → paging pressure). Use `--ngram` and a
  smaller K for gpt-oss spec runs.

## mlx-community/Llama-3.3-70B-Instruct-4bit (dense, 70B)

### Apple M1 Max, 64 GB RAM, macOS 26.5.2 — nunspark 0.5.0, mlx 0.32.0 (2026-07-15)

| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |
|---|---|---|---|---|---|---|
| code | greedy | 3.39 | — | — | — | 40.48 |
| code | spec | 4.57 | 14.29 | — | — | 40.79 |
| prose | greedy | 3.00 | — | — | — | 40.49 |
| prose | spec | 1.57 | 4.00 | — | — | 40.82 |
| reasoning | greedy | 3.22 | — | — | — | 40.50 |
| reasoning | spec | 2.42 | 7.69 | — | — | 40.83 |

settings: budget=48GB, max_tokens<=100, speculative=on, K=24 (default Qwen3-0.6B draft)

Notes:
- Volunteer originally attempted Qwen3-235B but it destabilized the machine
  ("connection became unstable"); switched to the 70B.
- First community **spec win**: code 4.57 vs 3.39 greedy (M=14.29) — consistent with
  the thesis that speculative decoding is the right lever for *dense* streamed models.
- OPEN QUESTION: prose (M=4.00) and reasoning (M=7.69) spec arms are *slower* than
  greedy despite healthy acceptance. For a dense model a K-token verify pass should
  cost about one greedy token of weight I/O, so M=4 should be ~faster, not 0.5x.
  Also suspicious: Qwen3-0.6B and Llama-3.3 tokenizers differ, so M values this high
  need explanation. Needs local reproduction — see backlog.

## mlx-community/Qwen3-30B-A3B-4bit (MoE, 48L x 128 experts, 8 fired)

### Apple M1 Max, 64 GB RAM, macOS 26.5.2 — nunspark 0.5.0, mlx 0.32.0 (2026-07-15)

| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |
|---|---|---|---|---|---|---|
| prose | greedy | 1.60 | — | 77.2 | 285 | 8.91 |

settings: budget=8GB, max_tokens<=50, speculative=off

Notes:
- Matches the maintainer's M4 16 GB numbers at the same 8 GB budget (1.33–2.13 tok/s
  greedy) — the budget, not the machine's total RAM, governs MoE streaming speed.

---

## Maintainer reference (Apple M4, 16 GB RAM, 460 GB SSD)

### gpt-oss-120b, nunspark git version2.0 (2026-07-15)

Greedy floor (`scripts/results/m5_120b_greedy_bench.json`): 0.106 / 0.113 / 0.102
tok/s (code / prose / reasoning) @ budget=8GB, peak ~12.8 GB, prefill 57–95 s.
A 5 GB budget measured faster (0.17 tok/s) — less paging pressure. On 16 GB the
model is doubly capacity-starved (expert demand ~1.8 GB/token vs <=9% pool cached;
non-expert cores ~7–9 GB alone exceed the budget). 120B's real audience is
32 GB+ machines with `--budget 24GB+`.

n-gram spec (K=16) final (`scripts/results/m5_120b_ngram_bench.json`):

| workload | mode | tok/s | M | expert hit% | MB/token |
|---|---|---|---|---|---|
| code | greedy | 0.11 | — | 43.1 | 1602 |
| code | ngram-spec | 0.05 | 1.30 | 24.5 | 5005 |
| prose | greedy | 0.14 | — | 52.8 | 1386 |
| prose | ngram-spec | 0.06 | 1.39 | 22.3 | 4598 |
| reasoning | greedy | 0.14 | — | 41.7 | 1648 |
| reasoning | ngram-spec | 0.04 | 1.15 | 26.2 | 8237 |

The MoE verify-union tax loses even with a zero-cost, tokenizer-correct drafter:
3–5x bytes/token vs greedy, expert hit rate halves, M never exceeds 1.39.
deviation_rate = 0.0 everywhere — output bit-identical to greedy (lossless
acceptance verified end-to-end on rotating caches). Conclusion in
`docs/plan4-m5-gate-summary.md`: speculative decoding is the wrong lever for
sparse-MoE streaming; it is the lever for dense streamed models.

### Qwen3-30B-A3B, M3 gate (2026-07-14)

Spec (Qwen3-0.6B draft) 0.52 / 1.17 / **1.54** tok/s vs 0.35 / 0.56 / 0.65 control
(code / prose / reasoning) @ budget=8GB. Details: `docs/plan4-m3-gate-summary.md`.
