# Plan 4 — M6 gate summary: consolidate (report, web UI, docs)

**Verdict: PASS** — README quickstart for Qwen3-30B-A3B verified copy-paste on a
fresh pack; MoE models serve through the web UI with expert telemetry; report and
README updated with the full Plan 4 story.

## What shipped

1. **report.md Part 2** — "MoE Expert Streaming (Qwen3-30B-A3B and GPT-OSS-120B)":
   the M1→M2→M3 milestone ablation (0.51–0.53 → 1.33–1.85 greedy → 1.54 tok/s
   best spec), the M1 verify-pass Jaccard insight, the M3 staging-buffer design
   story, the 120B headline (16 GB correctness floor + community 64 GB
   1.65–1.96 tok/s), the spec-on-MoE negative result, and future work from
   docs/backlog.md.
2. **README** — gpt-oss-120b rows in the headline table (community-verified,
   M1 Max 64 GB, v0.5.0), a "Which model for your RAM" guide (16 GB → 30B MoE /
   32–48 GB → dense 70B + spec / 64 GB+ → 120B), a 120B pack+generate quickstart,
   `--ngram-draft`/`--ngram-max` documented, the greedy-for-MoE/spec-for-dense
   guidance, links to docs/community-results.md.
3. **Web UI MoE wiring** — fixed a fatal bug: runner.py registered
   `pool.release`, which does not exist on EnginePool, so every real
   (non-mocked) job died with AttributeError; tests passed only because they
   mocked the pool. Added a `MagicMock(spec=EnginePool)` regression test.
   `_metrics()` now reports `expert_hit_pct` and `mb_per_token` as per-job
   deltas against a job-start snapshot (the pooled engine's counters persist
   across jobs); frontend renders both when present.
4. **RAM-resident fast path** (bonus, user-requested) — when every streamed
   piece provably fits the cache budget (per-region check for MoE manifests),
   the per-layer `mx.eval(h)` barriers are skipped: they exist only to
   materialize a layer before eviction, and a fully-resident cache never
   evicts. Llama-3.2-1B: 74 → 106 tok/s (~1.43×), bit-identical output
   (tested fast-path on vs off across dense/quant/MoE fixtures). Models that
   exceed the budget take the unchanged streaming path.

## Gate G-M6: README quickstart, copy-paste, fresh pack

- `nunspark pack mlx-community/Qwen3-30B-A3B-4bit ./packed/qwen3-30b` →
  6,201 pieces, 16 GB, 48 layers (matches README's "~6200 files").
- `nunspark generate ./packed/qwen3-30b --budget 8GB --max-tokens 200 --metrics
  --prompt "Explain how a B-tree stays balanced."` → coherent output,
  **1.65 tok/s greedy**, peak 8.89 GB — inside the published 1.33–2.13 band.
- Web serve: `nunspark web --packed-root packed`, job submitted via
  `/api/batch` (greedy, budget=8GB) → status `done`, no error (exercises the
  pool.release fix on a real engine), metrics include
  `expert_hit_pct: 75.2`, `mb_per_token: 314` — MoE telemetry end-to-end.
  (Web `tok_per_s` 0.70 for a 40-token job: its clock includes prefill, so
  short jobs read low; decode-rate parity with the CLI was shown above.)

Test suite: 271 passed; 4 failures pre-existing (3 tests/test_cli.py optional-dep,
1 tests/webapp bad-output-dir), verified identical on the base via stash.

Plan 4 milestones M1–M6 are all closed.
