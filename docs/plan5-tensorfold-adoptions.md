# Plan 5 — TensorFold-inspired adoptions

**Origin.** TensorFold (github.com/ashhart/TensorFold, MIT) is an independent MoE
weight-streaming runtime on MLX/Apple Silicon — same problem, same physics, and its
honest numbers cross-validate ours (pure streaming 2.4–2.7 tok/s vs 80 full-resident;
full-resident ~9× faster even batched). Three of its ideas survive review against our
codebase; one of its negative results (predictive expert prefetch from the previous
token = net slowdown at ~43% consecutive-token overlap) independently confirms our
Phase-1 "warming is not the lever for decode" verdict and needs no work.

| # | Adoption | TensorFold evidence | Our gap |
|---|----------|--------------------|---------|
| A1 | Garbage-drafter invariant test | "a drafter with nothing ever accepted must reproduce baseline greedy token-for-token" — isolates rollback correctness from drafter quality | `verify_forward`/`commit_verified` have NO covering tests (codegraph blast-radius check); the kv.truncate class of bugs (backlog #4) would be caught by exactly this test |
| A2 | Adaptive proposal shrink (n-gram drafter) | adaptive shrink "caps the worst (novel-text) case near break-even"; 15% acceptance → no slowdown | our K=16 n-gram spec loses 2–4× on MoE (backlog #5, m5_120b_ngram_bench.json); `NGramDrafter` is stateless with fixed K |
| A3 | Batched decode sharing per-layer expert unions | batch 16: 3.4 → 49 tok/s TOTAL, peak 1.97 → 2.64 GB — weight sweep amortized across requests | engine decodes strictly B=1; web UI "batch" runs prompts sequentially, each paying the full sweep |

**Orchestration model** (same as Plan 4). Fable orchestrates; implementers are subagents:
Sonnet for well-specified mechanical work, Opus for design-sensitive engine work.
Every milestone ends with a **gate** checked by the orchestrator before the next opens.
Working branch: `version2.0`. Nothing is committed — the user owns commits.

**Standing rules for all implementer prompts**
- Python 3.11 via `uv venv --python 3.11`; run tests with plain `uv run` (never `--active`).
- Bit-identical output is the invariant: lossless spec paths must match `generate()`
  exactly. Known pre-existing failures to ignore: 3 optional-dep tests in
  tests/test_cli.py + tests/webapp/test_runner.py::test_run_generation_bad_output_dir_does_not_raise.
- Stock-mlx_lm compatibility over arch hacks; registry-wide, no per-model special cases.
- Small diffs, each with a test; benchmarks write JSON to `scripts/results/`.
- Never touch `pyproject.toml` version; never commit.

---

## M1 — Garbage-drafter invariant test (A1)

*Implementer: Sonnet. Tests only — no engine changes.*

The hard invariant, quoted from TensorFold: a drafter whose proposals are NEVER
accepted must still reproduce baseline greedy output token-for-token. This isolates
the verify/commit/rollback machinery (`engine.verify_forward` + `engine.commit_verified`,
draft-cache `trim`) from drafter quality: any state leak from rejected tokens shows up
as a byte diff.

1. **Adversarial n-gram drafter test.** A stub drafter whose `propose()` returns
   deliberately-wrong tokens (e.g. `[(argmax+1) % V] * K`, or fixed junk ids) —
   `ngram_speculative_generate` output must equal `generate()` exactly, and
   `SpecStats.accepted_total == 0`.
2. **Adversarial model-draft test.** A stub draft model (callable, with a trimmable
   mlx_lm-style cache) that always proposes wrong tokens — `speculative_generate`
   at `accept_top_k=1` must equal `generate()` exactly. This exercises the
   draft-cache `trim(drop)` rollback path every round at m=0.
3. **Mixed-acceptance variant.** A drafter that alternates right/wrong proposals, so
   commits happen at varying m — catches off-by-one in `commit_verified(kv, recs, m+1)`.
4. Run on the small test fixtures the existing suite uses (incl. the gpt-oss
   sliding-window fixture if one exists, since RotatingKVCache rotation is the
   documented risk (backlog #4)).

**Gate G-M1:** new tests pass; full suite green (minus known failures); the tests fail
if `commit_verified`'s count is deliberately perturbed by ±1 (mutation check, done once
manually by the implementer and reported).

---

## M2 — Adaptive proposal shrink for the n-gram drafter (A2)

*Implementer: Sonnet (M1's tests are the safety net; adaptivity is output-safe by
construction — lossless acceptance emits only target-argmax tokens, so proposal LENGTH
cannot change output bytes, only cost).*

Mechanism (TensorFold-style, adapted to our loop):
1. `NGramDrafter` becomes stateful: `k_cur` starts at `num_draft_tokens` (cap);
   new method `observe(proposed: int, accepted: int)` called by
   `ngram_speculative_generate` after each verify round with (Kq, m).
2. Policy — multiplicative decrease, slow recovery:
   - full/near-full acceptance (`accepted >= proposed - 1`): `k_cur = min(cap, k_cur * 2)`
   - poor round (`accepted < proposed // 4`): `k_cur = max(k_min, k_cur // 2)`
   - otherwise hold. `k_min = 2` (a 1-token proposal can't amortize a sweep).
   Empty proposals (no n-gram match) don't call observe — no signal.
3. `propose()` caps its return at `k_cur` instead of `num_draft_tokens`.
   Default ON for the ngram path; `--ngram-adaptive/--no-ngram-adaptive` CLI escape hatch.
4. SpecStats: record `k_cur` trajectory min/max (extend AdaptiveSpecStats or add two
   ints) so bench reports show the adaptation worked.
5. **A/B bench** on Qwen3-30B-A3B, 8 GB budget, the three report.md workloads,
   ngram spec K-cap=16, fixed vs adaptive, JSON to
   `scripts/results/ngram_adaptive_ab.json`.

**Gate G-M2:** bit-identical output fixed vs adaptive vs greedy on all three workloads;
novel-prose (worst case, currently 2–4× LOSS per backlog #5) improves to ≥ break-even-ish
(≤1.2× slower than greedy); high-M workloads regress ≤5%. If the worst case doesn't
reach ~break-even, report the measured k_cur trajectory and stop for a policy rethink —
do not tune blindly.

---

## M3 — Batched decode sharing per-layer expert unions (A3)

*Design-sensitive. M3a is an Opus DESIGN PASS (no code) gated before implementation.*

The prize: in the disk-bound regime the weight sweep is the cost; B sequences decoded
in lock-step share each layer's core read and the UNION of their fired experts, so
total tok/s scales ~linearly in B while bytes/token collapses (TensorFold: 3.4 → 49
total tok/s at B=16, memory nearly flat). Stacks conceptually with deep-K spec
(amortize across draft tokens) — but M3 targets the multi-request web-UI/serve
workload first, greedy only.

**M3a — design memo** (Opus, read-only): how `engine.forward` handles `[B, 1]` input
today; whether `_moe_layer_forward` expert gather already unions across batch rows;
KVStore shape — per-sequence stores vs one batched store with per-row offsets
(sequences have different lengths — masks/padding? mlx_lm cache semantics?);
prefill strategy (sequential per-request prefill into row slots vs padded batch
prefill); web runner integration (currently sequential loop); failure isolation
(one bad request must not kill the batch); scope cut for v1 (fixed B, all requests
same max_tokens? dynamic joins later). Deliverable:
`docs/plan5-m3-design.md` with a chosen design + risks + test plan.
**Gate G-M3a:** orchestrator (and ideally user) signs off on the memo.

**M3b — implement** (Opus for engine, Sonnet for web-runner plumbing + tests), per memo.

**Gate G-M3:** each batched sequence bit-identical to its sequential run; B=4 total
tok/s ≥ 2× sequential total on Qwen3-30B-A3B @ 8 GB budget; peak memory growth < 25%.

---

## Sequencing

M1 first (it is the safety net). M2 after G-M1. M3a (design, read-only) may run in
parallel with M1/M2. M3b only after G-M1, G-M2, G-M3a.

## Gate outcomes (2026-07-16/17)

- **G-M1 PASSED** — 6 invariant tests incl. gpt-oss sliding-window, mutation check
  confirmed (perturbing the commit count fails exactly the ngram-path tests).
- **G-M2 v1 FAILED, then PASSED at v4 after two findings.** (1) The A/B mismatch led
  to the fp16 investigation (docs/plan5-m2-mismatch-investigation.md): verify_forward
  == same-shape forward BITWISE — the machinery is exact; "bit-identical to greedy"
  is false cross-shape at model scale (~1 near-tie flip/110 tokens, self-healing) —
  claims softened repo-wide, SpecStats.near_tie_rows added. (2) Shrink-to-floor
  policy replaced by disable-to-zero + 50-round re-probe; a latent int-division bug
  (`accepted < proposed // 4` unreachable at proposed<=3) initially made disable
  unreachable — fixed as `accepted * 4 < proposed`. Final A/B (ngram_adaptive_ab_v4):
  adaptive M~1.01-1.06, k reaches 0, tok/s vs greedy: code 1.28/1.07, prose
  1.64/1.87, reasoning 1.01/1.09 — within the <=1.2x gate; fixed-K still loses
  2.5-3.5x. Adaptive n-gram is now a never-loses default.
- **G-M3a PASSED** with the token-identical amendment (plan5-m3-design.md §14) —
  later empirically vindicated at fixture scale (§15).
- **G-M3b-1 PASSED** — engine core landed, mask correctness proven directly
  (§15); 341 tests green.
- **G-M3b-2 ACCEPTED at B=8** — 2.30x total throughput at +8.4% peak memory; the
  literal B=4 target measured 1.4-1.5x twice. Expert union saturates at ~23/128 from
  B=4, so larger B wins harder (§16). Default batch size guidance: 8.
