# Plan 5 / M2 — spec-vs-greedy mismatch investigation

**Verdict: H1 (shape-dependent floating-point numerics, NOT a bug).**

`ngram_speculative_generate` is *lossless in the target-verified sense* (it emits
only tokens that are the target's own verify-pass argmax) and `verify_forward` /
`commit_verified` are numerically correct. But its output is **not** byte-identical
to greedy `generate()` on the real fp16 model: greedy and spec build the KV cache
through **different forward-pass shapes**, which produces fp16-scale differences in
the cached K/V at every position, and those differences occasionally flip a
sub-0.15-logit **near-tie** in the target's argmax. This is precisely the effect the
orchestrator predicted for batched decode in `docs/plan5-m3-design.md` §14 (the
"TOKEN-identical, not bit-identical" amendment, lines 324–343).

All experiments: `./packed/qwen3-30b` (Qwen3-30B-A3B-4bit, qwen3_moe, 48 layers),
budget 8 GB (`_fully_resident = False`, confirmed — the 30B does not fit 8 GB),
`nunspark.bench.PROMPTS[0]` (code, 36 prompt tokens), temp 0, eos 151645, fixed-K=16.
Scratch scripts under the session scratchpad; nothing in `src/`/`tests/` was touched.

---

## 1. The index-79 divergence, per-round mechanics

Greedy and fixed-K spec agree on output indices 0–78, diverge at 79, and
**re-converge at 80** (both emit `13` = `.`):

| index | 78 | **79** | 80 | 81 |
|---|---|---|---|---|
| greedy | 323 | **8837** (` updates`) | 13 | 4710 |
| spec   | 323 | **17030** (` modification`) | 13 | 4710 |

The single flip is isolated and self-healing — not a cascade.

The round that emits index 79 (from the instrumented run):

- persistent kv `offset = 114` (= 36 prompt + 78 committed emitted tokens, all on the greedy path)
- bootstrap `b = 323`; drafter proposed `q` = 16 tokens `[2182, 7525, 13, 6771, …]`
- verify input = `[323] + q` (17 tokens, one `[1,17]` pass)
- **accepted m = 0** (drafter's `q[0]=2182` ≠ verify row-0 argmax), so index 79 is the
  **BONUS** token = row-0 argmax of the verify pass = **17030**.

So the spec arm emitted 17030 **because the verify pass's own argmax was 17030** — the
lossless-acceptance invariant is intact (the emitted token *is* the target argmax; the
accept-only-if-equals-argmax logic did exactly the right thing). The verify pass's
own float32 row-0 logits at that position:

```
17030 = 32.6411   (argmax / bonus, emitted by spec)
 8837 = 32.5592   (arg2 — the verify pass's SECOND choice — this is greedy's token)
gap  = 0.082
```

A genuine near-tie: the two candidates are 0.082 apart and the greedy token is the
verify pass's exact runner-up. The finding's independent fresh chunked-prefill
recompute ranks them the *other* way (8837 = 32.6402 vs 17030 = 32.4912, gap 0.149).
Both gaps are well under 0.5.

---

## 2. Discriminating experiment — verify_forward vs forward vs stepwise (bitwise)

Built a clean kv by chunked-prefilling `prompt + greedy[:78]` (offset 114), then on
three independently-built engines+kv ran the divergent round's 17-token input:

| arm | row-0 argmax | max-abs-diff vs stepwise |
|---|---|---|
| (a) `verify_forward([1,17])` | 8837 | 1.738 |
| (b) `forward([1,17])`        | 8837 | 1.738 |
| (c) stepwise `forward([1,1])`×17 | 8837 | — |

**(a) == (b) BITWISE** — `mx.array_equal(A, B) = True`, **max-abs-diff = 0.0** across
all 17×vocab logits. Same shape + same cache state ⇒ identical kernels ⇒ identical
bits. This is H1's exact prediction and **rules out H2**: `verify_forward`'s ephemeral
clone + `_RecordingCache` + `cache_override` path computes precisely what `forward()`
computes; there is no offset/mask/expert-cache bug in verify.

(a)/(b) differ from (c) stepwise by up to 1.738 logits (row-level shape-dependent
numerics are real and large), but note: **this clean replay did NOT reproduce the
flip** — all three arms give row-0 argmax 8837 (agreeing with greedy). Row-0 detail:

```
(a) verify   8837=32.6399  17030=32.4910  -> 8837
(b) fwd[1,17] 8837=32.6399  17030=32.4910  -> 8837   (bit-identical to a)
(c) stepwise 8837=32.5349  17030=32.4973  -> 8837
```

Same 17 tokens, same offset, same `[1,17]` shape as the live run — yet the live verify
pass reported 17030 (32.641) and this replay reports 8837 (32.640). The only remaining
difference is the **KV-cache contents**: the numerical history of how offset-114 was
reached.

---

## 3. Root cause — fp16 rounding accumulated in the KV cache

Compared the offset-114 KV built two ways:

- **clean**: one chunked `_prefill` of all 114 tokens (mostly a single `[1,114]` pass)
- **spec**: the live `ngram_speculative_generate` path — `[1,36]` prompt prefill, then
  the generated tokens' K/V written by `[1,17]` verify passes (`commit_verified`) and
  `[1,1]` greedy-fallback steps.

Result (per-layer K/V, |K| ≈ 43):

```
KV max-abs-diff clean vs spec, all 48 layers: 1.292 (layer 29 K)
worst-layer K per-position diff: min 5.5e-4, max 1.292, nonzero at 114/114 positions
per-position diff grows toward later positions (first 5 ≈ 5e-4–0.10, last 5 ≈ 0.04–0.19)
```

The difference is **spread smoothly across every position** and **accumulates with
position** — the signature of fp16 rounding compounding through differing pass shapes.
It is **not** a localized block of ~100 % diffs at a shifted position (which an
off-by-one in `commit_verified` would produce) and **not** zero-elsewhere. So
`commit_verified` writes exactly the (correct) K/V the verify pass computed; those K/V
are simply fp16-different from what `[1,1]` greedy stepping would compute for the same
tokens. Propagated through 48 layers of attention, that ~1e-3–1e-1 K/V shift moves the
final logits by ~0.1–0.15 and flips the near-tie.

The flip is not even deterministic w.r.t. the token sequence: the clean `[1,114]`
prefill yields a *third* KV state and reproduces greedy's answer (8837), while only the
live spec numerical path lands on 17030.

**Mechanism, end to end:** greedy KV (built by `[1,1]` steps) ≠ spec KV (built by
`[1,36]` + `[1,17]` + `[1,1]` passes) at the fp16 level at every position → attention
over those K/V shifts the target logits by ~0.1 → a sub-0.15-gap argmax near-tie flips
→ one emitted token differs. Rare, small-gap, self-healing. Textbook H1.

---

## 4. Flip rate and gap distribution

Over the 110-token run, walking each verify pass's argmax rows along the greedy path
(a row is a "flip" when its argmax ≠ the greedy token at that global position, counted
only while the row's prefix still matches greedy):

- **on-path verify rows checked: 53; flips: 1 ⇒ 1.9 flips / 100 on-path rows**
- **emitted-token divergences: 1 in 110 tokens**
- the one flip: verify-pass top-2 gap **0.082** (fresh recompute 0.149); greedy token
  was the verify pass's exact 2nd choice (`arg2[0] == 8837`).

Both independent gap estimates are < 0.5 — consistent with rare near-tie flips, not
systematic drift. (Broader gap statistics would need a longer run; a `near_tie_events`
counter, see §6, would make this a standing measurement.)

---

## 5. Sanity — divergence can only enter via verify rounds

The empty-drafter fallback round (`q == []`, 56 of them in this run) is
`generate.py:444`:

```python
nl = engine.forward(mx.array([b])[None], kv=kv)[:, -1, :]
```

byte-for-byte the same call shape as greedy's decode step at `generate.py:153`:

```python
logits = engine.forward(mx.array([nxt])[None], kv=kv)[:, -1, :]
```

Both are `[1,1]` `forward` passes against the persistent kv, so a fallback round is
bit-identical to a greedy step. Divergence can therefore enter **only** through the
`[1,K+1]` verify passes — exactly where the shape changes — which is what §2–§3 show.

`_cur_pass_multi` (gates bulk-warm / speculative expert prefetch on multi-token
passes) is I/O-only and does not enter the numerics; `_fully_resident` is False at
8 GB so per-layer `mx.eval` runs, and `mx.eval` is scheduling-only regardless. Nothing
suspicious in either.

---

## 6. Recommended actions (H1)

### 6a. Reword the cross-shape "bit-identical" claims for SPEC paths (do NOT edit — list only)

Two *different* claims live in the repo and must not be conflated:

- **Same-shape determinism** — "engine forward pass bit-identical to full-load mlx-lm"
  (`README.md:17`, `README.md:134`), and same-arm/same-config rerun determinism
  (community 19-run study). These are **per-shape** and remain **TRUE**. Leave as-is.
- **Cross-shape** — spec output "bit-identical to greedy `generate()`". This is
  **FALSE** on the real fp16 model because greedy and spec build the KV through
  different pass shapes. These are the sites to soften to something like: *"lossless in
  the target-verified sense (emits only the target's argmax token); token-identical to
  greedy except for rare characterized fp near-tie flips — not guaranteed
  byte-identical, because spec builds the KV cache through different pass shapes than
  single-token decode."*

Claim sites to reword (file:line):

- `src/nunspark/generate.py:230-231` — `speculative_generate` docstring "output is bit-identical to greedy `generate()`"
- `src/nunspark/generate.py:385-386` — `ngram_speculative_generate` docstring "output is bit-identical to plain `generate()`"
- `src/nunspark/generate.py:403-404` — "acceptance is lossless, so proposal LENGTH cannot change which tokens are emitted" (true for a *fixed* numerical path, but fixed-vs-adaptive change proposal length ⇒ change verify shapes ⇒ can themselves flip near-ties; see §6c)
- `README.md:16` — "Lossless: output is bit-identical to running the target alone."
- `README.md:33` — "'lossless' means the streamed model produces exactly the tokens the full-RAM model would."
- `README.md:88` — "it's lossless: the target model verifies every draft"
- `docs/community-results.md:256` — "deviation_rate = 0.0 everywhere — output bit-identical to greedy (lossless…)"
- `docs/plan4-m5-gate-summary.md:33` ("bit-identical to greedy past rotation"), `:56`
- `docs/plan4-moe-streaming.md:30` — "Bit-identical output is the invariant for engine changes"
- `docs/plan5-tensorfold-adoptions.md:24` ("lossless spec paths must match `generate()`"), `:86` (Gate G-M2)

`docs/plan5-m3-design.md:324-347` already states the honest contract for batched decode
("token-identical … not guaranteed byte-identical … per-row output is target-greedy
correct") — the spec-path docs should adopt the same wording.

### 6b. Grow SpecStats a near-tie / deviation counter

`SpecStats.deviation_rate` today only tracks `accept_top_k > 1` off-path accepts
(`accepted_offpath`), which is 0 here — it does **not** capture this shape-numerics
flip. Add a TensorFold-style `near_tie_events` counter (exactly as
`plan5-m3-design.md:330,339` recommends): per verify row, increment when
`top1 - top2 < threshold` (e.g. 0.5). Optionally, when a greedy reference is available,
count actual emitted-token divergences. This turns the honest contract into a measured,
reportable quantity instead of an unverifiable claim.

### 6c. Fix the unachievable G-M2 bit-identity gate

`scripts/bench_ngram_adaptive_ab.py` asserts `greedy == fixed == adaptive` out_ids, and
the plan's Gate G-M2 says "bit-identical output fixed vs adaptive vs greedy". This is
**not achievable on a real fp16 model** and is why "all three A/B arms pairwise
disagreed on all three bench workloads": adaptive changes proposal length → changes
verify-pass shapes → independently flips near-ties, so even fixed-vs-adaptive is not
guaranteed identical. It passes only on tiny fixtures where no near-tie is hit. Retarget
the gate to **token-identical-modulo-characterized-near-ties** (every diff shown to be a
sub-threshold top-2 gap), mirroring the M3b-1 amendment already in the repo.

### No H2 fix required

`verify_forward` == `forward` bitwise at fixed shape+state (§2), and `commit_verified`
writes correct, non-structural K/V (§3). There is no bug to fix in verify/commit.
