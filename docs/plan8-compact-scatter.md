# Plan 8 — Compact fired-only expert scatter (backlog #16 / #10 candidate (a))

Status: M0 PASSED 2026-07-27 (4/4 bit-identical, tiny qwen3_moe + gpt_oss ×
fp16/4-bit — tests/test_compact_scatter_numerics.py). M1 IMPLEMENTED 2026-07-27
(engine flag + CLI/serve plumbing + regression tests; test_cli expected-dict
updated for the new `compact_scatter=False` kwarg — pending a green suite run on
the user's env). M2 RAN 2026-07-27 — **GATE NOT CLEARED for greedy decode: Plan 8 STOPS here as
scoped.** The shipped compact mechanism (zeros+setitem) was a +26% regression;
only compact-via-`mx.stack` beats the default, and even that is ~3-4% end-to-end
at greedy k=8 (under the M3 ≥10% bar). M1 flag mechanism corrected to stack and
left default-OFF; M3 NOT run. Verify-union k had a −30% win → parked as backlog
#17 (speculative-verify scatter), a separate direction. See M2 RESULT below.

## Motivation (all measured)

The wired-baseline decode attribution (backlog #16,
scripts/results/decode_attribution_wired.json; Qwen3-30B, 10 GB wired, clean
7.9 GB-compressed run) flipped the model: decode is **barrier-bound, not
disk-bound**.

- Disk I/O (`stall_demand_load`) is **17%** of decode. Every I/O-targeted lever
  is measured dead against this slice (#5 spec, #9 lookahead, #9 decode-warm,
  #15 prewarm). Do not re-target I/O on this hardware class.
- `scatter_minus_stall` is the **#1 bucket at 36%** (53.5 ms/tok) — MORE than
  attention+router (28) and more than the expert FFN matmul itself (28). #10
  shipped candidate (b) (persistent buffers, no re-zero: 524–789 → 53 ms/tok)
  but kept **full-size `[num_experts,…]` buffers** and a per-layer forcing
  `mx.eval` (engine.py:591). We setitem-scatter ~8 fired rows into a 128-row
  buffer and hard-sync it every layer.
- #10's deferred, never-measured candidate **(a)** — compact fired-only
  buffers, ~16× fewer scatter bytes — targets exactly this bucket.

Upside bound: attacking scatter_minus_stall (36%) plus folding the redundant
barrier into layer_sync (part of 19%). Realistic combined win is a chunk of
that 55%, i.e. low-tens-of-percent decode tok/s — sized like the #9 gate, not a
2×. Gate accordingly; it may not clear the bar, in which case it's recorded dead
like the others.

## Design (the mechanism)

Today (`_scatter_experts`, engine.py:551): persistent buffers `bufs[proj,comp]`
shaped `[num_experts, …]`, allocated once; each layer writes the fired rows at
their GLOBAL expert index `e` (`bufs[(proj,comp)][e] = piece[…]`), evals the
whole buffer set, and the switch/SwitchGLU matmul gathers rows by the router's
`inds` (values in `0..num_experts-1`).

Compact scatter — **keep the persistent full allocation (no realloc churn — that
was the #10 lesson), but only ever touch the first `k` rows:**

1. `fired` is already the sorted list of distinct fired experts (len `k`; k=8 at
   qwen3 decode, larger for multi-token verify unions).
2. Build a global→local remap: expert global-id `e` → its position `j` in
   `fired`. Apply it to `inds` (a small on-device `take`/gather, or a
   `[num_experts]` int lookup scattered once per layer) → `inds_local` with
   values in `0..k-1`.
3. Scatter fired pieces into rows `0..k-1` of the persistent buffers:
   `bufs[(proj,comp)][j] = piece[…]` for `j, e in enumerate(fired)`.
4. `mx.eval` **only the k-row slice** `buf[:k]` (a view — no copy); pass
   `buf[:k]` and `inds_local` to the expert module.

Net: ~16× fewer scatter bytes materialized and a k-sized eval instead of a
128-sized one, with zero new allocation per layer. Buffer set still invalidated
in `_make_slot` on a slot structural-variant change.

**Why it might not be free / the risk this gates on:** MLX's SwitchGLU /
`quantized_matmul` may pick different tiling or reduction order when the expert
dimension is `k` vs `num_experts`, changing fp16 rounding — the "cross-shape
fp16 caution" #10 flagged. Per (token, selected expert) the gathered rows are
numerically identical, so output SHOULD be bit-identical; that is an assertion to
PROVE, not assume.

## Design constraints (fixed — do not relax)

1. **Losslessness is the product.** Compact-scatter output must be bit-identical
   to the full-buffer path, fp16 and 4-bit, on every registered MoE arch it's
   enabled for. Never weaken `test_engine_forward`, `test_engine_moe_selective`,
   `test_scatter_persistent_bufs`, `test_garbage_drafter_invariant` to pass.
2. **Opt-in flag until M3 passes:** `StreamingEngine(…, compact_scatter=False)`.
   Default flips only on a passed gate. CLI `--compact-scatter` mirrors it.
3. **Registry-wide with graceful fallback.** If any arch's expert module is not
   bit-identical under the remap in M0, that arch silently stays on the
   full-buffer path (per-arch verdict cached, like `_lookahead_supported`) —
   never guess, never crash, never ship a lossy arch.
4. **Covers every scatter call site.** `_scatter_experts` is hit from
   single-token decode (`_moe_layer_forward`), multi-token prefill/verify, and
   `tree_forward`'s ephemeral batched cache. The k differs (decode k≈8; verify
   union k up to 51–65% of experts). Compact must be correct and allocation-free
   across all three; the win is largest at decode (smallest k).
5. **No new alloc per layer.** Persistent buffers stay allocated at full size;
   compaction is slice usage, not reallocation (the #10 churn lesson).
6. **Never read num_experts from the (mutating) expert weight.** The engine
   reuses ONE slot object across all uniform layers; writing a compact `[k,…]`
   weight into it means a later layer that reads `weight.shape[0]` gets the
   previous layer's `k`, not the true count (this bit the M0 probe —
   IndexError). Source num_experts from the router (`gate_logits.shape[-1]`),
   which is read before any compaction. Corollary for the persistent form: keep
   the backing buffer `[num_experts,…]` and hand the module a `[:k]` SLICE, so
   the module's structural view is consistent regardless of per-layer k.
7. House rules bind all agents: never `git commit`, never touch `version` in
   pyproject.toml, plain `uv run`, leave work uncommitted for review.

## Milestones (each has a pass/fail gate)

**M0 — offline numerics gate (go/no-go; cheapest first). SCAFFOLDED
2026-07-27: `tests/test_compact_scatter_numerics.py`.**
Implemented as a pytest (reuses the seeded `tiny_qwen3_moe_*` / `tiny_gpt_oss_*`
fixtures + `pack` directly — no config duplication) rather than a scripts/ probe.
It prototypes compact scatter as an instance monkeypatch of `_moe_attn_and_mix`
(fresh `[k,…]` buffers per call + a 0..k-1 `inds` remap; the persistent-slice
perf form is M1) and asserts its logits are **bit-identical (`==`, not
allclose)** to the STOCK streamed engine across a multi-token prefill + 8 greedy
decode steps — a stricter oracle than full-load mlx-lm, since compact-vs-stock is
the same streamed math with only the gather reindexed. Covers qwen3_moe + gpt_oss
× fp16 + 4-bit. **GATE: all four bit-identical.** Any arch that flips is recorded
and excluded (constraint 3); if BOTH bench archs flip, Plan 8 stops here (dead,
recorded in #16). Run: `uv run pytest tests/test_compact_scatter_numerics.py`.
**RESULT 2026-07-27: PASSED 4/4** — compact scatter is bit-identical to the
stock streamed engine on both bench archs, both quants (the fp16-tiling risk did
not materialize on the tiny fixtures; M3 re-confirms at real scale). GO to M1.
The probe also surfaced the shared-slot num_experts hazard now captured as
design constraint #6.

**M1 — engine implementation behind the flag. IMPLEMENTED 2026-07-27.**
Shipped (uncommitted): `StreamingEngine(compact_scatter=False)` (engine.py) —
`_scatter_experts` packs fired experts into a FRESH `[k,…]` buffer (rows 0..k-1)
and returns a global→local `lut`; `_moe_attn_and_mix` remaps the expert-gather
`inds` via `remap[inds]` (full path returns None → `inds` unchanged, byte-for-byte
untouched). num_experts from `self._num_experts` (constraint #6). One change
covers all call sites — decode, prefill, and `tree_forward` all route through
`_moe_attn_and_mix`. Counter `compact_rows_scattered` (also in `prefetch_stats`).
CLI `--compact-scatter` on generate + serve (→ `run_server`/`build_server`).
DEVIATION from constraint #5: fresh `[k,…]` per layer, NOT a persistent
`[num_experts,…]` slice. **CORRECTED post-M2 (2026-07-27):** the first M1 build
used fresh-zeros + per-row setitem, which M2 measured a +26% REGRESSION vs the
persistent-full default (it re-committed the #10 realloc/zero cost). The compact
branch now builds the `[k,…]` buffer via a single `mx.stack` per subkey (no
zero-fill, no per-row nodes) — the only compact form M2 found faster than the
default (−11% at decode). Still bit-identical (same gathered rows); still
default-OFF (M2 sized its greedy-decode win under the M3 bar — see M2 RESULT).
Re-run `test_compact_scatter*` to reconfirm bit-identity after the mechanism
swap. Tests: tests/test_compact_scatter.py —
flag-on vs full-buffer bit-identity across prefill+decode on qwen3_moe + gpt_oss
× fp16/4-bit, counter-moved assertion, and default-off (counter stays 0).
**GATE PASSED 2026-07-27: full suite green (same 4 known #8 failures),
test_compact_scatter + test_compact_scatter_numerics pass — reconfirmed after
the M2 mechanism swap (zeros+setitem → mx.stack), so the compact path is
bit-identical to the full-buffer path on both bench archs × both quants.** Run:
`uv run pytest tests/test_compact_scatter.py tests/test_compact_scatter_numerics.py`
then `uv run pytest`.

**M2 — scatter micro-bench. SCAFFOLDED 2026-07-27 (pending Mac run).**
Extended `scripts/scatter_strategy_microbench.py`: now parameterized over a k-set
— `decode` (top-8, capped for tiny fixtures) and `verify_union` (~0.55·num_experts,
the K-token verify distinct-expert union) — and times four strategies per k with
disk + FFN isolated (all experts preloaded resident, only the scatter+eval is
timed): **B_persistent_default_path** (the shipped default: k row-scatters into a
`[num_experts,…]` persistent buffer + eval the whole buffer) vs
**D_compact_zeros_scatter** (the shipped M1 compact branch, faithful: fresh
`[k,…]` `mx.zeros` + local-row setitem + eval only the k rows), plus A (pre-#10
fresh-full-zeros) and C (compact via `mx.stack`, alt build) for reference. The
sizing head-to-head is **B vs D at k=decode**. num_experts read from the fresh
(unmutated) slot's `gate_proj.weight.shape[0]` (plan8 #6-safe here since the
bench slot is never compacted). Compiles (py_compile OK); needs a real pack on
the Mac to produce numbers. Run: `uv run python
scripts/scatter_strategy_microbench.py ./packed/<qwen3-30b> [n_layers]`.
**GATE (informational, sizes M3): D_compact_zeros_scatter ms/token materially
below B_persistent_default_path's ~53 ms/tok at decode k; if it isn't, M3 A/B
won't clear and Plan 8 stops.**

**RESULT 2026-07-27 — GATE NOT CLEARED as scoped (qwen3-30b, 128 experts,
scatter_strategy_microbench.json). D (the shipped M1 mechanism) is a REGRESSION,
not a win** (decode k=8, ms/tok over 48 layers): B_default=45.2, **D_compact=56.8
(+26%)**, C_compact_stack=40.3 (−11%), A_fresh_full=267.4. The fresh-alloc +
zero-fill + k setitem nodes in D cost more than the [8,…]-vs-[128,…] eval saving
— my M1 deviation from constraint #5 (fresh [k,…] not a persistent slice)
re-committed the #10 realloc/zero sin. Only **compact-via-stack (C)** beats the
default, because it fuses the build into one op with no zero-fill. But C's decode
win is ~5 ms of the 45 ms scatter bucket ≈ **3–4% end-to-end at greedy k=8, under
the M3 ≥10% bar** — so M3 on greedy decode will not clear. The large compact win
(C −30%) is at **verify-union k=70** (B=325.2 → C=226.6), which only helps
multi-token verify (speculative/tree) passes, not #16's single-token decode. →
Plan 8 stops for greedy decode; the mechanism finding (stack, not zeros+setitem)
and the verify-k potential are recorded. The shipped M1 `compact_scatter` flag
(mechanism D) is a latent regression — switch it to stack or remove it before it
is ever trusted on.

**M3 — real-model flushed A/B gate.**
Qwen3-30B @ 10 GB wired, 200 greedy tokens, ABBA-interleaved control(full)/
experiment(compact), page cache flushed between runs, per-run
`compressed_gb_during_run` as a #15 validity check (drop/re-run any run > 30 GB),
token streams byte-identical across ALL runs. Re-run the #16 attribution probe
under compact to confirm the bucket moved. **GATE: ≥10% median end-to-end decode
tok/s improvement AND no pair worse than −5%** (the house gate shape from #9/#15).
Pass → default-on + README/CLAUDE.md update + retire from #16. Fail → record dead
in #16 with the numbers, flag stays opt-in/off.

**M4 — (optional stack) drop the per-layer eval barrier.**
Only if M3 passes and prof(re-attribution) still shows a forcing-eval barrier.
Pin the k fired pieces against cache eviction until the layer's compute drains,
remove `mx.eval` at engine.py:591, let `layer_sync` (659) cover it — 2 host
drains → 1 per layer. Lossless (memory policy only). Same A/B gate. This is a
cache-eviction-lifetime change (design-sensitive — Opus), spec it separately if
reached.

## Notes

- Re-check the bucket split (the #16 probe reports it) on any new machine/model
  before assuming this lever applies: on a bigger model or tighter budget the
  disk share rises and scatter's share shrinks.
- The barrier-count framing (144 host↔device drains/token) suggests a further,
  out-of-scope direction if M3/M4 disappoint: reducing per-layer sync points
  structurally (e.g. deferring the router tolist). Not in this plan — the
  router sync is fundamental to selective streaming (you must read routing on
  host to know what to fetch); parked here only so the idea has a home.
