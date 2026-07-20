# Plan 6 — GLM-5.2 (`glm_moe_dsa`) streaming support

Status: M0–M4 COMPLETE (all gates PASSED, 2026-07-20); M5 real-model numbers pending community/bench
Branch: version2.0 (uncommitted, user owns commits)

## Gate outcomes (2026-07-20)

- **M0 PASS** — `scripts/probe_glm_moe_dsa.py`: tiny glm_moe_dsa runs in pinned
  mlx-lm 0.31.3; sparse indexer fires exactly past index_topk; cached decode ==
  full re-forward 12/12 across the dense→sparse boundary; cache is
  `CacheList(KVCache, KVCache)`. HF file-index check: the real checkpoint has
  per-layer indexer weights (IndexShare shares indices at runtime, not
  weights), so mlx-lm's per-layer indexer loads it; the BF16 repo has no fp8
  `*_scale_inv` keys.
- **M1 PASS** — packer needed ZERO code changes (selective_moe + expert_attr
  drive the generic split; MTP layers are dropped by mlx-lm's sanitize
  upstream). Gate test `test_packed_pieces_reassemble_byte_identical`:
  fp16 pack reassembles the source tree byte-identically.
- **M2 PASS, stronger than gated** — streamed forward is max|Δ| = 0 vs
  full-load mlx-lm in fp16 AND 4-bit, dense and sparse regimes, and across a
  KVStore-backed greedy decode crossing index_topk. Shipped as archspec
  `Paired` cache kind + `ArrayCausal` mask plan + ArchSpec
  `shared_experts_attr`/`moe_mix_cast` hooks (registry-driven, no engine
  branches); `deepseek_v32` registered too (same spec, different args class).
- **M3 PASS** — selective expert streaming came free via the shared
  `_moe_attn_and_mix` path (identity `moe_route`: MoEGate already returns
  (inds, scores)). CacheList audit: KVStore nbytes/evict/reload round-trips
  CacheLists (zero-width indexer values stored as a shape marker);
  `cache_offset()` helper fixes prefix_cache.commit / _maybe_reset_prefetch /
  _padded_mask_index; `batched_generate` and `tree_forward` refuse Paired
  archs with a clear ValueError (sequential paths only, v1).
- **M4 PASS** — `_clone_kv_cache` clones CacheLists; `_RecordingCacheList`
  gives verify_forward per-sub-cache recorders (indexable, as the deepseek
  attention requires) and `commit_verified` walks them. Garbage-drafter
  invariant and mixed-acceptance (j = 1, K−1, K) tests pass with generation
  crossing index_topk.
- All gates live in `tests/test_engine_glm_moe_dsa.py` (11 tests); full suite
  368 passed, only the 4 known CI-deselected failures remain (backlog #8).

## What GLM-5.2 is

`zai-org/GLM-5.2` — 744B-total / ~39B-active MoE, `model_type: "glm_moe_dsa"`,
`architectures: ["GlmMoeDsaForCausalLM"]`. Architecturally it is a
DeepSeek-V3.2 clone: mlx-lm implements it as a thin `ModelArgs` wrapper over
`mlx_lm.models.deepseek_v32` (`glm_moe_dsa.Model(DSV32Model)`), merged via
mlx-lm PR #1419 and present in our pinned mlx-lm 0.31.3.

Key config (from the HF repo):

| field | value | consequence for us |
|---|---|---|
| num_hidden_layers | 78 (+1 MTP layer in checkpoint) | packer must drop/segregate layer 78 |
| hidden_size | 6144 | |
| MLA | q_lora_rank 2048, kv_lora_rank 512, qk_nope 192, qk_rope 64, v_head 256 | latent-KV cache, NOT plain per-head K/V |
| DSA indexer | index_n_heads 32, index_head_dim 128, index_topk 2048 | second KV cache per layer; exact full attention until seq > 2048 |
| MoE | 256 routed + 1 shared expert, top-8, moe_intermediate 2048, first_k_dense_replace 3 | selective expert streaming, shared expert always resident |
| routing | sigmoid scoring, noaux_tc, e_score_correction_bias, routed_scaling_factor 2.5 | router is NOT an nn.Linear + softmax; correction bias must stay fp32 |
| MTP | num_nextn_predict_layers 1 | future eagle-style drafter opportunity (backlog, not this plan) |

## Why this is engine work, not a registry line

The registry comment in `architectures.py` deliberately excludes
"deepseek/MLA" for two reasons; both must be solved:

1. **Cache contract.** `deepseek_v32.make_cache()` returns
   `CacheList(KVCache(), KVCache())` per layer — the MLA latent cache plus the
   indexer's key cache. Our `cache_plan` seam (archspec `KV` / `Rotating`
   kinds) has no paired-cache kind, and `kv_store.py` spill / `prefix_cache`
   / batched decode have never seen a `CacheList`.
2. **MoE block shape.** `_moe_attn_and_mix` (engine.py:607) assumes
   `router(x) -> logits`, `moe_route(args, logits) -> (inds, scores)`,
   `switch_mlp(x, inds)`, then `(y * scores).sum(-2)`. DeepSeek's `MoEGate`
   returns `(inds, scores)` directly (sigmoid + bias + group-select + scaling,
   fp32), the mix needs a trailing `.astype(y.dtype)`, and there is a
   **shared expert** added after the routed mix — none of which the current
   mix path expresses.

What we do NOT have to reimplement: the entire MLA/indexer attention interior.
`DeepseekV32DecoderLayer` has the standard
`__call__(x, mask, cache)` shape (input_layernorm → self_attn → residual →
post_attention_layernorm → mlp → residual), so the streamed slot reuses the
stock block and the L==1 absorbed path / L>1 sparse-mask path come for free.
Final norm is plain `nn.RMSNorm`, no embed scaling, no softcap, untied lm_head
— all contract-clean.

Registering `glm_moe_dsa` makes `deepseek_v32` (and V3.2-family fp8-derived
conversions) nearly free — same ArchSpec, different args class. Ship both.

## The validation constraint (the big one)

**We cannot download or load the real model** (744B; smallest community MLX
quant ~2.4 bpw is still >200 GB). Therefore, per standing practice, all
correctness gates run on **tiny seeded models built in `tests/conftest.py`**
— exactly how qwen3_moe / gpt_oss / gemma4 shipped. Losslessness is defined
as bit-identity **to full-load mlx-lm's `glm_moe_dsa` forward on the same
tensors**, which is checkable at any size. What tiny models cannot tell us
(real-model tok/s, expert-reuse rates, TTFT) is explicitly out of scope for
the gates; real-model numbers come later from community bench reports
(`nunspark bench`), never invented.

Tiny-config choices that make the tiny model exercise every real code path:
- `index_topk` small (e.g. 8) so prompts > 8 tokens exercise the **sparse
  indexer path** (real model: only beyond 2048 ctx) and prompts ≤ 8 the
  dense path; both must be bit-identical.
- `first_k_dense_replace=1`, `n_routed_experts=8`, `top_k=2`,
  `n_shared_experts=1` — dense/moe layer split, selective streaming, shared
  expert all exercised.
- `n_group=1` (matches GLM-5.2; the grouped-select branch is dead code for
  this model — assert, don't implement speculatively).
- fp16 AND 4-bit variants (4-bit exercises quantized `MultiLinear`
  embed_q/unembed_out and fp32 `e_score_correction_bias` survival).

## Milestones

### M0 — Reference fidelity probe (bounded, `scripts/`)
Build the tiny `glm_moe_dsa` model; run stock mlx-lm full-load forward.
Verify: (a) it runs at all in our pinned mlx-lm; (b) L==1 vs L>1 paths agree
where they must (greedy continuation consistency); (c) confirm from the
community MLX conversions' config/weight index that mlx-lm's sanitize
round-trips the real checkpoint layout (expert stacking, kv_b_proj →
embed_q/unembed_out split, MTP-layer drop) — read the HF file index, no
download.
**Gate:** stock tiny forward runs; sparse path triggers at seq > index_topk;
layout questions answered in a short write-up. Kill criteria: mlx-lm impl
broken → file upstream, stop.

### M1 — Packer
Pack the tiny model: per-layer core piece (attention incl. indexer +
`MultiLinear` embed_q/unembed_out, norms, `MoEGate` weight +
`e_score_correction_bias`, shared expert), per-expert pieces from the stacked
`switch_mlp` (qwen3_moe precedent), dense-layer pieces for layers <
first_k_dense_replace, drop MTP layer(s) ≥ num_hidden_layers, respect
`cast_predicate` (`e_score_correction_bias` stays fp32 through 4-bit packing;
`MoEGate.weight` has no `to_quantized` → naturally unquantized).
**Gate:** pack tiny fp16 + 4-bit; reassembling all pieces reproduces the
source weight tree byte-identically.

### M2 — Engine: streamed forward, all-experts correctness first
1. New archspec cache kind `Paired` → `CacheList(KVCache(), KVCache())`;
   `cache_plan` returns it per layer.
2. New mask plan mirroring the model's
   `create_attention_mask(h, cache[0][0], return_array=True)` (boolean array
   mask — the indexer's `mx.where` and sparse-mask `&` require an array, so
   `UniformCausal` is not reusable as-is).
3. ArchSpec entry: `layer_key_fn` = dense/moe by `first_k_dense_replace` +
   `moe_layer_freq`; `router_attr="gate"` with `moe_route` = unpack (MoEGate
   already returns `(inds, scores)`); `expert_attr="switch_mlp"`;
   `supports_quantized_kv=False` (MLA latent + sparse mask paths unaudited
   under quantized SDPA).
4. `_moe_attn_and_mix` extensions: optional `.astype(y.dtype)` after the
   scored sum, and an optional shared-expert add (`shared_experts_attr` on
   ArchSpec) — both registry-driven, no per-arch branches in the engine.
**Gate:** streamed forward **bit-identical (max|Δ| = 0)** to full-load
mlx-lm on tiny fp16 and 4-bit, at seq lengths below AND above index_topk,
single-token decode and multi-token passes. This is the losslessness gate —
never weakened.

### M3 — Selective expert streaming + KV/cache integration audit
Selective path (only fired experts read from disk) with expert-aware
prefetch/warm reuse; shared expert lives in the always-read core piece.
Audit and gate the surrounding machinery against `CacheList`:
`kv_store` spill, `prefix_cache`, `verify_forward`'s ephemeral cache clone,
`batched_generate`'s cache handling. Anything not made correct in this
milestone must **raise a clear unsupported error** for this arch — silent
wrong output is the only forbidden outcome.
**Gate:** bit-identity preserved; piece-read trace shows only fired experts
loaded; each generation mode either passes its existing invariant test on the
tiny model or cleanly refuses.

### M4 — Speculative + generation modes
`speculative_generate` / `ngram_speculative_generate` on the tiny model:
verify-pass shapes hit the L>1 sparse-mask path, so run the garbage-drafter
invariant test (exact greedy-token match) at seqs straddling index_topk.
Known fp16 near-tie caveat applies as elsewhere — characterize, don't
overclaim.
**Gate:** `test_garbage_drafter_invariant`-class test passes for
glm_moe_dsa tiny; CLI `generate`/`serve`/web pool run against the tiny pack.

### M5 — Real-model enablement (user-owned, no gate we can run)
Docs: which community MLX conversion to pack from (packing streams
shard-by-shard — disk-bound, not RAM-bound; needs ~2× quant size free disk),
expected regime (active ~39B/token ⇒ even at 4-bit ~20 GB of touched expert
weight per cold token — this targets high-RAM Macs where most experts stay
cached; on 16 GB it will run but crawl). Add to README's supported table only
with measured community numbers, per standing rule.

## Risks / open questions
- **IndexShare divergence:** GLM-5.2's `index_topk_freq=4` /
  `index_skip_topk_offset` fields are absent from mlx-lm's ModelArgs — the
  mlx-lm impl runs a per-layer indexer. Our parity target is mlx-lm, so our
  gates are unaffected, but real-checkpoint behavior beyond 2048 ctx depends
  on how the checkpoint stores indexer weights (M0 item c resolves this).
- **`CacheList` blast radius** is the largest unknown (spill, prefix cache,
  batched, tree spec). M3's "correct or cleanly refuse" rule bounds it.
- **Perf on 16 GB is expected to be poor**; that's a physics statement, not
  a gate failure. The deliverable is correctness + the high-RAM story.
- Router host-sync cost (`inds.tolist()` over top-8×256) — same pattern as
  qwen3_moe/gpt_oss, no new risk expected; check in M3 trace.
