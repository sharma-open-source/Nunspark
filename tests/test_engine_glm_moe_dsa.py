"""Plan 6 gates: glm_moe_dsa (GLM-5.2-style MLA + DSA indexer + MoE).

M1 gate: packed pieces reassemble the source weight tree byte-identically.
M2 gate: streamed forward is bit-identical to full-load mlx-lm, fp16 and
4-bit, at sequence lengths below AND above index_topk (dense and sparse
attention regimes), and across a cached greedy decode that crosses the
index_topk boundary.
"""
import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.glm_moe_dsa import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.kv_store import KVStore

SHORT = [3, 7, 42, 1, 9]                                   # L=5 <= index_topk=8
LONG = [3, 7, 42, 1, 9, 11, 250, 8, 33, 2, 100, 63, 5, 9]  # L=14 > index_topk=8


def _reference_model(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    q = config.get("quantization")
    if q:
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def _reference_logits(model_dir, tokens):
    model = _reference_model(model_dir)
    out = model(mx.array(tokens)[None])
    mx.eval(out)
    return out


# --- M1 gate: byte-identical reassembly ------------------------------------

def test_packed_pieces_reassemble_byte_identical(tiny_glm_moe_dsa_model_dir, tmp_path):
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.model_type == "glm_moe_dsa"

    src = mx.load(str(tiny_glm_moe_dsa_model_dir / "model.safetensors"))
    n_experts = manifest.config["n_routed_experts"]
    n_layers = manifest.config["num_hidden_layers"]

    rebuilt = {}
    for piece in manifest.pieces:
        tensors = mx.load(str(out / piece.file))
        pid = piece.piece_id
        if pid == "embed" or pid == "norm_head":
            for k, v in tensors.items():
                rebuilt[k if k.startswith("lm_head") else f"model.{k}"] = v
        elif "expert" in pid:
            continue   # stacked back below, layer by layer
        else:  # layer_NNN / layer_NNN_core
            layer = int(pid.split("_")[1])
            for k, v in tensors.items():
                rebuilt[f"model.layers.{layer}.{k}"] = v

    # re-stack per-expert pieces into the switch_mlp tensors
    for layer in range(n_layers):
        epid0 = Manifest.layer_expert_piece_id(layer, 0)
        if not manifest.has_piece(epid0):
            continue
        parts = [mx.load(str(out / f"{Manifest.layer_expert_piece_id(layer, e)}.safetensors"))
                 for e in range(n_experts)]
        for sub in parts[0]:
            rebuilt[f"model.layers.{layer}.{sub}"] = mx.stack([p[sub] for p in parts])

    assert set(rebuilt) == set(src)
    for k in src:
        assert src[k].dtype == rebuilt[k].dtype, k
        assert mx.array_equal(src[k], rebuilt[k]).item(), k


# --- M2 gate: streamed forward bit-identity --------------------------------

def test_streamed_forward_matches_full_load_dense_regime(tiny_glm_moe_dsa_model_dir, tmp_path):
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    ref = _reference_logits(tiny_glm_moe_dsa_model_dir, SHORT)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(SHORT)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_streamed_forward_matches_full_load_sparse_regime(tiny_glm_moe_dsa_model_dir, tmp_path):
    # L > index_topk: the DSA indexer fires and attention runs on its top-k
    # boolean mask — the streamed mask/index path must match exactly.
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    ref = _reference_logits(tiny_glm_moe_dsa_model_dir, LONG)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(LONG)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_streamed_quant_forward_matches_full_load(tiny_glm_moe_dsa_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-glm-q4.nunspark"
    pack(tiny_glm_moe_dsa_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    for tokens in (SHORT, LONG):
        ref = _reference_logits(tiny_glm_moe_dsa_quant_model_dir, tokens)
        engine = StreamingEngine(out, manifest, budget_bytes=10**9)
        try:
            got = engine.forward(mx.array(tokens)[None])
            mx.eval(got)
            assert float(mx.max(mx.abs(got - ref))) == 0.0
        finally:
            engine.close()


def test_streamed_quant_matches_with_tiny_budget(tiny_glm_moe_dsa_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-glm-q4.nunspark"
    pack(tiny_glm_moe_dsa_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    ref = _reference_logits(tiny_glm_moe_dsa_quant_model_dir, SHORT)

    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        got = engine.forward(mx.array(SHORT)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_cached_greedy_decode_matches_full_load(tiny_glm_moe_dsa_model_dir, tmp_path):
    # Prefill 5 tokens then greedy-decode 8 more with a persistent KVStore of
    # Paired caches — the sequence crosses index_topk=8, so decode transitions
    # from the dense regime into the sparse indexer regime mid-stream. Every
    # step's logits must be bit-identical to the stock model decoding on its
    # own make_cache() CacheLists.
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    ref_model = _reference_model(tiny_glm_moe_dsa_model_dir)
    ref_cache = ref_model.make_cache()

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    kv = KVStore(tmp_path / "kv", budget_bytes=10**9,
                 cache_kinds=engine.cache_kinds)
    try:
        ref = ref_model(mx.array(SHORT)[None], cache=ref_cache)
        got = engine.forward(mx.array(SHORT)[None], kv=kv)
        mx.eval(ref, got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0

        tok = mx.argmax(ref[:, -1, :], axis=-1)
        for _ in range(8):
            ref = ref_model(tok[:, None], cache=ref_cache)
            got = engine.forward(tok[:, None], kv=kv)
            mx.eval(ref, got)
            assert float(mx.max(mx.abs(got - ref))) == 0.0
            tok = mx.argmax(ref[:, -1, :], axis=-1)
    finally:
        kv.close()
        engine.close()


# --- M3 gate: KVStore spill round-trip + clean refusals ---------------------

def test_kv_spill_roundtrip_bit_identical(tiny_glm_moe_dsa_model_dir, tmp_path):
    # budget_bytes=1 forces every layer's CacheList to spill to disk and reload
    # on each subsequent access — the save/load round-trip (both children,
    # including the indexer cache's zero-width values array) must not perturb
    # a single bit of the decode.
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    ref_model = _reference_model(tiny_glm_moe_dsa_model_dir)
    ref_cache = ref_model.make_cache()

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    kv = KVStore(tmp_path / "kv-spill", budget_bytes=1,
                 cache_kinds=engine.cache_kinds)
    try:
        ref = ref_model(mx.array(SHORT)[None], cache=ref_cache)
        got = engine.forward(mx.array(SHORT)[None], kv=kv)
        mx.eval(ref, got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0

        tok = mx.argmax(ref[:, -1, :], axis=-1)
        for _ in range(8):
            ref = ref_model(tok[:, None], cache=ref_cache)
            got = engine.forward(tok[:, None], kv=kv)
            mx.eval(ref, got)
            assert float(mx.max(mx.abs(got - ref))) == 0.0
            tok = mx.argmax(ref[:, -1, :], axis=-1)
        assert kv.peak_bytes > 0          # nbytes accounting saw the CacheLists
    finally:
        kv.close()
        engine.close()


def test_batched_generate_cleanly_refuses(tiny_glm_moe_dsa_model_dir, tmp_path):
    import pytest
    from nunspark.generate import batched_generate

    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        with pytest.raises(ValueError, match="paired per-layer caches"):
            batched_generate(engine, [SHORT], max_tokens=4)
    finally:
        engine.close()


def test_tree_forward_cleanly_refuses(tiny_glm_moe_dsa_model_dir, tmp_path):
    import pytest

    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    kv = KVStore(tmp_path / "kv-tree", budget_bytes=10**9,
                 cache_kinds=engine.cache_kinds)
    try:
        engine.forward(mx.array(SHORT)[None], kv=kv)
        with pytest.raises(ValueError, match="paired per-layer caches"):
            engine.tree_forward(mx.array([SHORT[:2]]), kv, prefix_len=len(SHORT))
    finally:
        kv.close()
        engine.close()


# --- M4 gate: speculative decoding invariants --------------------------------

def test_adversarial_ngram_drafter_matches_greedy_glm(tiny_glm_moe_dsa_model_dir, tmp_path):
    # The garbage-drafter invariant on the Paired-cache arch: a drafter that is
    # NEVER accepted must reproduce plain greedy token-for-token. max_tokens=24
    # from an 8-token prompt crosses index_topk=8, so verify passes run in the
    # sparse-indexer regime with _RecordingCacheList clones.
    from nunspark.generate import generate, ngram_speculative_generate, SpecStats
    import test_garbage_drafter_invariant as gdi

    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        drafter = gdi._WrongEveryTimeNGramDrafter(ref, len(prompt), 320, num_draft_tokens=5)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=24, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 24
    assert stats.draft_tokens_proposed > 0
    assert stats.accepted_total == 0


def test_mixed_acceptance_ngram_bit_identical_glm(tiny_glm_moe_dsa_model_dir, tmp_path):
    # Partial-correct drafter: commit_verified(kv, recs, m + 1) at controlled m
    # walks the _RecordingCacheList commit path (both sub-caches receive the
    # accepted rows) every round, at partial and full acceptance.
    from nunspark.generate import generate, ngram_speculative_generate, SpecStats
    import test_garbage_drafter_invariant as gdi

    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    K = 5
    out = tmp_path / "tiny-glm.nunspark"
    pack(tiny_glm_moe_dsa_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        for j in (1, K - 1, K):
            drafter = gdi._PartialCorrectNGramDrafter(
                ref, len(prompt), 320, j=j, num_draft_tokens=K)
            stats = SpecStats()
            got = list(ngram_speculative_generate(
                engine, drafter, prompt, max_tokens=24, stats=stats))
            assert got == ref, f"mismatch at j={j}"
            assert len(got) == 24
    finally:
        engine.close()
