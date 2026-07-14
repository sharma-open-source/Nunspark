import json
import tempfile

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_moe import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.kv_store import KVStore


def _reference_logits(model_dir, tokens):
    """Full-load the Qwen3-MoE model the way mlx-lm would, and run it."""
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    q = config.get("quantization")
    if q:
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    out = model(mx.array(tokens)[None])
    mx.eval(out)
    return out


def test_selective_moe_forward_matches_full_load(tiny_qwen3_moe_model_dir, tmp_path):
    out = tmp_path / "moe.nunspark"
    pack(tiny_qwen3_moe_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.has_piece(Manifest.layer_core_piece_id(0))   # selective pack

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_moe_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4).item()
    finally:
        engine.close()


_MOE_CFG = {  # mirrors conftest.TINY_QWEN3_MOE_CONFIG
    "model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 4,
    "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "num_experts": 8, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
    "mlp_only_layers": [], "moe_intermediate_size": 64, "norm_topk_prob": True,
    "rms_norm_eps": 1e-5, "vocab_size": 320, "max_position_embeddings": 2048,
    "rope_theta": 10000.0, "tie_word_embeddings": True,
}


def test_selective_moe_quant_forward_matches_full_load(tiny_qwen3_moe_quant_model_dir, tmp_path):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_moe_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        # quantized matmuls are deterministic -> streamed output is bit-identical
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_selective_moe_quant_tiny_budget(tiny_qwen3_moe_quant_model_dir, tmp_path):
    # A budget too small to keep any piece resident must still be correct: each
    # fired expert's rows are scattered in and the piece may be evicted immediately.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_moe_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_selective_moe_mixed_dense_and_moe(tmp_path):
    # mlp_only_layers=[0,2] -> layers 0,2 dense (whole-layer path), 1,3 MoE (selective).
    config = {**_MOE_CFG, "mlp_only_layers": [0, 2]}
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    src = tmp_path / "mixed"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(src / "model.safetensors"), dict(tree_flatten(model.parameters())))

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(src, tokens)
    out = tmp_path / "mixed.nunspark"
    pack(src, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.has_piece(Manifest.layer_piece_id(0))        # dense whole-layer
    assert manifest.has_piece(Manifest.layer_core_piece_id(1))   # moe selective

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4).item()
    finally:
        engine.close()


def test_legacy_wholelayer_moe_pack_still_loads(tiny_qwen3_moe_quant_model_dir, tmp_path, monkeypatch):
    # Force OLD whole-layer packing (no expert split) to prove the engine's
    # whole-layer fallback still loads a pre-selective MoE pack bit-identically.
    from nunspark import packer
    monkeypatch.setattr(packer, "supported_model_types", lambda: [])

    out = tmp_path / "legacy-moe.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.has_piece(Manifest.layer_piece_id(0))          # whole-layer
    assert not manifest.has_piece(Manifest.layer_core_piece_id(0)) # not split

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_moe_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_selective_moe_reads_only_fired_experts(tiny_qwen3_moe_quant_model_dir, tmp_path):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    num_experts = manifest.config["num_experts"]        # 8
    top_k = manifest.config["num_experts_per_tok"]      # 2

    # prefetch=False so loads are exactly what the forward pulls (no speculative cores)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9, prefetch=False)
    loaded: list[str] = []
    orig = engine.cache._loader
    def spy(pid):
        loaded.append(pid)
        return orig(pid)
    engine.cache._loader = spy
    try:
        engine.forward(mx.array([[7]]))   # single token -> <= top_k experts fire/layer
    finally:
        engine.close()

    for layer in range(manifest.num_layers):
        prefix = f"layer_{layer:03d}_expert_"
        experts = {p for p in loaded if p.startswith(prefix)}
        assert Manifest.layer_core_piece_id(layer) in loaded     # core read
        assert 0 < len(experts) <= top_k                         # only fired ones
        assert len(experts) < num_experts                        # never all experts


def test_two_region_cache_output_bit_identical(tiny_qwen3_moe_quant_model_dir, tmp_path):
    # plan4 M2: the two-region policy (LRU expert region + pinned cores) is a
    # caching change only — output must stay bit-identical to the reference
    # full-load AND to the old single-region-equivalent config (frac=0.0,
    # which gives experts no region, i.e. every expert read hits disk).
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_moe_quant_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)  # default frac
    try:
        got_new = engine.forward(mx.array(tokens)[None])
        mx.eval(got_new)
        # cores are pinned at construction (they fit the main region's cap)
        core_pids = {Manifest.layer_core_piece_id(l) for l in range(manifest.num_layers)}
        assert engine.cache._pinned == core_pids
        stats = engine.cache.stats()
        assert stats["resident_bytes"]["expert"] > 0     # expert region in use
    finally:
        engine.close()

    engine_old = StreamingEngine(out, manifest, budget_bytes=10**9,
                                 expert_cache_frac=0.0)
    try:
        got_old = engine_old.forward(mx.array(tokens)[None])
        mx.eval(got_old)
    finally:
        engine_old.close()

    assert float(mx.max(mx.abs(got_new - ref))) == 0.0
    assert float(mx.max(mx.abs(got_new - got_old))) == 0.0


def _kv():
    d = tempfile.mkdtemp(prefix="nunspark_kv_test_")
    return KVStore(d, budget_bytes=10**12)


def test_tree_forward_selective_matches_load_all_moe(tiny_qwen3_moe_quant_model_dir, tmp_path):
    # M4: tree_forward now scatters only the fired-expert UNION per selectively-
    # packed MoE layer instead of every expert. Since _scatter_experts zero-fills
    # unfired rows the same way regardless of which experts were requested, and
    # the expert-mix gather (x, inds) only ever reads fired rows, logits AND the
    # KV stash must be bit-identical to the pre-M4 load-all-experts behavior.
    # We prove this by comparing tree_forward's output against per-path forward()
    # calls from the same prefix — forward() already runs the selective
    # (load-only-fired) path per token, so this is the same identity oracle
    # test_engine_tree_forward.py uses for the dense/llama case.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        prompt = [3, 7, 42, 1, 9]
        token_paths = [[2, 3, 3], [2, 3, 7], [5, 5, 1], [5, 5, 8]]

        kv = _kv()
        try:
            engine.forward(mx.array(prompt)[None], kv=kv)
            logits, stash = engine.tree_forward(
                mx.array(token_paths), kv, prefix_len=len(prompt))
            mx.eval(logits)
        finally:
            kv.close()

        for i, path in enumerate(token_paths):
            kvi = _kv()
            try:
                engine.forward(mx.array(prompt)[None], kv=kvi)
                ref = engine.forward(mx.array(path)[None], kv=kvi)
                mx.eval(ref)
            finally:
                kvi.close()
            # quantized weights -> exact match, no tolerance needed.
            assert float(mx.max(mx.abs(logits[i] - ref[0]))) == 0.0

            # stash for this path must also match: replay the prefix+path and
            # commit, then confirm the next-token logits agree exactly.
            kvc = _kv()
            try:
                engine.forward(mx.array(prompt)[None], kv=kvc)
                _, stash2 = engine.tree_forward(
                    mx.array(token_paths), kvc, prefix_len=len(prompt))
                engine.commit_path(kvc, stash2, path_index=i, accepted_len=len(path))
                probe = engine.forward(mx.array([11])[None], kv=kvc)[:, -1, :]
                mx.eval(probe)
            finally:
                kvc.close()

            kvr = _kv()
            try:
                engine.forward(mx.array(prompt)[None], kv=kvr)
                engine.forward(mx.array(path)[None], kv=kvr)
                probe_ref = engine.forward(mx.array([11])[None], kv=kvr)[:, -1, :]
                mx.eval(probe_ref)
            finally:
                kvr.close()
            assert float(mx.max(mx.abs(probe - probe_ref))) == 0.0
    finally:
        engine.close()


def test_tree_forward_loads_fewer_experts_than_all(tiny_qwen3_moe_quant_model_dir, tmp_path):
    # M4: tree_forward must scatter only the union of fired experts per layer,
    # not every expert — spy on the cache loader and confirm expert-piece reads
    # stay under num_experts for a small verify batch whose per-position top_k
    # union cannot plausibly cover all 8 experts.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    num_experts = manifest.config["num_experts"]         # 8

    engine = StreamingEngine(out, manifest, budget_bytes=10**9, prefetch=False)
    prompt = [3, 7, 42, 1, 9]
    token_paths = [[2], [7]]   # B=2, d=1 -> 2 positions/layer, union <= 4 < 8

    loaded: list[str] = []
    orig = engine.cache._loader
    def spy(pid):
        loaded.append(pid)
        return orig(pid)
    engine.cache._loader = spy

    kv = _kv()
    try:
        engine.forward(mx.array(prompt)[None], kv=kv)     # cores get pinned/loaded here
        engine.tree_forward(mx.array(token_paths), kv, prefix_len=len(prompt))
    finally:
        kv.close()
        engine.close()

    for layer in range(manifest.num_layers):
        prefix = f"layer_{layer:03d}_expert_"
        experts = {p for p in loaded if p.startswith(prefix)}
        assert Manifest.layer_core_piece_id(layer) in loaded
        assert 0 < len(experts)
        assert len(experts) < num_experts   # never all experts for this tiny batch


def test_core_pinning_skipped_when_budget_too_small(tiny_qwen3_moe_quant_model_dir, tmp_path):
    # budget=1: cores cannot fit the main region's cap, so nothing is pinned
    # (pinning must never blow the byte budget) and output stays exact.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_moe_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        assert engine.cache._pinned == set()
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()
