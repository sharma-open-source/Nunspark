import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_moe import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine


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
