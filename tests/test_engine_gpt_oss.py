import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.gpt_oss import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine

_NUM_LOCAL_EXPERTS = 8  # mirrors conftest.TINY_GPT_OSS_CONFIG["num_local_experts"]


def _reference_logits(model_dir, tokens):
    """Full-load the GPT-OSS model the way mlx-lm would, and run it. Honors a
    heterogeneous quantization dict (per-path override dicts over a scalar base)
    via the same class_predicate contract mlx_lm.load_model uses, so it is a valid
    reference for both uniform and mixed-quant checkpoints."""
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    q = config.get("quantization")
    if q:
        def class_predicate(path, module):
            if path in q:
                return q[path]
            if not hasattr(module, "to_quantized"):
                return False
            return True
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    mode=q.get("mode", "affine"), class_predicate=class_predicate)
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    out = model(mx.array(tokens)[None])
    mx.eval(out)
    return out


def test_gpt_oss_pack_splits_every_layer_selectively(tiny_gpt_oss_model_dir, tmp_path):
    out = tmp_path / "gpt-oss.nunspark"
    manifest = pack(tiny_gpt_oss_model_dir, out)
    # every gpt_oss layer carries an MoE block -> every layer packs core + experts
    for layer in range(manifest.num_layers):
        assert manifest.has_piece(Manifest.layer_core_piece_id(layer))
        assert not manifest.has_piece(Manifest.layer_piece_id(layer))
        for e in range(_NUM_LOCAL_EXPERTS):
            assert manifest.has_piece(Manifest.layer_expert_piece_id(layer, e))


def test_gpt_oss_forward_matches_full_load(tiny_gpt_oss_model_dir, tmp_path):
    out = tmp_path / "gpt-oss.nunspark"
    pack(tiny_gpt_oss_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.has_piece(Manifest.layer_core_piece_id(0))   # selective pack

    # 6 tokens > sliding_window=4, so the windowed mask differs from a plain causal one
    tokens = [3, 7, 42, 1, 9, 15]
    ref = _reference_logits(tiny_gpt_oss_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4).item()
    finally:
        engine.close()


def test_gpt_oss_quant_forward_matches_full_load(tiny_gpt_oss_quant_model_dir, tmp_path):
    out = tmp_path / "gpt-oss-q4.nunspark"
    pack(tiny_gpt_oss_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9, 15]
    ref = _reference_logits(tiny_gpt_oss_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        # quantized matmuls are deterministic -> streamed output is bit-identical
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_gpt_oss_mixed_quant_forward_matches_full_load(
        tiny_gpt_oss_mixed_quant_model_dir, tmp_path):
    # Heterogeneous checkpoint (mxfp4 experts + 8-bit-affine attn/router/embed/head).
    # The engine must build each module from its OWN resolved config, incl. mode;
    # experts have no quant biases while attention does. Bit-identical to full-load.
    out = tmp_path / "gpt-oss-mixed.nunspark"
    pack(tiny_gpt_oss_mixed_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.has_piece(Manifest.layer_core_piece_id(0))   # selective pack

    tokens = [3, 7, 42, 1, 9, 15]
    ref = _reference_logits(tiny_gpt_oss_mixed_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        # quantized matmuls are deterministic -> streamed output is bit-identical
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_gpt_oss_mixed_quant_tiny_budget(tiny_gpt_oss_mixed_quant_model_dir, tmp_path):
    # Same mixed checkpoint under a budget too small to keep any expert piece
    # resident: scatter-then-discard must still be bit-identical.
    out = tmp_path / "gpt-oss-mixed.nunspark"
    pack(tiny_gpt_oss_mixed_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9, 15]
    ref = _reference_logits(tiny_gpt_oss_mixed_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_gpt_oss_quant_tiny_budget(tiny_gpt_oss_quant_model_dir, tmp_path):
    # A budget too small to keep any piece resident must still be correct: each
    # fired expert's rows are scattered in and the piece may be evicted immediately.
    out = tmp_path / "gpt-oss-q4.nunspark"
    pack(tiny_gpt_oss_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9, 15]
    ref = _reference_logits(tiny_gpt_oss_quant_model_dir, tokens)
    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()
