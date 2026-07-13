import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine


def _reference_logits(model_dir, tokens):
    """Full-load the Qwen3 model the way mlx-lm would, and run it."""
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


def test_streamed_qwen3_forward_matches_full_load(tiny_qwen3_model_dir, tmp_path):
    out = tmp_path / "tiny-qwen3.nunspark"
    pack(tiny_qwen3_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.model_type == "qwen3"

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4).item()
    finally:
        engine.close()


def test_streamed_qwen3_quant_forward_matches_full_load(tiny_qwen3_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-qwen3-q4.nunspark"
    pack(tiny_qwen3_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_quant_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        # quantized matmuls are deterministic -> streamed output is bit-identical
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_streamed_qwen3_quant_matches_with_tiny_budget(tiny_qwen3_quant_model_dir, tmp_path):
    # A budget too small to keep any layer resident must still be correct.
    out = tmp_path / "tiny-qwen3-q4.nunspark"
    pack(tiny_qwen3_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_qwen3_quant_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()
