import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from quant_helpers import load_quantized_model


def _reference_quant_logits(model_dir, tokens):
    """Full-load the quantized model the way mlx-lm would, and run it."""
    model, _ = load_quantized_model(model_dir)
    out = model(mx.array(tokens)[None])
    mx.eval(out)
    return out


def test_streamed_quant_forward_matches_full_load(tiny_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-q4.nunspark"
    pack(tiny_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_quant_logits(tiny_quant_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        # quantized matmuls are deterministic -> streamed output is bit-identical
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_streamed_quant_forward_matches_with_tiny_budget(tiny_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-q4.nunspark"
    pack(tiny_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = [3, 7, 42, 1, 9]
    ref = _reference_quant_logits(tiny_quant_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=1)  # no layer stays resident
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()


def test_streamed_quant_untied_forward_matches_full_load(tiny_quant_untied_model_dir, tmp_path):
    out = tmp_path / "tiny-q4-untied.nunspark"
    pack(tiny_quant_untied_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.tie_word_embeddings is False  # ensure we exercise the untied path

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_quant_logits(tiny_quant_untied_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert float(mx.max(mx.abs(got - ref))) == 0.0
    finally:
        engine.close()
