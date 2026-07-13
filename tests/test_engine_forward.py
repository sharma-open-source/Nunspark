import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine


def _reference_logits(model_dir, tokens):
    config = json.loads((model_dir / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    model = Model(args)
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    out = model(mx.array(tokens)[None])
    mx.eval(out)
    return out


def test_streamed_forward_matches_full_load(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert got.shape == ref.shape
        assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4).item()
    finally:
        engine.close()


def test_streamed_forward_matches_with_tiny_budget(tiny_model_dir, tmp_path):
    # A budget too small to cache anything must still be correct (just no reuse).
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = [3, 7, 42, 1, 9]
    ref = _reference_logits(tiny_model_dir, tokens)

    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        got = engine.forward(mx.array(tokens)[None])
        mx.eval(got)
        assert mx.allclose(got, ref, atol=1e-4, rtol=1e-4).item()
    finally:
        engine.close()


def test_warm_window_prefetches_future_layers(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    engine = StreamingEngine(out, manifest, budget_bytes=10**9,
                             io_threads=4, warm_window=3)
    calls = []
    real = engine.cache.prefetch

    def spy(pids):
        pids = list(pids)
        calls.append(pids)
        return real(pids)

    engine.cache.prefetch = spy
    try:
        engine.forward(mx.array([3, 7, 42, 1, 9])[None])
    finally:
        engine.close()

    assert calls, "prefetch should be called during forward"
    assert all(len(c) <= 3 for c in calls)         # never exceeds warm_window
    assert max(len(c) for c in calls) >= 2         # window >1 actually engaged (n=4)
    flat = [pid for c in calls for pid in c]
    assert flat and all(manifest.has_piece(pid) for pid in flat)  # all valid layers


def test_streamed_forward_bit_identical_with_warmer(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = [3, 7, 42, 1, 9]

    base = StreamingEngine(out, manifest, budget_bytes=10**9)  # io_threads=1, W=1
    try:
        a = base.forward(mx.array(tokens)[None])
        mx.eval(a)
    finally:
        base.close()

    warm = StreamingEngine(out, manifest, budget_bytes=10**9,
                           io_threads=4, warm_window=4)
    try:
        b = warm.forward(mx.array(tokens)[None])
        mx.eval(b)
    finally:
        warm.close()

    assert mx.array_equal(a, b).item()  # warmer changes only timing -> identical
