"""Plan 9 / M1-lite — windowed per-layer eval, engine implementation.

The `eval_window=W` flag drains the per-layer materialization barrier every W-th
layer instead of every layer (pipelining ~W layers). mx.eval is scheduling-only,
so the output MUST be byte-for-byte identical to the default W=1 path, across
prefill + decode, on both bench MoE archs and both quantizations, and the flag
must actually take effect (prefetch_stats reports it). Losslessness is the
product — this guards it. Measured +7.6% decode at W=3 (windowed_eval_probe),
below the 10% default-on bar, hence shipped opt-in (default W=1).
"""
import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest
from nunspark.packer import pack

import tempfile

PROMPT_TOKENS = [3, 7, 42, 1, 9, 2, 5, 11]
N_DECODE = 8
WINDOW = 3


def _decode_logits(engine, tokens, n_decode):
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_window_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    out = []
    try:
        logits = _prefill(engine, tokens, kv, 1024)
        out.append(mx.array(logits))
        tok = int(mx.argmax(logits, axis=-1).item())
        for _ in range(n_decode):
            logits = engine.forward(mx.array([[tok]]), kv=kv)[:, -1, :]
            mx.eval(logits)
            out.append(mx.array(logits))
            tok = int(mx.argmax(logits, axis=-1).item())
        return out
    finally:
        kv.close()
        tmp.cleanup()


def _assert_window_bit_identical(model_dir, tmp_path):
    packed = tmp_path / "window.nunspark"
    pack(model_dir, packed)
    manifest = Manifest.load(packed / "manifest.json")

    # Force the streaming barrier path (wire_limit=False + a budget below full
    # residency would evict; here we keep a generous budget but disable the
    # _fully_resident skip so _sync_layer's windowing actually runs) — set
    # wire_limit=False so both engines share identical numerics conditions.
    ref = StreamingEngine(packed, manifest, budget_bytes=10**9, wire_limit=False)
    ref._fully_resident = False
    try:
        base = _decode_logits(ref, PROMPT_TOKENS, N_DECODE)
    finally:
        ref.close()

    win = StreamingEngine(packed, manifest, budget_bytes=10**9,
                          wire_limit=False, eval_window=WINDOW)
    win._fully_resident = False
    try:
        got = _decode_logits(win, PROMPT_TOKENS, N_DECODE)
        assert win.prefetch_stats()["eval_window"] == WINDOW
    finally:
        win.close()

    assert len(base) == len(got)
    for i, (a, b) in enumerate(zip(base, got)):
        assert a.shape == b.shape, f"step {i}: shape {a.shape} != {b.shape}"
        assert bool(mx.all(a == b).item()), (
            f"step {i}: eval_window={WINDOW} diverged from W=1 "
            f"(max abs diff {float(mx.max(mx.abs(a - b)).item()):.2e})")


def test_eval_window_qwen3_moe_fp16(tiny_qwen3_moe_model_dir, tmp_path):
    _assert_window_bit_identical(tiny_qwen3_moe_model_dir, tmp_path)


def test_eval_window_qwen3_moe_q4(tiny_qwen3_moe_quant_model_dir, tmp_path):
    _assert_window_bit_identical(tiny_qwen3_moe_quant_model_dir, tmp_path)


def test_eval_window_gpt_oss_fp16(tiny_gpt_oss_model_dir, tmp_path):
    _assert_window_bit_identical(tiny_gpt_oss_model_dir, tmp_path)


def test_eval_window_dense_llama_fp16(tiny_model_dir, tmp_path):
    # dense arch: _sync_layer is on the same path, windowing must be exact there too
    _assert_window_bit_identical(tiny_model_dir, tmp_path)


def test_eval_window_default_is_one(tiny_qwen3_moe_model_dir, tmp_path):
    """Default engine must be W=1 (windowing off, exact per-layer eval)."""
    packed = tmp_path / "off.nunspark"
    pack(tiny_qwen3_moe_model_dir, packed)
    manifest = Manifest.load(packed / "manifest.json")
    engine = StreamingEngine(packed, manifest, budget_bytes=10**9, wire_limit=False)
    try:
        assert engine.prefetch_stats()["eval_window"] == 1
    finally:
        engine.close()
