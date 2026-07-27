"""Plan 8 / M1 — compact fired-only expert scatter, engine implementation.

M0 (test_compact_scatter_numerics.py) proved a *prototype* compact scatter
bit-identical to stock. This tests the SHIPPED engine flag
(`StreamingEngine(compact_scatter=True)`): its output must be byte-for-byte
identical to the default full-buffer path, across prefill + decode, on both
bench MoE archs and both quantizations, and it must actually take the compact
path (counter moves). Losslessness is the product — this guards it.
"""
import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest
from nunspark.packer import pack

import tempfile

PROMPT_TOKENS = [3, 7, 42, 1, 9, 2, 5, 11]
N_DECODE = 8


def _decode_logits(engine, tokens, n_decode):
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_compact_kv_")
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


def _assert_flag_bit_identical(model_dir, tmp_path):
    packed = tmp_path / "compact.nunspark"
    pack(model_dir, packed)
    manifest = Manifest.load(packed / "manifest.json")

    full = StreamingEngine(packed, manifest, budget_bytes=10**9, wire_limit=False)
    try:
        ref = _decode_logits(full, PROMPT_TOKENS, N_DECODE)
    finally:
        full.close()

    compact = StreamingEngine(packed, manifest, budget_bytes=10**9,
                              wire_limit=False, compact_scatter=True)
    try:
        got = _decode_logits(compact, PROMPT_TOKENS, N_DECODE)
        # The compact path must actually have been exercised.
        assert compact.compact_rows_scattered > 0
        assert compact.prefetch_stats()["compact_rows_scattered"] > 0
    finally:
        compact.close()

    assert len(ref) == len(got)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape, f"step {i}: shape {a.shape} != {b.shape}"
        assert bool(mx.all(a == b).item()), (
            f"step {i}: compact_scatter=True diverged from the full-buffer path "
            f"(max abs diff {float(mx.max(mx.abs(a - b)).item()):.2e})")


def test_compact_scatter_flag_qwen3_moe_fp16(tiny_qwen3_moe_model_dir, tmp_path):
    _assert_flag_bit_identical(tiny_qwen3_moe_model_dir, tmp_path)


def test_compact_scatter_flag_qwen3_moe_q4(tiny_qwen3_moe_quant_model_dir, tmp_path):
    _assert_flag_bit_identical(tiny_qwen3_moe_quant_model_dir, tmp_path)


def test_compact_scatter_flag_gpt_oss_fp16(tiny_gpt_oss_model_dir, tmp_path):
    _assert_flag_bit_identical(tiny_gpt_oss_model_dir, tmp_path)


def test_compact_scatter_flag_gpt_oss_q4(tiny_gpt_oss_quant_model_dir, tmp_path):
    _assert_flag_bit_identical(tiny_gpt_oss_quant_model_dir, tmp_path)


def test_compact_scatter_default_off(tiny_qwen3_moe_model_dir, tmp_path):
    """Default engine must not touch the compact path at all."""
    packed = tmp_path / "off.nunspark"
    pack(tiny_qwen3_moe_model_dir, packed)
    manifest = Manifest.load(packed / "manifest.json")
    engine = StreamingEngine(packed, manifest, budget_bytes=10**9, wire_limit=False)
    try:
        engine.forward(mx.array(PROMPT_TOKENS)[None])
        assert engine.compact_rows_scattered == 0
    finally:
        engine.close()
