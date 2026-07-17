"""Backlog #10: persistent expert scatter buffers must be invisible in output.

_scatter_experts reuses one full-size buffer set across layers/tokens WITHOUT
re-zeroing — a stale row from a previous layer is safe only because the expert
module's forward gathers exactly the router's `inds`. These tests pin that
invariant: an engine whose buffers are full of another pass's leftovers must
produce BITWISE the same logits as a fresh engine, on single-token decode and
multi-token (verify-shaped) passes alike.
"""
import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.manifest import Manifest
from nunspark.packer import pack


def _packed(model_dir, tmp_path):
    out = tmp_path / "moe-q4.nunspark"
    pack(model_dir, out)
    return out, Manifest.load(out / "manifest.json")


def test_stale_buffer_rows_bitwise_invisible(tiny_qwen3_moe_quant_model_dir, tmp_path):
    out, manifest = _packed(tiny_qwen3_moe_quant_model_dir, tmp_path)
    probe = mx.array([[5, 1, 33]])

    # Dirty engine: several prior passes with different inputs leave the
    # persistent buffers full of stale rows from other fired sets.
    dirty = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        dirty.forward(mx.array([[3, 7, 42, 1, 9]]))
        dirty.forward(mx.array([[11]]))
        dirty.forward(mx.array([[2, 8]]))
        assert dirty._scatter_bufs is not None   # buffers exist and were reused
        got_dirty = dirty.forward(probe)
        mx.eval(got_dirty)
    finally:
        dirty.close()

    fresh = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got_fresh = fresh.forward(probe)
        mx.eval(got_fresh)
    finally:
        fresh.close()

    assert got_dirty.dtype == got_fresh.dtype
    assert float(mx.max(mx.abs(got_dirty - got_fresh))) == 0.0


def test_buffers_persist_across_passes_and_reset_on_slot_rebuild(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    out, manifest = _packed(tiny_qwen3_moe_quant_model_dir, tmp_path)
    eng = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        eng.forward(mx.array([[3, 7]]))
        bufs = eng._scatter_bufs
        assert bufs is not None
        eng.forward(mx.array([[4]]))
        assert eng._scatter_bufs is bufs        # same dict reused, not rebuilt
        eng._make_slot(0)                       # structural rebuild invalidates
        assert eng._scatter_bufs is None
    finally:
        eng.close()
