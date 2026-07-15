"""RAM-resident fast path (skip per-layer mx.eval when nothing is ever evicted).

Covers: the _fully_resident predicate (dense + MoE region-split accounting) and
bit-identity of generation/forward with the fast path forced ON vs OFF.
"""
import os

import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.piece_store import PieceStore
from nunspark.engine import StreamingEngine
from nunspark.generate import generate


def _expected_fully_resident(root, manifest, budget, frac):
    """Independent re-derivation of the predicate, region-aware, from on-disk
    file sizes — mirrors StreamingEngine._compute_fully_resident so the test
    pins the exact math (including the expert/main split) rather than a heuristic."""
    store = PieceStore(root, manifest)
    expert_bytes = 0
    main_bytes = 0
    for piece in manifest.pieces:
        pid = piece.piece_id
        if not pid.startswith("layer_"):
            continue
        size = os.stat(store.path_for(pid)).st_size
        if "_expert_" in pid:
            expert_bytes += size
        else:
            main_bytes += size
    if expert_bytes == 0:
        return main_bytes <= budget
    expert_budget = int(budget * frac)
    return expert_bytes <= expert_budget and main_bytes <= budget - expert_budget


# ---- predicate ----

def test_fast_path_enabled_when_model_fits(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)  # >> whole model
    try:
        assert engine._fully_resident is True
    finally:
        engine.close()


def test_fast_path_disabled_when_model_exceeds_budget(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    # A 1-byte budget can never hold even one layer -> must stay False.
    engine = StreamingEngine(out, manifest, budget_bytes=1)
    try:
        assert engine._fully_resident is False
    finally:
        engine.close()


def test_fast_path_disabled_just_below_full_fit(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    store = PieceStore(out, manifest)
    total = sum(
        os.stat(store.path_for(p.piece_id)).st_size
        for p in manifest.pieces if p.piece_id.startswith("layer_")
    )
    # One byte short of the whole set -> not fully resident (an eviction is
    # possible, so the syncs must stay).
    below = StreamingEngine(out, manifest, budget_bytes=total - 1)
    exact = StreamingEngine(out, manifest, budget_bytes=total)
    try:
        assert below._fully_resident is False
        assert exact._fully_resident is True
    finally:
        below.close()
        exact.close()


def test_predicate_matches_region_formula_moe(tiny_qwen3_moe_quant_model_dir, tmp_path):
    """MoE: the predicate must honor the expert/main region split, not just the
    total budget. Checked against an independent re-derivation over several
    budgets — including budget == total, where the split can make it False even
    though everything nominally 'fits'."""
    out = tmp_path / "moe.nunspark"
    manifest = pack(tiny_qwen3_moe_quant_model_dir, out)
    store = PieceStore(out, manifest)
    total = sum(
        os.stat(store.path_for(p.piece_id)).st_size
        for p in manifest.pieces if p.piece_id.startswith("layer_")
    )
    frac = 0.9  # expert_cache_frac default, and there ARE core pieces here
    for budget in (10**9, total, total * 2, total // 3, 1):
        engine = StreamingEngine(out, manifest, budget_bytes=budget)
        try:
            assert engine._fully_resident == _expected_fully_resident(
                out, manifest, budget, frac), f"budget={budget}"
        finally:
            engine.close()


# ---- bit-identity ----

def _forward_tokens(engine, tokens):
    got = engine.forward(mx.array(tokens)[None])
    mx.eval(got)
    return got


def test_forward_bit_identical_fast_path_on_off(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    tokens = [3, 7, 42, 1, 9]

    on = StreamingEngine(out, manifest, budget_bytes=10**9)
    on._fully_resident = True
    off = StreamingEngine(out, manifest, budget_bytes=10**9)
    off._fully_resident = False
    try:
        a = _forward_tokens(on, tokens)
        b = _forward_tokens(off, tokens)
        assert mx.array_equal(a, b).item()   # eval scheduling only -> identical bits
    finally:
        on.close()
        off.close()


def _generate_forced(root, manifest, force, budget=10**9):
    engine = StreamingEngine(root, manifest, budget_bytes=budget)
    engine._fully_resident = force
    try:
        return generate(engine, [3, 7, 1], max_tokens=8, temp=0.0)
    finally:
        engine.close()


def test_generate_bit_identical_dense(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    on = _generate_forced(out, manifest, force=True)
    off = _generate_forced(out, manifest, force=False)
    assert on == off


def test_generate_bit_identical_quant(tiny_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-q4.nunspark"
    manifest = pack(tiny_quant_model_dir, out)
    on = _generate_forced(out, manifest, force=True)
    off = _generate_forced(out, manifest, force=False)
    assert on == off


def test_generate_bit_identical_moe(tiny_qwen3_moe_quant_model_dir, tmp_path):
    out = tmp_path / "moe.nunspark"
    manifest = pack(tiny_qwen3_moe_quant_model_dir, out)
    on = _generate_forced(out, manifest, force=True)
    off = _generate_forced(out, manifest, force=False)
    assert on == off
