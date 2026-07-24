"""Backlog #14: the engine raises the MLX wired-memory limit by default.

Wiring is memory policy only — numerics are already covered by every
forward-parity test in the suite, all of which now run with the default
wire_limit=True. These tests pin the plumbing: the helper never raises,
the engine wires by default, and the opt-out is honored.
"""

import nunspark.engine as engine_mod
from nunspark import sysmem
from nunspark.engine import StreamingEngine
from nunspark.manifest import Manifest
from nunspark.packer import pack


def test_wire_memory_limit_never_raises():
    # On CI or non-Metal builds this returns None; on a real Mac, a positive
    # byte count. Either is fine — raising is the only failure mode.
    out = sysmem.wire_memory_limit()
    assert out is None or (isinstance(out, int) and out > 0)


def _build_engine(tmp_path, model_dir, **kwargs):
    out = tmp_path / "tiny.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9, **kwargs)


def test_engine_wires_by_default(tmp_path, tiny_qwen3_model_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "wire_memory_limit",
                        lambda: calls.append(1) or 4096)
    engine = _build_engine(tmp_path, tiny_qwen3_model_dir)
    try:
        assert calls, "default construction must attempt to wire"
        assert engine.wired_limit_bytes == 4096
    finally:
        engine.close()


def test_engine_no_wire_opt_out(tmp_path, tiny_qwen3_model_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "wire_memory_limit",
                        lambda: calls.append(1) or 4096)
    engine = _build_engine(tmp_path, tiny_qwen3_model_dir, wire_limit=False)
    try:
        assert not calls, "wire_limit=False must not touch the wired limit"
        assert engine.wired_limit_bytes is None
    finally:
        engine.close()
