import json

import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine


def test_expert_trace_schema_and_fired_sets_match_loaded_experts(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    trace_path = tmp_path / "trace.jsonl"

    # prefetch=False so `loaded` is exactly what this forward pulls (no
    # speculative next-layer core reads muddying the expert-piece spy).
    engine = StreamingEngine(out, manifest, budget_bytes=10**9, prefetch=False,
                             expert_trace=trace_path)
    loaded: list[str] = []
    orig = engine.cache._loader

    def spy(pid):
        loaded.append(pid)
        return orig(pid)

    engine.cache._loader = spy
    tokens = [3, 7, 42, 1, 9]
    try:
        engine.forward(mx.array(tokens)[None])
    finally:
        engine.close()

    records = [json.loads(l) for l in trace_path.read_text().strip().splitlines()]
    assert len(records) == manifest.num_layers    # one record per MoE layer call

    ts = [r["t"] for r in records]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)   # strictly increasing counter

    for r in records:
        assert set(r.keys()) == {"t", "layer", "fired", "batch_tokens"}
        assert r["batch_tokens"] == len(tokens)
        prefix = f"layer_{r['layer']:03d}_expert_"
        loaded_experts = {int(p[len(prefix):]) for p in loaded if p.startswith(prefix)}
        assert set(r["fired"]) == loaded_experts       # trace matches what the engine actually loaded


def test_expert_trace_off_by_default(tiny_qwen3_moe_model_dir, tmp_path):
    out = tmp_path / "moe.nunspark"
    pack(tiny_qwen3_moe_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        assert engine._trace_fh is None
        engine.forward(mx.array([[3, 7]]))    # must not error with tracing off
    finally:
        engine.close()


def test_expert_trace_output_bit_identical_on_vs_off(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = [3, 7, 42, 1, 9]

    engine_off = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        out_off = engine_off.forward(mx.array(tokens)[None])
        mx.eval(out_off)
    finally:
        engine_off.close()

    trace_path = tmp_path / "trace.jsonl"
    engine_on = StreamingEngine(out, manifest, budget_bytes=10**9, expert_trace=trace_path)
    try:
        out_on = engine_on.forward(mx.array(tokens)[None])
        mx.eval(out_on)
    finally:
        engine_on.close()

    assert float(mx.max(mx.abs(out_off - out_on))) == 0.0
    assert trace_path.exists() and trace_path.read_text().strip() != ""
