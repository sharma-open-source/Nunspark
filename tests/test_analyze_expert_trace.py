"""Tests for scripts/analyze_expert_trace.py. Loaded by path (scripts/ isn't a
package) the same way a standalone probe would be run."""
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_expert_trace.py"
_spec = importlib.util.spec_from_file_location("analyze_expert_trace", _SCRIPT)
analyze_expert_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_expert_trace)


def _synthetic_records() -> list[dict]:
    # layer 0: the SAME fired set every call -> consecutive Jaccard must be 1.0.
    # layer 1: fired set alternates between two disjoint pairs -> Jaccard 0.0.
    records = []
    t = 0
    for i in range(6):
        t += 1
        records.append({"t": t, "layer": 0, "fired": [1, 2], "batch_tokens": 1})
        t += 1
        fired = [0, 1] if i % 2 == 0 else [2, 3]
        records.append({"t": t, "layer": 1, "fired": fired, "batch_tokens": 1})
    return records


def test_analyze_produces_sane_numbers():
    records = _synthetic_records()
    results = analyze_expert_trace.analyze(records, expert_bytes={}, capacities=None)

    layer0 = results["per_layer"][0]
    layer1 = results["per_layer"][1]
    assert layer0["mean_consecutive_jaccard"] == 1.0
    assert layer1["mean_consecutive_jaccard"] == 0.0

    agg = results["aggregate"]
    assert agg["num_records"] == len(records)
    assert agg["num_layers"] == 2
    assert agg["num_experts"] == 4

    sweep = agg["cache_sweep"]
    assert sweep["capacities"]
    for rate in sweep["lru_hit_rate"] + sweep["lfu_decay_hit_rate"]:
        assert 0.0 <= rate <= 1.0

    for k, v in agg["union_by_window"].items():
        assert v >= 0.0
    assert agg["union_by_window"][1] == 2.0    # every call fires exactly 2 experts


def test_cli_runs_end_to_end_on_small_trace(tmp_path):
    records = _synthetic_records()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(r) for r in records))
    out_json = tmp_path / "out.json"

    old_argv = sys.argv
    try:
        sys.argv = ["analyze_expert_trace.py", str(trace_path), "--json", str(out_json)]
        rc = analyze_expert_trace.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "aggregate" in data and "per_layer" in data


def test_expert_bytes_changes_cache_capacity_semantics(tmp_path):
    records = _synthetic_records()
    expert_bytes = {"0:1": 1000, "0:2": 1000, "1:0": 1, "1:1": 1, "1:2": 1, "1:3": 1}
    results = analyze_expert_trace.analyze(records, expert_bytes=expert_bytes, capacities=[6])
    sweep = results["aggregate"]["cache_sweep"]
    assert sweep["bytes_mode"] == "per-expert"
    assert sweep["capacities"] == [6]
