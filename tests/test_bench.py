from nunspark.bench import run_bench, system_info, format_report
from nunspark.packer import pack


def test_run_bench_greedy_only_on_tiny_model(tiny_model_dir_with_tokenizer, tmp_path):
    packed = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir_with_tokenizer, packed)

    results = run_bench(
        packed, draft=None, budget=10**9, max_tokens=5, workloads=["prose"],
    )

    assert len(results) == 1
    r = results[0]
    assert r["label"] == "prose"
    assert r["mode"] == "greedy"
    assert r["tok_s_decode"] > 0
    assert r["tokens_generated"] > 0

    report = format_report(system_info(), results, "tiny-test-model")
    assert "tok/s" in report
    assert "prose" in report


def test_format_report_renders_markdown_table_from_fabricated_results():
    info = {
        "chip": "Apple M4", "ram_gb": "16", "macos_version": "15.5",
        "python_version": "3.11.9", "nunspark_version": "0.1.0", "mlx_version": "0.31.0",
    }
    results = [
        {
            "label": "code", "mode": "greedy", "tok_s_decode": 1.50,
            "peak_memory_gb": 10.1, "bytes_per_token": 118e6, "has_experts": True,
            "expert_hit_pct": 89.2, "budget_bytes": 8_000_000_000, "tokens_generated": 100,
        },
        {
            "label": "code", "mode": "spec", "tok_s_decode": 2.10,
            "peak_memory_gb": 10.3, "bytes_per_token": 90e6, "has_experts": True,
            "expert_hit_pct": 91.0, "budget_bytes": 8_000_000_000, "tokens_generated": 100,
            "spec_stats": {"multiplier": 3.2, "draft_tokens_per_sweep": 24},
        },
        {
            "label": "prose", "mode": "greedy", "tok_s_decode": 0.95,
            "peak_memory_gb": 9.5, "bytes_per_token": None, "has_experts": False,
            "expert_hit_pct": None, "budget_bytes": 8_000_000_000, "tokens_generated": 100,
        },
    ]

    report = format_report(info, results, "mlx-community/Qwen3-30B-A3B-4bit")

    assert "### NunSpark bench -- mlx-community/Qwen3-30B-A3B-4bit" in report
    assert "Apple M4" in report and "16 GB RAM" in report and "15.5" in report
    assert "| workload | mode | tok/s | M | expert hit% | MB/token | peak GB |" in report
    assert "| code | greedy | 1.50 |" in report
    assert "3.20" in report          # spec multiplier for the code/spec row
    assert "89.2" in report          # expert hit% for the code/greedy row
    assert "\N{EM DASH}" in report   # dense-model row shows "—" for expert hit%/MB-token
    assert "settings:" in report
    assert "budget=8GB" in report
    assert "K=24" in report


def test_system_info_returns_all_keys_and_never_raises():
    info = system_info()
    for key in ("chip", "ram_gb", "macos_version", "python_version",
                "nunspark_version", "mlx_version"):
        assert key in info
        assert info[key] is not None
