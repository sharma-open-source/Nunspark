from nunspark.cli import main, _parse_size, build_parser
from nunspark.manifest import Manifest


def test_parse_size_units():
    assert _parse_size("512") == 512
    assert _parse_size("1KB") == 1000
    assert _parse_size("2MB") == 2_000_000
    assert _parse_size("1.5GB") == 1_500_000_000


def test_cli_pack_then_generate(tiny_model_dir, tmp_path, capsys):
    out = tmp_path / "tiny.nunspark"
    rc = main(["pack", str(tiny_model_dir), str(out)])
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert Manifest.load(out / "manifest.json").num_layers == 4

    rc = main(["generate", str(out), "--prompt-ids", "3,7,1",
               "--max-tokens", "4", "--budget", "256MB"])
    assert rc == 0
    printed = capsys.readouterr().out.strip()
    assert len(printed.split()) == 4


def test_generate_cli_kv_budget_arg_accepted(tiny_model_dir, tmp_path, capsys):
    packed = tmp_path / "packed"
    assert main(["pack", str(tiny_model_dir), str(packed)]) == 0
    capsys.readouterr()
    rc = main(["generate", str(packed), "--prompt-ids", "3,7,42",
               "--max-tokens", "4", "--kv-budget", "1"])
    assert rc == 0
    toks = capsys.readouterr().out.split()
    assert len(toks) == 4          # 4 generated token ids, KV streamed at budget=1


def test_generate_cli_io_warmer_flags_bit_identical(tiny_model_dir, tmp_path, capsys):
    packed = tmp_path / "packed"
    assert main(["pack", str(tiny_model_dir), str(packed)]) == 0
    capsys.readouterr()

    assert main(["generate", str(packed), "--prompt-ids", "3,7,42",
                 "--max-tokens", "6", "--temp", "0"]) == 0
    base = capsys.readouterr().out.split()

    assert main(["generate", str(packed), "--prompt-ids", "3,7,42",
                 "--max-tokens", "6", "--temp", "0",
                 "--io-threads", "4", "--warm-window", "4"]) == 0
    warm = capsys.readouterr().out.split()

    assert len(base) == 6
    assert base == warm  # warmer is opt-in and only changes timing


def test_cli_serve_dispatches_to_run_server_with_parsed_args(monkeypatch, tiny_model_dir, tmp_path):
    packed = tmp_path / "packed"
    assert main(["pack", str(tiny_model_dir), str(packed)]) == 0

    calls = []
    monkeypatch.setattr(
        "nunspark.cli.run_server",
        lambda packed_dir, host, port, **kwargs: calls.append((packed_dir, host, port, kwargs)),
    )

    rc = main(["serve", str(packed),
               "--host", "0.0.0.0", "--port", "9000", "--model-name", "my-model",
               "--budget", "512MB", "--kv-budget", "256MB",
               "--io-threads", "4", "--warm-window", "2", "--no-prefetch"])

    assert rc == 0
    assert len(calls) == 1
    packed_dir, host, port, kwargs = calls[0]
    assert (packed_dir, host, port) == (str(packed), "0.0.0.0", 9000)
    assert kwargs == {
        "model_name": "my-model",
        "budget_bytes": 512_000_000,
        "kv_budget": 256_000_000,
        "prefetch": False,
        "io_threads": 4,
        "warm_window": 2,
        "draft_model_path": None,
        "num_draft_tokens": 16,
        "accept_top_k": 1,
        "kv_quant": None,
        "use_prefix_cache": True,
        "lookahead_prefetch": False,
        "wire_limit": True,
        "compact_scatter": False,
        "eval_window": 1,
    }


def test_cli_serve_uses_documented_defaults(monkeypatch, tiny_model_dir, tmp_path):
    packed = tmp_path / "packed"
    assert main(["pack", str(tiny_model_dir), str(packed)]) == 0

    # --budget now defaults to "auto" (derived from unified RAM); pin RAM so
    # the resolved byte count is deterministic: 16 GiB -> 10 GiB budget
    # (auto = 0.75 * RAM - 2 GiB, the measured WIRED 30B optimum on 16 GiB).
    monkeypatch.setattr("nunspark.sysmem.unified_ram_bytes", lambda: 16 * 2**30)

    calls = []
    monkeypatch.setattr(
        "nunspark.cli.run_server",
        lambda packed_dir, host, port, **kwargs: calls.append((packed_dir, host, port, kwargs)),
    )

    assert main(["serve", str(packed)]) == 0
    packed_dir, host, port, kwargs = calls[0]
    assert (packed_dir, host, port) == (str(packed), "127.0.0.1", 8080)
    assert kwargs == {
        "model_name": None,
        "budget_bytes": 10 * 2**30,
        "kv_budget": 10**12,
        "prefetch": True,
        "io_threads": 1,
        "warm_window": 1,
        "draft_model_path": None,
        "num_draft_tokens": 16,
        "accept_top_k": 1,
        "kv_quant": None,
        "use_prefix_cache": True,
        "lookahead_prefetch": False,
        "wire_limit": True,
        "compact_scatter": False,
        "eval_window": 1,
    }


def test_cli_kv_bits_flags_parse():
    parser = build_parser()
    args = parser.parse_args(["generate", "pk", "--kv-bits", "8",
                              "--kv-group-size", "32"])
    assert args.kv_bits == 8 and args.kv_group_size == 32
    args = parser.parse_args(["serve", "pk", "--kv-bits", "4"])
    assert args.kv_bits == 4 and args.kv_group_size == 64
    args = parser.parse_args(["generate", "pk"])
    assert args.kv_bits is None


def test_cli_lookahead_flag_parses_default_off_for_all_subcommands():
    # plan7 M2: --lookahead is opt-in (default off) on generate/serve/bench.
    parser = build_parser()
    for sub in (["generate", "pk"], ["serve", "pk"], ["bench"]):
        args = parser.parse_args(sub)
        assert args.lookahead is False
    for sub in (["generate", "pk", "--lookahead"],
                ["serve", "pk", "--lookahead"],
                ["bench", "--lookahead"]):
        args = parser.parse_args(sub)
        assert args.lookahead is True


def test_cli_serve_lookahead_flag_reaches_run_server(monkeypatch, tiny_model_dir, tmp_path):
    # plan7 M2 gate: the CLI flag must actually reach the engine constructor
    # kwarg (lookahead_prefetch), mirroring how the other serve dispatch
    # tests assert plumb-through via the mocked run_server call.
    packed = tmp_path / "packed"
    assert main(["pack", str(tiny_model_dir), str(packed)]) == 0

    calls = []
    monkeypatch.setattr(
        "nunspark.cli.run_server",
        lambda packed_dir, host, port, **kwargs: calls.append(kwargs),
    )

    assert main(["serve", str(packed), "--lookahead"]) == 0
    assert calls[-1]["lookahead_prefetch"] is True

    assert main(["serve", str(packed)]) == 0
    assert calls[-1]["lookahead_prefetch"] is False
