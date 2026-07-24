"""Wired-memory-limit A/B: does wiring the MLX buffer pool move the budget cliff?

Question (backlog #10 follow-up): the 16 GB budget cliff (6 GB optimum; 8 GB
-12%, 10 GB -59%) is attributed to "the cache fights the macOS compressor" —
but the mechanism was never isolated. NunSpark never calls
mx.set_wired_limit(), so every cache buffer is pageable anonymous memory,
which is exactly what the compressor eats. mlx-lm's own generate loop sets
the wired limit to device_info()["max_recommended_working_set_size"] before
decoding large models; our generate loop does not. If wiring moves the cliff
right even 2 GB, hit rate jumps and every point of hit rate is ~10 MB/token
off the dominant (I/O) cost. If it doesn't move, the cliff mechanism is
something else and this question closes.

Design (house style):
  - One child process per run (the wired limit is process-global, and a fresh
    process gives clean memory state per run). Child = same harness as
    decode_bulkwarm_probe.py: short prefill (untimed), N greedy decode tokens
    timed, cache/stall/bytes deltas for the decode segment only.
  - Interleaved control/wired arms in ABBA order per pair to cancel drift.
  - OS page cache flushed between runs by streaming a ~(RAM-2GB) dummy file.
  - vm_stat sampled before/after every run: compressor-pages / compressions /
    swap deltas are the direct evidence of (or against) the compressor fight.
  - Token streams must be byte-identical across ALL runs (wiring is a memory
    policy; it must never change numerics). The driver asserts this.
  - Results -> scripts/results/wired_limit_ab.json (house schema).

Usage (from the repo root, on the Mac under test — close heavy apps, don't
run anything else disk-heavy alongside):

  uv run python scripts/wired_limit_probe.py ./packed/qwen3-30b

Options:
  --budgets 6,8,10      budget sweep in GB (default: the cliff region)
  --tokens 150          greedy decode tokens per run
  --pairs 2             (control, wired) pairs per budget — 3 if you have time
  --wired-limit-gb 0    0 = device max_recommended_working_set_size (default)
  --flush-file PATH     where to put the flush file (default ./probe_flush.bin;
                        KEPT after the run for reuse — delete it when done)
  --out PATH            default scripts/results/wired_limit_ab.json

Child mode (spawned internally, one arm per invocation):
  python wired_limit_probe.py child <packed_root> <budget_gb> <tokens> <tag> \
      [wired] [limit_gb=X]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROMPT = ("Explain, step by step, how a modern operating system schedules "
          "threads across performance and efficiency cores, and what a "
          "userspace developer can do to cooperate with the scheduler.")

RESULTS_DEFAULT = Path(__file__).parent / "results" / "wired_limit_ab.json"


# ---------------------------------------------------------------- child mode

def _apply_wired_limit(limit_bytes: int) -> int:
    """Set the MLX wired limit; returns the previous limit. Raises if the
    running mlx has no wired-limit API (driver preflights this)."""
    import mlx.core as mx

    if hasattr(mx, "set_wired_limit"):
        return mx.set_wired_limit(limit_bytes)
    return mx.metal.set_wired_limit(limit_bytes)


def _device_info() -> dict:
    import mlx.core as mx

    return dict(mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info())


def child_main(argv: list[str]) -> None:
    import resource
    import tempfile

    import mlx.core as mx

    from nunspark.bench import _load_tokenizer, _encode
    from nunspark.engine import StreamingEngine
    from nunspark.generate import _open_kv_store, _prefill
    from nunspark.manifest import Manifest

    root = Path(argv[0])
    budget = int(float(argv[1]) * (1 << 30))
    n_decode = int(argv[2])
    tag = argv[3]
    wired = "wired" in argv[4:]
    limit_gb = next((float(a.split("=")[1]) for a in argv[4:]
                     if a.startswith("limit_gb=")), 0.0)

    applied_limit = None
    if wired:
        limit = (int(limit_gb * (1 << 30)) if limit_gb
                 else int(_device_info()["max_recommended_working_set_size"]))
        _apply_wired_limit(limit)
        applied_limit = limit

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    # wire_limit=False ALWAYS: this probe owns the wiring decision itself (the
    # explicit _apply_wired_limit above, wired arm only). Without this, the
    # engine's wire-by-default (shipped from this probe's own gate, backlog
    # #14) would silently wire the control arm and the A/B would compare
    # wired against wired.
    engine = StreamingEngine(root, manifest, budget_bytes=budget,
                             wire_limit=False)
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        logits = _prefill(engine, ids, kv, 1024)   # already last-position logits
        tok = int(mx.argmax(logits, axis=-1).item())
        out_ids = [tok]

        mx.reset_peak_memory()
        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        t0 = time.monotonic()
        for _ in range(n_decode - 1):
            logits = engine.forward(mx.array([[tok]]), kv=kv)
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            out_ids.append(tok)
        decode_s = time.monotonic() - t0
        s1 = engine.cache.stats()
        stall1 = engine._stall_seconds

        bytes_delta = {k: s1["bytes_loaded"][k] - s0["bytes_loaded"][k]
                       for k in s1["bytes_loaded"]}
        misses = {k: s1["misses"][k] - s0["misses"][k] for k in s1["misses"]}
        total_gb = sum(bytes_delta.values()) / (1 << 30)
        n_timed = len(out_ids) - 1
        out = {
            "tag": tag,
            "arm": "wired" if wired else "control",
            "wired_limit_gb": (round(applied_limit / (1 << 30), 2)
                               if applied_limit else None),
            "prompt_tokens": len(ids),
            "decode_tokens_timed": n_timed,
            "budget_gb": budget / (1 << 30),
            "decode_s": round(decode_s, 2),
            "decode_tok_s": round(n_timed / decode_s, 3) if decode_s else None,
            "expert_stall_s": round(stall1 - stall0, 2),
            "bytes_loaded_gb": {k: round(v / (1 << 30), 3)
                                for k, v in bytes_delta.items()},
            "mb_per_token": round(total_gb * 1024 / n_timed, 1) if n_timed else None,
            "misses": misses,
            "expert_misses_per_token": round(misses["expert"] / n_timed, 2)
            if n_timed else None,
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
            # macOS ru_maxrss is bytes (Linux is KB; this probe is macOS-only).
            "peak_rss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
            "token_ids": out_ids,
        }
        print(json.dumps(out, indent=2))
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()


# --------------------------------------------------------------- driver mode

def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def _vm_stat() -> dict:
    """Parse `vm_stat` into {metric: pages}, plus page_size_bytes."""
    raw = _sh(["vm_stat"])
    out: dict = {}
    m = re.search(r"page size of (\d+) bytes", raw)
    out["page_size_bytes"] = int(m.group(1)) if m else 16384
    for line in raw.splitlines():
        m = re.match(r'^"?([A-Za-z -]+)"?:\s+([\d.]+)\.?$', line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            out[key] = int(float(m.group(2)))
    return out


def _vm_delta(before: dict, after: dict) -> dict:
    """Deltas of the counters that evidence compressor/swap activity."""
    page = after.get("page_size_bytes", 16384)
    keys = ("pages_occupied_by_compressor", "compressions", "decompressions",
            "swapins", "swapouts", "pageins", "pageouts")
    d = {}
    for k in keys:
        if k in before and k in after:
            d[k] = after[k] - before[k]
    if "compressions" in d:
        d["compressed_gb_during_run"] = round(d["compressions"] * page / (1 << 30), 2)
    if "pages_occupied_by_compressor" in d:
        d["compressor_pages_net_gb"] = round(
            d["pages_occupied_by_compressor"] * page / (1 << 30), 2)
    return d


def _make_flush_file(path: Path, size_bytes: int) -> None:
    if path.exists() and path.stat().st_size >= size_bytes:
        print(f"flush file exists: {path} "
              f"({path.stat().st_size / (1 << 30):.1f} GB), reusing", flush=True)
        return
    free = shutil.disk_usage(path.parent).free
    if free < size_bytes + (5 << 30):
        raise SystemExit(
            f"not enough free disk for the {size_bytes / (1 << 30):.0f} GB flush "
            f"file at {path} ({free / (1 << 30):.0f} GB free). Pass --flush-file "
            f"pointing at a roomier volume or free space.")
    print(f"creating {size_bytes / (1 << 30):.0f} GB flush file at {path} "
          f"(one-time; kept for reuse)...", flush=True)
    block = os.urandom(4 << 20) * 16  # 64 MB of non-compressible-ish data
    written = 0
    t0 = time.monotonic()
    with open(path, "wb") as f:
        while written < size_bytes:
            f.write(block)
            written += len(block)
    print(f"  wrote {written / (1 << 30):.1f} GB in "
          f"{time.monotonic() - t0:.0f}s", flush=True)


def _flush_page_cache(path: Path) -> None:
    """Evict the model's pages by streaming the dummy file through the page
    cache (house method — no sudo/purge needed)."""
    t0 = time.monotonic()
    with open(path, "rb") as f:
        while f.read(256 << 20):
            pass
    print(f"  [flush {time.monotonic() - t0:.0f}s]", flush=True)
    time.sleep(2)


def _run_child(root: Path, budget_gb: float, tokens: int, tag: str,
               wired: bool, limit_gb: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "child",
           str(root), str(budget_gb), str(tokens), tag]
    if wired:
        cmd.append("wired")
        if limit_gb:
            cmd.append(f"limit_gb={limit_gb}")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"child failed ({tag}):\n{proc.stderr[-4000:]}")
    run = json.loads(proc.stdout)
    run["wall_s_incl_prefill"] = round(time.monotonic() - t0, 1)
    return run


def _preflight(root: Path) -> dict:
    """Runs in the driver process on the same Mac: capture context, verify
    the wired-limit API exists before burning 30+ minutes."""
    if sys.platform != "darwin":
        raise SystemExit("this probe must run on the macOS machine under test")
    if not (root / "manifest.json").exists():
        raise SystemExit(f"no manifest.json under {root} — pass the packed dir")
    import mlx.core as mx

    if not (hasattr(mx, "set_wired_limit")
            or hasattr(getattr(mx, "metal", None), "set_wired_limit")):
        raise SystemExit("this mlx build has no set_wired_limit API — "
                         "record that in the backlog and stop here")
    info = _device_info()
    import mlx

    return {
        "chip": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_gb": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / (1 << 30)),
        "macos": _sh(["sw_vers", "-productVersion"]),
        "mlx_version": getattr(mlx, "__version__", "unknown"),
        "max_recommended_working_set_gb": round(
            info.get("max_recommended_working_set_size", 0) / (1 << 30), 2),
        "iogpu_wired_limit_mb": _sh(["sysctl", "-n", "iogpu.wired_limit_mb"]),
    }


def _summarize(runs: list[dict]) -> dict:
    """Per-(budget, arm) medians + a preliminary read. Final verdict is the
    analyst's, from the full JSON."""
    table: dict = {}
    for r in runs:
        table.setdefault(r["budget_gb"], {}).setdefault(r["arm"], []).append(r)
    summary = {}
    for budget in sorted(table):
        row: dict = {}
        for arm, arm_runs in table[budget].items():
            toks = [r["decode_tok_s"] for r in arm_runs]
            row[arm] = {
                "runs": len(arm_runs),
                "tok_s_median": round(statistics.median(toks), 3),
                "tok_s_all": toks,
                "expert_stall_s_median": round(statistics.median(
                    r["expert_stall_s"] for r in arm_runs), 2),
                "misses_per_token_median": round(statistics.median(
                    r["expert_misses_per_token"] for r in arm_runs), 2),
                "compressed_gb_median": round(statistics.median(
                    r.get("vm_delta", {}).get("compressed_gb_during_run", 0.0)
                    for r in arm_runs), 2),
            }
        if "control" in row and "wired" in row:
            c, w = row["control"]["tok_s_median"], row["wired"]["tok_s_median"]
            row["wired_vs_control_pct"] = round((w - c) / c * 100, 1) if c else None
        summary[f"{budget}GB"] = row
    return summary


def driver_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_root", type=Path)
    ap.add_argument("--budgets", default="6,8,10")
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--wired-limit-gb", type=float, default=0.0,
                    help="0 = device max_recommended_working_set_size")
    ap.add_argument("--flush-file", type=Path, default=Path("./probe_flush.bin"))
    ap.add_argument("--out", type=Path, default=RESULTS_DEFAULT)
    args = ap.parse_args()

    # Never clobber an earlier sweep's raw data: results are the product of
    # 30-60 min of machine time (the 2026-07-24 9/10/11 sweep overwrote the
    # original 6/8/10 gate data before this guard existed — it had to be
    # reconstructed by hand). Auto-suffix _v2, _v3, ... instead.
    if args.out.exists():
        n = 2
        while (cand := args.out.with_name(
                f"{args.out.stem}_v{n}{args.out.suffix}")).exists():
            n += 1
        print(f"{args.out} exists — writing to {cand} instead", flush=True)
        args.out = cand

    budgets = [float(b) for b in args.budgets.split(",")]
    ctx = _preflight(args.packed_root)
    print(f"machine: {ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
          f"mlx {ctx['mlx_version']}", flush=True)
    print(f"wired limit for experiment arm: "
          f"{args.wired_limit_gb or ctx['max_recommended_working_set_gb']} GB "
          f"(device max_recommended = {ctx['max_recommended_working_set_gb']} GB)",
          flush=True)

    ram_bytes = ctx["ram_gb"] << 30
    _make_flush_file(args.flush_file, max(ram_bytes - (2 << 30), 4 << 30))

    n_runs = len(budgets) * args.pairs * 2
    print(f"{n_runs} runs ({args.tokens} decode tokens each) + a flush before "
          f"every run — expect roughly 30-60 min total. Don't use the machine "
          f"for heavy work meanwhile.\n", flush=True)

    runs: list[dict] = []
    for budget in budgets:
        for pair in range(args.pairs):
            # ABBA: alternate which arm goes first to cancel slow drift.
            order = [False, True] if pair % 2 == 0 else [True, False]
            for wired in order:
                arm = "wired" if wired else "control"
                tag = f"{arm}-{budget:g}GB-run{pair + 1}"
                print(f"[{len(runs) + 1}/{n_runs}] {tag}", flush=True)
                _flush_page_cache(args.flush_file)
                vm0 = _vm_stat()
                run = _run_child(args.packed_root, budget, args.tokens, tag,
                                 wired, args.wired_limit_gb)
                vm1 = _vm_stat()
                run["vm_delta"] = _vm_delta(vm0, vm1)
                runs.append(run)
                print(f"  -> {run['decode_tok_s']} tok/s, "
                      f"stall {run['expert_stall_s']}s, "
                      f"misses/tok {run['expert_misses_per_token']}, "
                      f"compressed during run "
                      f"{run['vm_delta'].get('compressed_gb_during_run', '?')} GB",
                      flush=True)

    # Byte-identity across ALL runs (budget and wiring must not change tokens).
    ref = runs[0]["token_ids"]
    mismatches = [r["tag"] for r in runs if r["token_ids"] != ref]
    for r in runs:  # keep the stored file small, house style
        r["token_ids_len"] = len(r.pop("token_ids"))
    identical = not mismatches

    summary = _summarize(runs)
    result = {
        "experiment": "wired-limit A/B: does mx.set_wired_limit move the "
                      "16 GB budget cliff (backlog #10 compressor hypothesis)?",
        "date": datetime.date.today().isoformat(),
        "hardware": f"{ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
                    f"mlx {ctx['mlx_version']}",
        "context": ctx,
        "model": f"{args.packed_root}, budgets {args.budgets} GB, "
                 f"{args.tokens} greedy decode tokens, {args.pairs} pairs/budget",
        "method": "scripts/wired_limit_probe.py; one fresh child process per "
                  "run, ABBA-interleaved control/wired, page cache flushed by "
                  "streaming the dummy file between runs, vm_stat compressor "
                  "deltas per run",
        "token_streams_identical": identical,
        "token_mismatch_tags": mismatches,
        "summary": summary,
        "verdict": "PENDING ANALYSIS — summary is preliminary; medians and "
                   "per-pair regressions to be read against the >=10%-median "
                   "no-pair-worse-than-5% house gate",
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print("\n=== summary (median tok/s) ===", flush=True)
    for budget, row in summary.items():
        c = row.get("control", {}).get("tok_s_median")
        w = row.get("wired", {}).get("tok_s_median")
        pct = row.get("wired_vs_control_pct")
        print(f"  {budget:>5}: control {c}  wired {w}  ({pct:+}%)"
              if pct is not None else f"  {budget:>5}: control {c}  wired {w}",
              flush=True)
    print(f"token streams identical: {identical}"
          + ("" if identical else f"  MISMATCH: {mismatches}"), flush=True)
    print(f"\nresults -> {args.out}", flush=True)
    print(f"(flush file kept at {args.flush_file} — delete it when the "
          f"investigation is done)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child_main(sys.argv[2:])
    else:
        driver_main()
