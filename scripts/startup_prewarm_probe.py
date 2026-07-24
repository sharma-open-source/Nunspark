"""Startup bulk-prewarm A/B: does sequential-reading the cache full at load
time kill the first-run cold ramp?

Question (backlog #14 follow-up / C2's parked "cold-start warming"): the first
cold run at the wired 10 GB budget is ramp-dominated — 2.68 tok/s over 200
tokens, because ~9 GB of cache fill happens at single-token demand-fault speed
(~0.45 GB/s) spread over the first ~150 decode tokens. The second run is
near-steady-state from token 1 (6.92 tok/s live) purely because the OS page
cache is warm. A startup prewarm manufactures that condition on run one: after
engine construction, raw-read the pieces the cache will want (dense/core
first, then experts round-robin across layers for even coverage) at bulk
sequential bandwidth (~1-2 GB/s, the warm_bulk mechanism), THEN decode. Unlike
the dead per-token decode warming (#9/Phase-1) this is a one-shot
known-in-advance bulk read — the same shape that made prefill warm_bulk (#1) a
2x win.

Gate metric: END-TO-END wall time (prewarm + prefill + decode) from a cold
page cache — the prewarm must pay for itself, not just flatter the decode
tok/s. House gate: >=10% median end-to-end win, no pair worse than 5%.
Mechanism evidence: the prewarm arm's first-50-token window speed should
match its last-50 window (no ramp), while control ramps.

Design (house style):
  - One child process per run (clean memory + wired state per run). Both arms
    use the shipped engine defaults (wire_limit=True) — the A/B isolates the
    prewarm only.
  - Interleaved control/prewarm arms in ABBA order per pair to cancel drift.
  - OS page cache flushed between runs (this is a COLD-start probe; the flush
    is the experiment's precondition, not just hygiene).
  - vm_stat sampled around every run; per-token timestamps give the ramp shape.
  - Token streams must be byte-identical across ALL runs (prewarming is I/O
    policy only). The driver asserts this.
  - Results -> scripts/results/startup_prewarm_ab.json (house schema).

Usage (from the repo root, on the Mac under test — close heavy apps, don't
run anything else disk-heavy alongside; ~15 min at the defaults):

  uv run python scripts/startup_prewarm_probe.py ./packed/qwen3-30b

Options:
  --budgets 10          budget sweep in GB (default: the shipped 16 GB optimum)
  --tokens 200          greedy decode tokens per run (the ramp is ~150 tokens)
  --pairs 3             (control, prewarm) pairs per budget
  --prewarm-cap-gb 0    prewarm byte cap; 0 = the engine budget (default)
  --prewarm-threads 8   raw-read pool width (warm_bulk uses 8)
  --flush-file PATH     default ./probe_flush.bin (reuses the wired-probe file;
                        KEPT after the run — delete it when done)
  --out PATH            default scripts/results/startup_prewarm_ab.json

Child mode (spawned internally, one arm per invocation):
  python startup_prewarm_probe.py child <packed_root> <budget_gb> <tokens> \
      <tag> [prewarm] [cap_gb=X] [threads=N]
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

RESULTS_DEFAULT = Path(__file__).parent / "results" / "startup_prewarm_ab.json"
WINDOW = 50  # ramp-shape window (tokens) for first/last speed comparison


# ---------------------------------------------------------------- child mode

def _select_prewarm_files(manifest, root: Path, cap_bytes: int):
    """Files to warm, in read order: every dense/core piece first (the cyclic
    scan needs them all, every token), then expert pieces round-robin across
    layers (expert-index-major) so each layer gets even coverage rather than
    early layers hogging the cap, stopping at cap_bytes. Returns (paths,
    total_bytes, n_dense, n_expert)."""
    dense, experts = [], []
    for p in manifest.pieces:
        m = re.match(r"layer_(\d+)_expert_(\d+)$", p.piece_id)
        if m:
            experts.append((int(m.group(2)), int(m.group(1)), p.file))
        else:
            dense.append(p.file)
    experts.sort()  # (expert_idx, layer_idx): round-robin layer coverage
    paths, total, n_dense, n_expert = [], 0, 0, 0
    for kind, rel in [("d", f) for f in dense] + [("e", t[2]) for t in experts]:
        path = root / rel
        size = os.stat(path).st_size
        if total + size > cap_bytes and paths:
            break
        paths.append(path)
        total += size
        if kind == "d":
            n_dense += 1
        else:
            n_expert += 1
    return paths, total, n_dense, n_expert


def _raw_read(path: Path) -> None:
    fd = os.open(os.fspath(path), os.O_RDONLY)
    try:
        while os.read(fd, 1 << 23):  # 8 MiB chunks, warm_bulk-style
            pass
    finally:
        os.close(fd)


def child_main(argv: list[str]) -> None:
    import resource
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx

    from nunspark.bench import _load_tokenizer, _encode
    from nunspark.engine import StreamingEngine
    from nunspark.generate import _open_kv_store, _prefill
    from nunspark.manifest import Manifest

    root = Path(argv[0])
    budget = int(float(argv[1]) * (1 << 30))
    n_decode = int(argv[2])
    tag = argv[3]
    prewarm = "prewarm" in argv[4:]
    cap_gb = next((float(a.split("=")[1]) for a in argv[4:]
                   if a.startswith("cap_gb=")), 0.0)
    threads = next((int(a.split("=")[1]) for a in argv[4:]
                    if a.startswith("threads=")), 8)

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    # Shipped defaults for BOTH arms (wire_limit=True): the A/B isolates the
    # startup prewarm, on top of the wired baseline that ships.
    engine = StreamingEngine(root, manifest, budget_bytes=budget)

    prewarm_info = {"prewarm_s": 0.0, "prewarm_gb": 0.0}
    if prewarm:
        cap = int(cap_gb * (1 << 30)) if cap_gb else budget
        paths, total, n_dense, n_expert = _select_prewarm_files(
            manifest, root, cap)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=threads,
                                thread_name_prefix="probe-prewarm") as pool:
            list(pool.map(_raw_read, paths))
        dt = time.monotonic() - t0
        prewarm_info = {
            "prewarm_s": round(dt, 2),
            "prewarm_gb": round(total / (1 << 30), 2),
            "prewarm_gb_s": round(total / (1 << 30) / dt, 2) if dt else None,
            "prewarm_files": {"dense": n_dense, "expert": n_expert},
            "prewarm_threads": threads,
        }

    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        t0 = time.monotonic()
        logits = _prefill(engine, ids, kv, 1024)   # last-position logits
        prefill_s = time.monotonic() - t0
        tok = int(mx.argmax(logits, axis=-1).item())
        out_ids = [tok]

        mx.reset_peak_memory()
        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        stamps: list[float] = []   # cumulative decode time after each token
        t0 = time.monotonic()
        for _ in range(n_decode - 1):
            logits = engine.forward(mx.array([[tok]]), kv=kv)
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            out_ids.append(tok)
            stamps.append(time.monotonic() - t0)
        decode_s = stamps[-1] if stamps else 0.0
        s1 = engine.cache.stats()
        stall1 = engine._stall_seconds

        n_timed = len(stamps)
        # Ramp shape: tok/s over the first and last WINDOW tokens.
        first_w = (round(WINDOW / stamps[WINDOW - 1], 3)
                   if n_timed >= 2 * WINDOW else None)
        last_w = (round(WINDOW / (stamps[-1] - stamps[-1 - WINDOW]), 3)
                  if n_timed >= 2 * WINDOW else None)

        bytes_delta = {k: s1["bytes_loaded"][k] - s0["bytes_loaded"][k]
                       for k in s1["bytes_loaded"]}
        misses = {k: s1["misses"][k] - s0["misses"][k] for k in s1["misses"]}
        total_gb = sum(bytes_delta.values()) / (1 << 30)
        out = {
            "tag": tag,
            "arm": "prewarm" if prewarm else "control",
            **prewarm_info,
            "prompt_tokens": len(ids),
            "decode_tokens_timed": n_timed,
            "budget_gb": budget / (1 << 30),
            "prefill_s": round(prefill_s, 2),
            "decode_s": round(decode_s, 2),
            "decode_tok_s": round(n_timed / decode_s, 3) if decode_s else None,
            # The gate metric: everything the user waits for, cold.
            "end_to_end_s": round(
                prewarm_info["prewarm_s"] + prefill_s + decode_s, 2),
            "first%d_tok_s" % WINDOW: first_w,
            "last%d_tok_s" % WINDOW: last_w,
            "expert_stall_s": round(stall1 - stall0, 2),
            "bytes_loaded_gb": {k: round(v / (1 << 30), 3)
                                for k, v in bytes_delta.items()},
            "mb_per_token": round(total_gb * 1024 / n_timed, 1) if n_timed else None,
            "misses": misses,
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
    page = after.get("page_size_bytes", 16384)
    keys = ("pages_occupied_by_compressor", "compressions", "decompressions",
            "swapins", "swapouts", "pageins", "pageouts")
    d = {}
    for k in keys:
        if k in before and k in after:
            d[k] = after[k] - before[k]
    if "compressions" in d:
        d["compressed_gb_during_run"] = round(d["compressions"] * page / (1 << 30), 2)
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
    block = os.urandom(4 << 20) * 16
    written = 0
    t0 = time.monotonic()
    with open(path, "wb") as f:
        while written < size_bytes:
            f.write(block)
            written += len(block)
    print(f"  wrote {written / (1 << 30):.1f} GB in "
          f"{time.monotonic() - t0:.0f}s", flush=True)


def _flush_page_cache(path: Path) -> None:
    t0 = time.monotonic()
    with open(path, "rb") as f:
        while f.read(256 << 20):
            pass
    print(f"  [flush {time.monotonic() - t0:.0f}s]", flush=True)
    time.sleep(2)


def _run_child(root: Path, budget_gb: float, tokens: int, tag: str,
               prewarm: bool, cap_gb: float, threads: int) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "child",
           str(root), str(budget_gb), str(tokens), tag]
    if prewarm:
        cmd.append("prewarm")
        if cap_gb:
            cmd.append(f"cap_gb={cap_gb}")
        cmd.append(f"threads={threads}")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"child failed ({tag}):\n{proc.stderr[-4000:]}")
    run = json.loads(proc.stdout)
    run["wall_s_child"] = round(time.monotonic() - t0, 1)
    return run


def _preflight(root: Path) -> dict:
    if sys.platform != "darwin":
        raise SystemExit("this probe must run on the macOS machine under test")
    if not (root / "manifest.json").exists():
        raise SystemExit(f"no manifest.json under {root} — pass the packed dir")
    import mlx
    import mlx.core as mx

    info = dict(mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info())
    return {
        "chip": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_gb": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / (1 << 30)),
        "macos": _sh(["sw_vers", "-productVersion"]),
        "mlx_version": getattr(mlx, "__version__", "unknown"),
        "max_recommended_working_set_gb": round(
            info.get("max_recommended_working_set_size", 0) / (1 << 30), 2),
    }


def _summarize(runs: list[dict]) -> dict:
    """Per-(budget, arm) medians. The gate reads end_to_end_s (lower = better);
    the window speeds evidence the mechanism (ramp gone vs ramp present)."""
    table: dict = {}
    for r in runs:
        table.setdefault(r["budget_gb"], {}).setdefault(r["arm"], []).append(r)
    summary = {}
    fw, lw = f"first{WINDOW}_tok_s", f"last{WINDOW}_tok_s"
    for budget in sorted(table):
        row: dict = {}
        for arm, arm_runs in table[budget].items():
            row[arm] = {
                "runs": len(arm_runs),
                "end_to_end_s_median": round(statistics.median(
                    r["end_to_end_s"] for r in arm_runs), 2),
                "end_to_end_s_all": [r["end_to_end_s"] for r in arm_runs],
                "decode_tok_s_median": round(statistics.median(
                    r["decode_tok_s"] for r in arm_runs), 3),
                f"{fw}_median": (round(statistics.median(
                    r[fw] for r in arm_runs), 2)
                    if all(r.get(fw) for r in arm_runs) else None),
                f"{lw}_median": (round(statistics.median(
                    r[lw] for r in arm_runs), 2)
                    if all(r.get(lw) for r in arm_runs) else None),
                "expert_stall_s_median": round(statistics.median(
                    r["expert_stall_s"] for r in arm_runs), 2),
            }
            if arm == "prewarm":
                row[arm]["prewarm_s_median"] = round(statistics.median(
                    r["prewarm_s"] for r in arm_runs), 2)
                row[arm]["prewarm_gb_s_median"] = round(statistics.median(
                    r["prewarm_gb_s"] for r in arm_runs
                    if r.get("prewarm_gb_s")), 2)
        if "control" in row and "prewarm" in row:
            c = row["control"]["end_to_end_s_median"]
            p = row["prewarm"]["end_to_end_s_median"]
            # Positive = prewarm arm finished faster end-to-end.
            row["end_to_end_win_pct"] = round((c - p) / c * 100, 1) if c else None
        summary[f"{budget}GB"] = row
    return summary


def driver_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_root", type=Path)
    ap.add_argument("--budgets", default="10")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--prewarm-cap-gb", type=float, default=0.0,
                    help="0 = the engine budget")
    ap.add_argument("--prewarm-threads", type=int, default=8)
    ap.add_argument("--flush-file", type=Path, default=Path("./probe_flush.bin"))
    ap.add_argument("--out", type=Path, default=RESULTS_DEFAULT)
    args = ap.parse_args()

    # Never clobber an earlier sweep's raw data (see wired_limit_probe.py:
    # the 2026-07-24 clobber). Auto-suffix _v2, _v3, ... instead.
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

    ram_bytes = ctx["ram_gb"] << 30
    _make_flush_file(args.flush_file, max(ram_bytes - (2 << 30), 4 << 30))

    n_runs = len(budgets) * args.pairs * 2
    print(f"{n_runs} runs ({args.tokens} decode tokens each) + a flush before "
          f"every run — expect roughly 15-25 min total. Don't use the machine "
          f"for heavy work meanwhile.\n", flush=True)

    runs: list[dict] = []
    for budget in budgets:
        for pair in range(args.pairs):
            # ABBA: alternate which arm goes first to cancel slow drift.
            order = [False, True] if pair % 2 == 0 else [True, False]
            for prewarm in order:
                arm = "prewarm" if prewarm else "control"
                tag = f"{arm}-{budget:g}GB-run{pair + 1}"
                print(f"[{len(runs) + 1}/{n_runs}] {tag}", flush=True)
                _flush_page_cache(args.flush_file)
                vm0 = _vm_stat()
                run = _run_child(args.packed_root, budget, args.tokens, tag,
                                 prewarm, args.prewarm_cap_gb,
                                 args.prewarm_threads)
                vm1 = _vm_stat()
                run["vm_delta"] = _vm_delta(vm0, vm1)
                runs.append(run)
                fw = run.get(f"first{WINDOW}_tok_s")
                lw = run.get(f"last{WINDOW}_tok_s")
                print(f"  -> end-to-end {run['end_to_end_s']}s "
                      f"(prewarm {run['prewarm_s']}s, decode "
                      f"{run['decode_tok_s']} tok/s; first{WINDOW} {fw}, "
                      f"last{WINDOW} {lw})", flush=True)

    # Byte-identity across ALL runs (prewarming must not change tokens).
    ref = runs[0]["token_ids"]
    mismatches = [r["tag"] for r in runs if r["token_ids"] != ref]
    for r in runs:  # keep the stored file small, house style
        r["token_ids_len"] = len(r.pop("token_ids"))
    identical = not mismatches

    summary = _summarize(runs)
    result = {
        "experiment": "startup bulk-prewarm A/B: does a one-shot sequential "
                      "read of the cache's pieces at load time kill the "
                      "first-run cold ramp (backlog #14 follow-up)?",
        "date": datetime.date.today().isoformat(),
        "hardware": f"{ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
                    f"mlx {ctx['mlx_version']}",
        "context": ctx,
        "model": f"{args.packed_root}, budgets {args.budgets} GB, "
                 f"{args.tokens} greedy decode tokens, {args.pairs} pairs/budget, "
                 f"prewarm cap {args.prewarm_cap_gb or 'budget'} GB, "
                 f"{args.prewarm_threads} prewarm threads",
        "method": "scripts/startup_prewarm_probe.py; one fresh child process "
                  "per run (both arms wired, shipped defaults), "
                  "ABBA-interleaved control/prewarm, page cache flushed by "
                  "streaming the dummy file before every run (cold-start "
                  "precondition), vm_stat deltas + per-token window speeds",
        "token_streams_identical": identical,
        "token_mismatch_tags": mismatches,
        "summary": summary,
        "verdict": "PENDING ANALYSIS — gate is END-TO-END wall time (>=10% "
                   "median win, no pair worse than 5%); window speeds must "
                   "show the ramp gone in the prewarm arm",
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print("\n=== summary (median end-to-end seconds, lower is better) ===",
          flush=True)
    for budget, row in summary.items():
        c = row.get("control", {}).get("end_to_end_s_median")
        p = row.get("prewarm", {}).get("end_to_end_s_median")
        pct = row.get("end_to_end_win_pct")
        line = f"  {budget:>5}: control {c}s  prewarm {p}s"
        if pct is not None:
            line += f"  (prewarm {pct:+}% faster end-to-end)"
        print(line, flush=True)
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
