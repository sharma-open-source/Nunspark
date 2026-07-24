"""Plan 7 M3 lookahead gate RE-RUN on the wired baseline (backlog #14-5 / #9).

Question: the 2026-07-20 M3 gate failed at 6 GB unwired partly on control
variance (2.40-3.02 tok/s across six flushed control runs — same order as the
win) and partly on read amplification starving the single materialize worker.
Wiring (backlog #14) removed the variance (wired spread 0.3-0.8%) and the new
10 GB optimum cuts expert misses ~7x (41.6 -> ~5.9 per token), which shrinks
the speculative issue volume and its amplification. Does the lookahead pass
its original gate on this baseline?

Gate (unchanged from docs/plan7-lookahead.md M3 — flips --lookahead default):
  - token streams byte-identical control vs lookahead in every run;
  - median decode tok/s improvement >= 10%; no pair slower than 5%;
  - staging waste bounded (< 15% of speculative bytes issued).
FAIL handling: flag stays opt-in; record numbers in backlog and close #14-5.

Design (house style): driver spawns the UNCHANGED original child harness
(scripts/lookahead_ab_probe.py — same code that produced lookahead_ab.json,
now building a wired engine via the shipped default) one fresh process per
run; ABBA-interleaved control/lookahead pairs per topn config; page cache
flushed between runs; vm_stat deltas; results ->
scripts/results/lookahead_wired_ab.json.

Usage (repo root, on the Mac, nothing disk-heavy alongside; ~20-25 min):

  uv run python scripts/lookahead_wired_rerun.py ./packed/qwen3-30b

Options:
  --budget 10           budget in GB (default: the wired 16 GB optimum)
  --topns 8,12          lookahead top-N configs to sweep (original M3 arms)
  --tokens 200          greedy decode tokens per run
  --pairs 3             (control, lookahead) pairs per topn
  --flush-file PATH     default ./probe_flush.bin (KEPT after the run)
  --out PATH            default scripts/results/lookahead_wired_ab.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

CHILD = Path(__file__).parent / "lookahead_ab_probe.py"
RESULTS_DEFAULT = Path(__file__).parent / "results" / "lookahead_wired_ab.json"


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def _vm_stat() -> dict:
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
    import os
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
               lookahead: bool, topn: int) -> dict:
    cmd = [sys.executable, str(CHILD.resolve()),
           str(root), str(budget_gb), str(tokens), tag]
    if lookahead:
        cmd += ["lookahead", f"topn={topn}"]
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
    if not CHILD.exists():
        raise SystemExit(f"child harness missing: {CHILD}")
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


def _summarize(blocks: dict[int, list[dict]]) -> dict:
    """Per-topn medians + per-pair regressions, read against the M3 gate."""
    summary: dict = {}
    for topn, runs in blocks.items():
        ctl = [r for r in runs if r["arm"] == "control"]
        exp = [r for r in runs if r["arm"] != "control"]
        row: dict = {}
        for name, arm_runs in (("control", ctl), (f"lookahead-top{topn}", exp)):
            toks = [r["decode_tok_s"] for r in arm_runs]
            row[name] = {
                "runs": len(arm_runs),
                "tok_s_median": round(statistics.median(toks), 3),
                "tok_s_all": toks,
                "expert_stall_s_median": round(statistics.median(
                    r["expert_stall_s"] for r in arm_runs), 2),
                "misses_per_token_median": round(statistics.median(
                    r["expert_misses_per_token"] for r in arm_runs), 2),
                "mb_per_token_median": round(statistics.median(
                    r["mb_per_token"] for r in arm_runs), 1),
            }
        row[f"lookahead-top{topn}"]["speculative"] = {
            "issued_median": statistics.median(
                r["speculative_issued"] for r in exp),
            "used_median": statistics.median(
                r["speculative_used"] for r in exp),
            "wasted_gb_median": round(statistics.median(
                r["speculative_wasted_bytes"] for r in exp) / (1 << 30), 2),
        }
        c, e = row["control"]["tok_s_median"], row[f"lookahead-top{topn}"]["tok_s_median"]
        row["lookahead_vs_control_pct"] = round((e - c) / c * 100, 1) if c else None
        # Per-pair regressions (pair i = i-th control vs i-th lookahead run,
        # the ABBA partner): the gate's "no pair slower than 5%" clause.
        pairs = []
        for i in range(min(len(ctl), len(exp))):
            pairs.append(round(
                (exp[i]["decode_tok_s"] - ctl[i]["decode_tok_s"])
                / ctl[i]["decode_tok_s"] * 100, 1))
        row["pair_pcts"] = pairs
        summary[f"top{topn}"] = row
    return summary


def driver_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_root", type=Path)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--topns", default="8,12")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--flush-file", type=Path, default=Path("./probe_flush.bin"))
    ap.add_argument("--out", type=Path, default=RESULTS_DEFAULT)
    args = ap.parse_args()

    # Never clobber earlier raw data (wired_limit_probe.py lesson).
    if args.out.exists():
        n = 2
        while (cand := args.out.with_name(
                f"{args.out.stem}_v{n}{args.out.suffix}")).exists():
            n += 1
        print(f"{args.out} exists — writing to {cand} instead", flush=True)
        args.out = cand

    topns = [int(t) for t in args.topns.split(",")]
    ctx = _preflight(args.packed_root)
    print(f"machine: {ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
          f"mlx {ctx['mlx_version']}", flush=True)

    ram_bytes = ctx["ram_gb"] << 30
    _make_flush_file(args.flush_file, max(ram_bytes - (2 << 30), 4 << 30))

    n_runs = len(topns) * args.pairs * 2
    print(f"{n_runs} runs ({args.tokens} decode tokens each, budget "
          f"{args.budget:g} GB, both arms WIRED via the shipped default) + a "
          f"flush before every run — expect ~20-25 min. Don't use the machine "
          f"for heavy work meanwhile.\n", flush=True)

    blocks: dict[int, list[dict]] = {t: [] for t in topns}
    all_runs: list[dict] = []
    done = 0
    for topn in topns:
        for pair in range(args.pairs):
            order = [False, True] if pair % 2 == 0 else [True, False]
            for lookahead in order:
                arm = f"lookahead-top{topn}" if lookahead else "control"
                tag = f"{arm}-{args.budget:g}GB-run{pair + 1}"
                done += 1
                print(f"[{done}/{n_runs}] {tag}", flush=True)
                _flush_page_cache(args.flush_file)
                vm0 = _vm_stat()
                run = _run_child(args.packed_root, args.budget, args.tokens,
                                 tag, lookahead, topn)
                vm1 = _vm_stat()
                run["vm_delta"] = _vm_delta(vm0, vm1)
                blocks[topn].append(run)
                all_runs.append(run)
                extra = ""
                if lookahead:
                    extra = (f", spec used {run['speculative_used']}"
                             f"/{run['speculative_issued']}, wasted "
                             f"{run['speculative_wasted_bytes'] / (1 << 30):.1f} GB")
                print(f"  -> {run['decode_tok_s']} tok/s, stall "
                      f"{run['expert_stall_s']}s, misses/tok "
                      f"{run['expert_misses_per_token']}{extra}", flush=True)

    # Byte-identity across ALL runs (lookahead is prefetch-only policy).
    ref = all_runs[0]["token_ids"]
    mismatches = [r["tag"] for r in all_runs if r["token_ids"] != ref]
    for r in all_runs:
        r["token_ids_len"] = len(r.pop("token_ids"))
    identical = not mismatches

    summary = _summarize(blocks)
    result = {
        "experiment": "Plan 7 M3 lookahead gate re-run on the WIRED baseline "
                      "(backlog #14-5): does router-lookahead prefetch pass "
                      "its original gate now that wiring removed control "
                      "variance and the 10 GB optimum cut miss volume ~7x?",
        "date": datetime.date.today().isoformat(),
        "hardware": f"{ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
                    f"mlx {ctx['mlx_version']}",
        "context": ctx,
        "model": f"{args.packed_root}, budget {args.budget:g} GB, topns "
                 f"{args.topns}, {args.tokens} greedy decode tokens, "
                 f"{args.pairs} pairs/topn",
        "method": "scripts/lookahead_wired_rerun.py driving the UNCHANGED "
                  "original child (scripts/lookahead_ab_probe.py, engine now "
                  "wired via shipped default); fresh child per run, "
                  "ABBA-interleaved, page cache flushed between runs, vm_stat "
                  "deltas",
        "original_gate_result": "2026-07-20, 6 GB unwired: top12 +3.9% / "
                                "top8 +8.4% median, pairs to -9.5% — FAIL "
                                "(scripts/results/lookahead_ab.json)",
        "token_streams_identical": identical,
        "token_mismatch_tags": mismatches,
        "summary": summary,
        "verdict": "PENDING ANALYSIS — M3 gate: >=10% median, no pair worse "
                   "than 5%, staging waste < 15% of speculative bytes; a "
                   "PASS flips the --lookahead default ON",
        "runs": all_runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print("\n=== summary (median decode tok/s) ===", flush=True)
    for topn, row in summary.items():
        c = row["control"]["tok_s_median"]
        e = row[f"lookahead-{topn}"]["tok_s_median"]
        print(f"  {topn:>6}: control {c}  lookahead {e}  "
              f"({row['lookahead_vs_control_pct']:+}%)  pairs {row['pair_pcts']}",
              flush=True)
    print(f"token streams identical: {identical}"
          + ("" if identical else f"  MISMATCH: {mismatches}"), flush=True)
    print(f"\nresults -> {args.out}", flush=True)
    print(f"(flush file kept at {args.flush_file} — delete it when the "
          f"investigation is done)", flush=True)


if __name__ == "__main__":
    driver_main()
