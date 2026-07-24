"""Draft-model spec vs greedy break-even on the wired baseline (backlog #5 / #10).

Question: every prior spec-vs-greedy number predates wiring (backlog #14).
Greedy on 16 GB moved 3.2 -> ~8 tok/s, which shifts every break-even toward
greedy; meanwhile the small-K sweep (#5: "K=2-4 could break even, union tax
shrinks ~5x") was never measured, and the wired world adds a new cost: the
draft model's ~0.5-1 GB must fit UNDER the 11.84 GB wire alongside the budget
(rule from #14: budget + overhead < max_recommended), so spec arms here run at
a 9 GB budget while greedy keeps its 10 GB optimum — each mode at its own best
config, which is the user-facing question ("which mode should I run?").
N-gram spec needs no re-run: ngram_adaptive_ab_v4 measured M=1.05-1.24 on this
model and the adaptive drafter already auto-disables.

Design (house style):
  - Arms: greedy @ 10 GB (control; known wired baseline ~8.0), spec with the
    bench-default matched draft (Qwen/Qwen3-0.6B) @ 9 GB, K in {4, 8, 16}.
  - One fresh child process per run (shipped engine defaults -> wired);
    generation via the SAME entry points as `nunspark bench`
    (stream_generate / speculative_generate, accept_top_k=1, eos honored),
    prefill/decode split timed like bench._run_one.
  - Rounds interleaved (order reversed on alternate rounds), page cache
    flushed before every run, vm_stat deltas.
  - Token identity: spec must emit the target's greedy stream; fp16 near-tie
    flips across verify shapes are a KNOWN characterized exception (CLAUDE.md)
    — the driver reports divergence position/counts per run instead of hard
    asserting; near_tie_rows from SpecStats gives the mechanism check.
  - Preflight resolves/downloads the draft in a throwaway subprocess BEFORE
    any timed run (backlog #6: never download mid-run).
  - Results -> scripts/results/spec_breakeven_wired.json.

Gate (house): a spec arm must beat the greedy median by >=10% (no
worse-round-than-5% clause across its rounds) to change any default/README
guidance; otherwise record the numbers and close #5's small-K question.

Usage (repo root, on the Mac, nothing disk-heavy alongside; ~20-30 min):

  uv run python scripts/spec_breakeven_probe.py ./packed/qwen3-30b

Options:
  --greedy-budget 10    control budget in GB (the wired 16 GB optimum)
  --spec-budget 9       spec-arm budget in GB (draft must fit under the wire)
  --ks 4,8,16           num_draft_tokens sweep
  --draft Qwen/Qwen3-0.6B
  --tokens 300          max new tokens per run (eos may stop earlier)
  --rounds 2            interleaved rounds over all arms
  --flush-file PATH     default ./probe_flush.bin (KEPT after the run)
  --out PATH            default scripts/results/spec_breakeven_wired.json

Child mode (spawned internally, one arm per invocation):
  python spec_breakeven_probe.py child <packed_root> <budget_gb> <tokens> \
      <tag> [spec] [k=N] [draft=ID]
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

PROMPT = ("Explain, step by step, how a modern operating system schedules "
          "threads across performance and efficiency cores, and what a "
          "userspace developer can do to cooperate with the scheduler.")

DRAFT_DEFAULT = "Qwen/Qwen3-0.6B"
RESULTS_DEFAULT = Path(__file__).parent / "results" / "spec_breakeven_wired.json"


# ---------------------------------------------------------------- child mode

def child_main(argv: list[str]) -> None:
    import resource

    import mlx.core as mx

    from nunspark.bench import _load_tokenizer, _encode
    from nunspark.engine import StreamingEngine
    from nunspark.generate import stream_generate, speculative_generate, SpecStats
    from nunspark.manifest import Manifest

    root = Path(argv[0])
    budget = int(float(argv[1]) * (1 << 30))
    max_tokens = int(argv[2])
    tag = argv[3]
    spec = "spec" in argv[4:]
    k = next((int(a.split("=")[1]) for a in argv[4:]
              if a.startswith("k=")), 8)
    draft = next((a.split("=", 1)[1] for a in argv[4:]
                  if a.startswith("draft=")), DRAFT_DEFAULT)

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)
    eos = getattr(tokenizer, "eos_token_id", None)

    draft_model = None
    draft_load_s = 0.0
    if spec:
        from mlx_lm import load as load_full
        t0 = time.monotonic()
        draft_model, _draft_tok = load_full(draft)
        draft_load_s = time.monotonic() - t0

    # Shipped defaults (wire_limit=True): both modes run wired.
    engine = StreamingEngine(root, manifest, budget_bytes=budget)
    try:
        spec_stats = SpecStats() if spec else None
        if spec:
            gen = speculative_generate(
                engine, draft_model, ids, max_tokens=max_tokens,
                num_draft_tokens=k, accept_top_k=1, eos_id=eos,
                stats=spec_stats)
        else:
            gen = stream_generate(engine, ids, max_tokens=max_tokens, temp=0.0)

        mx.reset_peak_memory()
        t0 = time.monotonic()
        first = next(gen)
        prefill_s = time.monotonic() - t0
        out_ids = [first]

        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        t1 = time.monotonic()
        if first != eos:
            for tok in gen:
                out_ids.append(tok)
                if tok == eos:
                    break
        decode_s = time.monotonic() - t1
        s1 = engine.cache.stats()
        stall1 = engine._stall_seconds

        n_decode = len(out_ids) - 1
        bytes_delta = {kk: s1["bytes_loaded"][kk] - s0["bytes_loaded"][kk]
                       for kk in s1["bytes_loaded"]}
        misses = {kk: s1["misses"][kk] - s0["misses"][kk] for kk in s1["misses"]}
        total_gb = sum(bytes_delta.values()) / (1 << 30)
        out = {
            "tag": tag,
            "arm": f"spec-k{k}" if spec else "greedy",
            "draft": draft if spec else None,
            "draft_load_s": round(draft_load_s, 1) if spec else None,
            "prompt_tokens": len(ids),
            "budget_gb": budget / (1 << 30),
            "tokens_emitted": len(out_ids),
            "stopped_at_eos": bool(out_ids and out_ids[-1] == eos),
            "prefill_s": round(prefill_s, 2),
            "decode_s": round(decode_s, 2),
            "decode_tok_s": round(n_decode / decode_s, 3) if decode_s else None,
            "expert_stall_s": round(stall1 - stall0, 2),
            "bytes_loaded_gb": {kk: round(v / (1 << 30), 3)
                                for kk, v in bytes_delta.items()},
            "mb_per_token": round(total_gb * 1024 / n_decode, 1) if n_decode else None,
            "misses": misses,
            "expert_misses_per_token": round(misses["expert"] / n_decode, 2)
            if n_decode else None,
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
            # macOS ru_maxrss is bytes (Linux is KB; this probe is macOS-only).
            "peak_rss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
            "token_ids": out_ids,
        }
        if spec_stats is not None:
            out["spec_stats"] = {
                "target_passes": spec_stats.target_passes,
                "draft_tokens_proposed": spec_stats.draft_tokens_proposed,
                "accepted_total": spec_stats.accepted_total,
                "multiplier": round(spec_stats.multiplier, 3),
                "deviation_rate": round(spec_stats.deviation_rate, 4),
                "near_tie_rows": spec_stats.near_tie_rows,
                "k": k,
            }
        print(json.dumps(out, indent=2))
    finally:
        engine.close()


# --------------------------------------------------------------- driver mode

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
               spec: bool, k: int, draft: str) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "child",
           str(root), str(budget_gb), str(tokens), tag]
    if spec:
        cmd += ["spec", f"k={k}", f"draft={draft}"]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"child failed ({tag}):\n{proc.stderr[-4000:]}")
    run = json.loads(proc.stdout)
    run["wall_s_child"] = round(time.monotonic() - t0, 1)
    return run


def _preflight(root: Path, draft: str) -> dict:
    if sys.platform != "darwin":
        raise SystemExit("this probe must run on the macOS machine under test")
    if not (root / "manifest.json").exists():
        raise SystemExit(f"no manifest.json under {root} — pass the packed dir")
    import mlx
    import mlx.core as mx

    info = dict(mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info())
    # Resolve/download the draft in a throwaway process BEFORE any timed run
    # (backlog #6: the first-ever spec run must not download mid-run), and
    # release its memory before the children start.
    print(f"resolving draft {draft} (throwaway subprocess; downloads if "
          f"missing)...", flush=True)
    r = subprocess.run(
        [sys.executable, "-c",
         f"from mlx_lm import load; load({draft!r}); print('draft ok')"],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"draft resolve failed:\n{r.stderr[-2000:]}")
    return {
        "chip": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_gb": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / (1 << 30)),
        "macos": _sh(["sw_vers", "-productVersion"]),
        "mlx_version": getattr(mlx, "__version__", "unknown"),
        "max_recommended_working_set_gb": round(
            info.get("max_recommended_working_set_size", 0) / (1 << 30), 2),
    }


def _divergence(ref: list[int], got: list[int]) -> dict:
    n = min(len(ref), len(got))
    div = next((i for i in range(n) if ref[i] != got[i]), None)
    if div is None and len(ref) == len(got):
        return {"identical_to_greedy_ref": True}
    return {
        "identical_to_greedy_ref": False,
        "first_divergence_at": div if div is not None else n,
        "len_ref": len(ref),
        "len_run": len(got),
    }


def _summarize(runs: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    greedy_med = (statistics.median(
        r["decode_tok_s"] for r in by_arm.get("greedy", []))
        if by_arm.get("greedy") else None)
    summary = {}
    for arm in sorted(by_arm, key=lambda a: (a != "greedy", a)):
        arm_runs = by_arm[arm]
        toks = [r["decode_tok_s"] for r in arm_runs]
        row = {
            "runs": len(arm_runs),
            "budget_gb": arm_runs[0]["budget_gb"],
            "tok_s_median": round(statistics.median(toks), 3),
            "tok_s_all": toks,
            "mb_per_token_median": round(statistics.median(
                r["mb_per_token"] for r in arm_runs), 1),
            "expert_stall_s_median": round(statistics.median(
                r["expert_stall_s"] for r in arm_runs), 2),
            "peak_mem_gb_max": max(r["peak_mem_gb"] for r in arm_runs),
        }
        if arm != "greedy":
            row["multiplier_median"] = round(statistics.median(
                r["spec_stats"]["multiplier"] for r in arm_runs), 3)
            row["deviation_rate_median"] = round(statistics.median(
                r["spec_stats"]["deviation_rate"] for r in arm_runs), 4)
            if greedy_med:
                row["vs_greedy_pct"] = round(
                    (row["tok_s_median"] - greedy_med) / greedy_med * 100, 1)
        summary[arm] = row
    return summary


def driver_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_root", type=Path)
    ap.add_argument("--greedy-budget", type=float, default=10.0)
    ap.add_argument("--spec-budget", type=float, default=9.0)
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=2)
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

    ks = [int(x) for x in args.ks.split(",")]
    ctx = _preflight(args.packed_root, args.draft)
    print(f"machine: {ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
          f"mlx {ctx['mlx_version']}", flush=True)

    ram_bytes = ctx["ram_gb"] << 30
    _make_flush_file(args.flush_file, max(ram_bytes - (2 << 30), 4 << 30))

    # (spec?, k, budget) per arm; greedy first, then the K sweep.
    arms = [(False, 0, args.greedy_budget)] + \
           [(True, k, args.spec_budget) for k in ks]
    n_runs = len(arms) * args.rounds
    print(f"{n_runs} runs (max {args.tokens} new tokens each; greedy @ "
          f"{args.greedy_budget:g} GB, spec @ {args.spec_budget:g} GB + draft "
          f"{args.draft}) + a flush before every run — expect ~20-30 min. "
          f"Don't use the machine for heavy work meanwhile.\n", flush=True)

    runs: list[dict] = []
    done = 0
    for rnd in range(args.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for spec, k, budget in order:
            arm = f"spec-k{k}" if spec else "greedy"
            tag = f"{arm}-{budget:g}GB-run{rnd + 1}"
            done += 1
            print(f"[{done}/{n_runs}] {tag}", flush=True)
            _flush_page_cache(args.flush_file)
            vm0 = _vm_stat()
            run = _run_child(args.packed_root, budget, args.tokens, tag,
                             spec, k, args.draft)
            vm1 = _vm_stat()
            run["vm_delta"] = _vm_delta(vm0, vm1)
            runs.append(run)
            extra = ""
            if spec:
                ss = run["spec_stats"]
                extra = (f", M {ss['multiplier']}, deviation "
                         f"{ss['deviation_rate']}, near-ties "
                         f"{ss['near_tie_rows']}")
            print(f"  -> {run['decode_tok_s']} tok/s "
                  f"({run['tokens_emitted']} tokens, "
                  f"{run['mb_per_token']} MB/tok, peak "
                  f"{run['peak_mem_gb']} GB{extra})", flush=True)

    # Identity vs the first greedy run: spec deviation is a KNOWN characterized
    # exception (fp16 near-ties across verify shapes) — report, don't assert.
    greedy_runs = [r for r in runs if r["arm"] == "greedy"]
    ref = greedy_runs[0]["token_ids"]
    for r in runs:
        r["identity"] = _divergence(ref, r["token_ids"])
    greedy_mismatch = [r["tag"] for r in greedy_runs
                       if not r["identity"].get("identical_to_greedy_ref")]
    for r in runs:  # keep the stored file small, house style
        r["token_ids_len"] = len(r.pop("token_ids"))

    summary = _summarize(runs)
    result = {
        "experiment": "draft-model spec vs greedy break-even on the wired "
                      "baseline (backlog #5 small-K / #10 REMAINING): can any "
                      "K beat the ~8 tok/s wired greedy optimum, with the "
                      "draft paying its memory out of the wire headroom?",
        "date": datetime.date.today().isoformat(),
        "hardware": f"{ctx['chip']} {ctx['ram_gb']} GB, macOS {ctx['macos']}, "
                    f"mlx {ctx['mlx_version']}",
        "context": ctx,
        "model": f"{args.packed_root} + draft {args.draft}; greedy @ "
                 f"{args.greedy_budget:g} GB, spec @ {args.spec_budget:g} GB, "
                 f"K sweep {args.ks}, max {args.tokens} new tokens, "
                 f"{args.rounds} interleaved rounds",
        "method": "scripts/spec_breakeven_probe.py; fresh wired child per run "
                  "via the bench entry points (accept_top_k=1, eos honored), "
                  "rounds interleaved with order reversed, page cache flushed "
                  "before every run, vm_stat deltas; identity vs greedy "
                  "reported per-run (near-tie deviation is a characterized "
                  "exception, not a failure)",
        "greedy_runs_mismatched": greedy_mismatch,
        "summary": summary,
        "verdict": "PENDING ANALYSIS — house gate: a spec arm must beat the "
                   "greedy median by >=10% (no round worse than 5%) to change "
                   "any default or README guidance",
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print("\n=== summary (median decode tok/s) ===", flush=True)
    for arm, row in summary.items():
        line = (f"  {arm:>9} @ {row['budget_gb']:g} GB: {row['tok_s_median']}"
                f" tok/s, {row['mb_per_token_median']} MB/tok")
        if "multiplier_median" in row:
            line += (f", M {row['multiplier_median']}"
                     f", vs greedy {row.get('vs_greedy_pct', '?'):+}%")
        print(line, flush=True)
    if greedy_mismatch:
        print(f"WARNING: greedy runs disagree with each other: "
              f"{greedy_mismatch}", flush=True)
    print(f"\nresults -> {args.out}", flush=True)
    print(f"(flush file kept at {args.flush_file} — delete it when the "
          f"investigation is done)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child_main(sys.argv[2:])
    else:
        driver_main()
