#!/usr/bin/env python
"""Plan 5 M3b-2 (docs/plan5-m3-design.md Sec 12): batched decode bench, on a
real packed model, of `batched_generate` vs sequential `generate()`.

For the LARGEST requested batch size B_max, builds a fixed prompt set of the
FIRST B_max entries of `nunspark.bench.PROMPTS` (or, with --prompt-file, the
first B_max lines of that file, cycling if the file is shorter). Every
requested B then uses the FIRST B prompts of that SAME fixed list -- B=1 is
prompt[0], B=4 is prompts[0..3], etc -- so results are directly comparable
across B and the sequential baseline is computed once and sliced.

Sequential baseline: for each prompt in the B_max set, a FRESH engine runs
`generate()` (greedy, temp=0) to `max_tokens`, recording wall time and output
token ids. For each requested B, the "sequential total tok/s" is the sum of
that B's prompt subset's tokens generated divided by the sum of their wall
times (i.e. what B independent sequential runs would cost in aggregate).

Batched arms: for each B in --batch-sizes (B=1 IS run batched -- it validates
parity with the pad-free path), a fresh engine runs `batched_generate()` over
prompts[:B], recording wall time, total tokens emitted (sum of per-row output
lengths -- rows may stop early at eos, `generate()` does not stop at eos so
sequential rows always run the full max_tokens), peak memory
(`mx.get_peak_memory()`), weight bytes loaded (`engine.cache.stats()`), and,
via the engine's `expert_trace` hook, the per-decode-step fired-expert union
size (mean |fired| and |fired|/num_experts over records with
`batch_tokens == B`, which isolates single-token decode steps from the
larger-batch_tokens prefill windows).

IMPORTANT (docs/plan5-m3-design.md Sec 15, G-M3b-1 gate record): at real-model
scale, batched rows are NOT expected to be byte-identical to the sequential
run of the same prompt -- left-padded RoPE position shifts and batched-GEMM
reduction tiling produce rare argmax near-tie flips that diverge from the
sequential run (an mlx batched-kernel property, reproduced in stock mlx_lm,
not a NunSpark correctness bug; proven separately by the M3b-1 bit-identical
unpadded/B=1 tests and per-row prefill-logit-match tests). This script reports
per-row exact-match / common-prefix-length as DATA, never asserts it.

Usage (synthetic smoke test against a tiny packed model):
  uv run python scripts/bench_batched_decode.py --packed /tmp/tiny-moe.nunspark \\
      --budget 100MB --max-tokens 8 --batch-sizes 1,2 \\
      --out /tmp/batched_decode_smoke.json

Real Qwen3-30B-A3B-4bit, 8 GB budget (Gate G-M3b-2 run):
  uv run python scripts/bench_batched_decode.py \\
      --packed ./packed/qwen3-30b --budget 8GB --batch-sizes 1,2,4,8 \\
      --max-tokens 200 --out scripts/results/batched_decode_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from nunspark.bench import PROMPTS, _encode, _load_tokenizer, _parse_size
from nunspark.engine import StreamingEngine
from nunspark.generate import batched_generate, generate
from nunspark.manifest import Manifest

GATE_B = 4
GATE_SPEEDUP = 2.0


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _load_prompt_ids(tokenizer, prompt_file: str | None, n_needed: int) -> list[list[int]]:
    """First `n_needed` texts of PROMPTS (or --prompt-file's lines), cycled if
    the source is shorter than n_needed, each encoded to token ids."""
    if prompt_file is not None:
        lines = [l.strip() for l in Path(prompt_file).read_text().splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"{prompt_file} has no non-empty lines")
        texts = [lines[i % len(lines)] for i in range(n_needed)]
    else:
        texts = [PROMPTS[i % len(PROMPTS)][1] for i in range(n_needed)]
    return [_encode(tokenizer, t) for t in texts]


def _run_sequential(packed: Path, manifest: Manifest, budget_bytes: int,
                     prompts_ids: list[list[int]], max_tokens: int) -> list[dict]:
    """Fresh engine per prompt, greedy `generate()` to completion. Returns one
    dict per prompt with wall time, tok/s, weight bytes loaded, and out_ids
    (kept for the batched-arm exact-match/common-prefix comparison, stripped
    before JSON serialization)."""
    results = []
    for ids in prompts_ids:
        engine = StreamingEngine(packed, manifest, budget_bytes=budget_bytes)
        try:
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            out_ids = generate(engine, ids, max_tokens=max_tokens, temp=0.0)
            wall_s = time.perf_counter() - t0
            cache_stats = engine.cache.stats()
            bytes_loaded = sum(cache_stats["bytes_loaded"].values())
        finally:
            engine.close()
        tok_s = len(out_ids) / wall_s if wall_s > 0 else 0.0
        results.append({
            "prompt_tokens": len(ids),
            "tokens_generated": len(out_ids),
            "wall_s": wall_s,
            "tok_s": tok_s,
            "bytes_loaded": bytes_loaded,
            "out_ids": out_ids,
        })
    return results


def _run_batched(packed: Path, manifest: Manifest, budget_bytes: int,
                  prompts_ids: list[list[int]], max_tokens: int, eos,
                  trace_path: Path) -> tuple[list[list[int]], float, int, int]:
    """Fresh engine, one `batched_generate` call over all of prompts_ids, with
    expert-union tracing enabled. Returns (outputs, wall_s, peak_memory_bytes,
    bytes_loaded_total)."""
    engine = StreamingEngine(packed, manifest, budget_bytes=budget_bytes,
                              expert_trace=trace_path)
    try:
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        outputs = batched_generate(engine, prompts_ids, max_tokens=max_tokens,
                                    temp=0.0, eos_id=eos)
        wall_s = time.perf_counter() - t0
        peak_bytes = mx.get_peak_memory()
        cache_stats = engine.cache.stats()
        bytes_loaded = sum(cache_stats["bytes_loaded"].values())
    finally:
        engine.close()
    return outputs, wall_s, peak_bytes, bytes_loaded


def _expert_union_stats(trace_path: Path, B: int, num_experts: int | None) -> dict:
    """Mean fired-expert union size (and fraction of num_experts) over DECODE
    steps only (`batch_tokens == B`, i.e. one position per row -- excludes the
    larger-batch_tokens prefill windows). `fired` in each record already IS the
    union over the B batch rows for that layer call (engine.py flattens
    router indices across the batch before recording), so no extra
    set-union step is needed here -- just average len(fired)."""
    text = trace_path.read_text().strip()
    if not text:
        return {"mean_union_size": None, "mean_union_frac": None,
                 "num_decode_records": 0, "num_experts": num_experts}
    records = [json.loads(line) for line in text.splitlines()]
    decode_records = [r for r in records if r["batch_tokens"] == B]
    if num_experts is None:
        all_fired = [e for r in records for e in r["fired"]]
        num_experts = (max(all_fired) + 1) if all_fired else None
    if not decode_records:
        return {"mean_union_size": None, "mean_union_frac": None,
                 "num_decode_records": 0, "num_experts": num_experts}
    sizes = [len(r["fired"]) for r in decode_records]
    mean_size = sum(sizes) / len(sizes)
    frac = (mean_size / num_experts) if num_experts else None
    return {
        "mean_union_size": mean_size,
        "mean_union_frac": frac,
        "num_decode_records": len(decode_records),
        "num_experts": num_experts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", required=True, help="already-packed model dir")
    ap.add_argument("--budget", default="8GB",
                     help='resident weight budget, e.g. 512MB, 8GB (default: 8GB)')
    ap.add_argument("--batch-sizes", default="1,2,4,8",
                     help="comma-separated batch sizes to bench (default: 1,2,4,8)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=None,
                     help="use this file's lines (cycled) instead of nunspark.bench.PROMPTS "
                          "as the fixed prompt list; every batch size B still uses the FIRST "
                          "B entries of the same fixed list, so results stay comparable "
                          "across B")
    ap.add_argument("--out", default="scripts/results/batched_decode_bench.json",
                     help="path to write JSON results "
                          "(default: scripts/results/batched_decode_bench.json)")
    args = ap.parse_args()

    packed = Path(args.packed)
    manifest = Manifest.load(packed / "manifest.json")
    budget_bytes = _parse_size(args.budget)
    tokenizer = _load_tokenizer(packed)
    eos = getattr(tokenizer, "eos_token_id", None)

    batch_sizes = sorted({int(x) for x in args.batch_sizes.split(",")})
    b_max = max(batch_sizes)
    num_experts = manifest.config.get("num_experts")

    prompts_ids = _load_prompt_ids(tokenizer, args.prompt_file, b_max)

    print(f"=== sequential baseline ({b_max} prompts, fresh engine each) ===", flush=True)
    seq_results = _run_sequential(packed, manifest, budget_bytes, prompts_ids, args.max_tokens)
    for i, r in enumerate(seq_results):
        print(f"  prompt[{i}] ({r['prompt_tokens']} prompt tok): "
              f"{r['tokens_generated']} gen, {r['tok_s']:.2f} tok/s, "
              f"{r['wall_s']:.2f}s", flush=True)

    per_b: dict[int, dict] = {}
    for B in batch_sizes:
        subset_ids = prompts_ids[:B]
        seq_subset = seq_results[:B]
        seq_tokens = sum(r["tokens_generated"] for r in seq_subset)
        seq_wall = sum(r["wall_s"] for r in seq_subset)
        seq_tok_s = (seq_tokens / seq_wall) if seq_wall > 0 else 0.0

        print(f"\n=== batched B={B} ===", flush=True)
        fd, trace_name = tempfile.mkstemp(suffix=".jsonl", prefix="nunspark_bdb_trace_")
        os.close(fd)
        trace_path = Path(trace_name)
        try:
            outputs, wall_s, peak_bytes, bytes_loaded = _run_batched(
                packed, manifest, budget_bytes, subset_ids, args.max_tokens, eos, trace_path)
            union_stats = _expert_union_stats(trace_path, B, num_experts)
        finally:
            trace_path.unlink(missing_ok=True)

        total_tokens = sum(len(row) for row in outputs)
        batched_tok_s = (total_tokens / wall_s) if wall_s > 0 else 0.0
        per_request_tok_s = (batched_tok_s / B) if B else 0.0
        bytes_per_token = (bytes_loaded / total_tokens) if total_tokens else 0.0
        speedup = (batched_tok_s / seq_tok_s) if seq_tok_s > 0 else None

        rows = []
        for i, row in enumerate(outputs):
            seq_row = seq_subset[i]["out_ids"]
            exact = row == seq_row
            prefix_len = _common_prefix_len(row, seq_row)
            rows.append({
                "row": i,
                "row_len": len(row),
                "seq_len": len(seq_row),
                "exact_match": exact,
                "common_prefix_len": prefix_len,
            })
            print(f"  row[{i}]: len={len(row)} seq_len={len(seq_row)} "
                  f"exact_match={exact} common_prefix_len={prefix_len}", flush=True)

        speedup_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
        print(f"  batched total tok/s={batched_tok_s:.2f}  "
              f"sequential total tok/s={seq_tok_s:.2f}  speedup={speedup_str}  "
              f"peak_mem={peak_bytes / 1e9:.3f}GB  "
              f"union={union_stats['mean_union_size']}/{union_stats['num_experts']}", flush=True)

        per_b[B] = {
            "batch_size": B,
            "wall_s": wall_s,
            "total_tokens": total_tokens,
            "batched_total_tok_s": batched_tok_s,
            "per_request_tok_s": per_request_tok_s,
            "sequential_total_tok_s": seq_tok_s,
            "sequential_tokens": seq_tokens,
            "sequential_wall_s": seq_wall,
            "speedup": speedup,
            "peak_memory_bytes": peak_bytes,
            "bytes_loaded_total": bytes_loaded,
            "bytes_per_token": bytes_per_token,
            "expert_union": union_stats,
            "rows": rows,
        }

    # --- summary table ---
    print("\n=== summary ===")
    header = (f"{'B':>3} | {'batched tok/s':>13} | {'sequential tok/s':>16} | "
              f"{'speedup':>8} | {'peak GB':>8} | {'union/experts':>14}")
    print(header)
    print("-" * len(header))
    for B in batch_sizes:
        r = per_b[B]
        sp = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "n/a"
        u = r["expert_union"]
        u_str = (f"{u['mean_union_size']:.1f}/{u['num_experts']}"
                 if u["mean_union_size"] is not None and u["num_experts"] else "n/a")
        print(f"{B:>3} | {r['batched_total_tok_s']:>13.2f} | "
              f"{r['sequential_total_tok_s']:>16.2f} | {sp:>8} | "
              f"{r['peak_memory_bytes'] / 1e9:>8.3f} | {u_str:>14}")

    gate_pass = None
    if GATE_B in per_b:
        sp = per_b[GATE_B]["speedup"]
        gate_pass = sp is not None and sp >= GATE_SPEEDUP
        verdict = "PASS" if gate_pass else "FAIL"
        sp_str = f"{sp:.2f}x" if sp is not None else "n/a"
        print(f"\nGATE (B={GATE_B} >= {GATE_SPEEDUP:.0f}x): {verdict}  (measured {sp_str})")
    else:
        print(f"\nGATE (B={GATE_B} >= {GATE_SPEEDUP:.0f}x): SKIPPED "
              f"(B={GATE_B} not in --batch-sizes {args.batch_sizes})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "packed": str(packed),
            "budget_bytes": budget_bytes,
            "batch_sizes": batch_sizes,
            "max_tokens": args.max_tokens,
            "num_experts": num_experts,
            "gate_b": GATE_B,
            "gate_speedup_threshold": GATE_SPEEDUP,
            "gate_pass": gate_pass,
            "note": (
                "Batched rows are NOT expected to be byte-identical to the "
                "sequential generate() run of the same prompt at real-model "
                "scale (left-pad RoPE position shift + batched-GEMM reduction "
                "tiling produce rare argmax near-tie divergence -- an mlx "
                "batched-kernel property, not a correctness bug; see "
                "docs/plan5-m3-design.md Sec 15, G-M3b-1 gate record). Per-row "
                "exact_match/common_prefix_len below is reported as data, "
                "never asserted. generate() also does not stop at eos "
                "(unlike batched_generate's per-row freeze), so seq_len may "
                "legitimately exceed row_len even under perfect agreement."
            ),
        },
        "sequential_baseline": [
            {k: v for k, v in r.items() if k != "out_ids"} for r in seq_results
        ],
        "by_batch_size": per_b,
    }, indent=2))
    print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
