#!/usr/bin/env python
"""M2 (Plan 5, docs/plan5-tensorfold-adoptions.md): A/B bench for the adaptive
n-gram drafter.

Runs, per workload prompt, THREE arms over a fresh engine each:
  1. greedy baseline (`generate()`)
  2. fixed-K n-gram speculative decode (`NGramDrafter(adaptive=False)`)
  3. adaptive n-gram speculative decode (`NGramDrafter(adaptive=True)`)

and asserts the emitted token ids are IDENTICAL across all three -- adaptivity
only changes proposal LENGTH, and acceptance in `ngram_speculative_generate`
is lossless-by-construction (a draft token is accepted only if it equals the
target's own argmax), so output must be bit-identical regardless of K or
adaptation. Records tok/s, SpecStats fields, and (for the adaptive arm) the
observed k_cur trajectory (k_min_seen/k_max_seen) to a JSON file, matching
Gate G-M2 in the plan: bit-identical output, and the novel-prose worst case
should improve to roughly break-even vs greedy while high-M workloads regress
<= 5%.

Workloads default to the same three (code / prose / reasoning) prompts as
`nunspark bench` / scripts/m1_baseline.py (imported from nunspark.bench.PROMPTS
so results stay comparable); pass --prompt-file to bench a custom prompt
instead (one prompt per line, each run as its own workload labelled by line
number).

Usage (synthetic smoke test against a tiny packed model):
  uv run python scripts/bench_ngram_adaptive_ab.py --packed ./packed/tiny \\
      --budget 100MB --max-tokens 20 --k-cap 6 \\
      --out /tmp/ngram_adaptive_ab_smoke.json

Real Qwen3-30B-A3B-4bit, 8 GB budget, K-cap=16 (Gate G-M2 run):
  uv run python scripts/bench_ngram_adaptive_ab.py \\
      --packed ./packed/qwen3-30b --budget 8GB --k-cap 16 --max-tokens 500 \\
      --out scripts/results/ngram_adaptive_ab.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from nunspark.bench import PROMPTS, _encode, _load_tokenizer, _parse_size
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, ngram_speculative_generate, SpecStats
from nunspark.manifest import Manifest
from nunspark.ngram_drafter import NGramDrafter


def _run_arm(packed: Path, manifest: Manifest, ids: list[int], eos, budget_bytes: int,
             max_tokens: int, mode: str, k_cap: int, min_draft_tokens: int) -> dict:
    """Build a fresh engine, run one arm (greedy / ngram-fixed / ngram-adaptive)
    to completion, and return token ids + timing + SpecStats + k trajectory."""
    engine = StreamingEngine(packed, manifest, budget_bytes=budget_bytes)
    try:
        mx.reset_peak_memory()
        stats = SpecStats() if mode != "greedy" else None
        drafter = None
        if mode == "ngram-fixed":
            drafter = NGramDrafter(num_draft_tokens=k_cap, adaptive=False)
        elif mode == "ngram-adaptive":
            drafter = NGramDrafter(num_draft_tokens=k_cap, adaptive=True,
                                    min_draft_tokens=min_draft_tokens)

        t0 = time.perf_counter()
        if mode == "greedy":
            out_ids = generate(engine, ids, max_tokens=max_tokens, temp=0.0)
        else:
            out_ids = list(ngram_speculative_generate(
                engine, drafter, ids, max_tokens=max_tokens, eos_id=eos, stats=stats))
        wall_s = time.perf_counter() - t0
        tok_s = len(out_ids) / wall_s if wall_s > 0 else 0.0
        peak_gb = mx.get_peak_memory() / 1e9
    finally:
        engine.close()

    result = {
        "mode": mode,
        "tokens_generated": len(out_ids),
        "wall_s": wall_s,
        "tok_s": tok_s,
        "peak_memory_gb": peak_gb,
        "out_ids": out_ids,
    }
    if stats is not None:
        result["spec_stats"] = {
            "target_passes": stats.target_passes,
            "tokens_emitted": stats.tokens_emitted,
            "draft_tokens_proposed": stats.draft_tokens_proposed,
            "accepted_total": stats.accepted_total,
            "accepted_offpath": stats.accepted_offpath,
            "multiplier": stats.multiplier,
            "deviation_rate": stats.deviation_rate,
        }
    if drafter is not None and getattr(drafter, "adaptive", False):
        result["k_min_seen"] = drafter.k_min_seen
        result["k_max_seen"] = drafter.k_max_seen
    return result


def _load_prompts(prompt_file: str | None) -> list[tuple[str, str]]:
    if prompt_file is None:
        return list(PROMPTS)
    lines = [l.strip() for l in Path(prompt_file).read_text().splitlines() if l.strip()]
    return [(f"prompt-{i}", text) for i, text in enumerate(lines)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", required=True, help="already-packed model dir")
    ap.add_argument("--budget", default="8GB",
                     help='resident weight budget, e.g. 512MB, 8GB (default: 8GB)')
    ap.add_argument("--k-cap", type=int, default=16,
                     help="n-gram drafter K cap, shared by fixed and adaptive arms "
                          "(default: 16)")
    ap.add_argument("--min-draft-tokens", type=int, default=2,
                     help="adaptive arm's k_cur floor (default: 2)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=None,
                     help="bench a custom prompt (one per line) instead of the three "
                          "built-in code/prose/reasoning workloads")
    ap.add_argument("--out", default="scripts/results/ngram_adaptive_ab.json",
                     help="path to write JSON results "
                          "(default: scripts/results/ngram_adaptive_ab.json)")
    args = ap.parse_args()

    packed = Path(args.packed)
    manifest = Manifest.load(packed / "manifest.json")
    budget_bytes = _parse_size(args.budget)
    tokenizer = _load_tokenizer(packed)
    eos = getattr(tokenizer, "eos_token_id", None)

    prompts = _load_prompts(args.prompt_file)

    runs: list[dict] = []
    all_ok = True
    for label, prompt in prompts:
        ids = _encode(tokenizer, prompt)
        print(f"\n=== {label} ({len(ids)} prompt tokens) ===")

        arms = {}
        for mode in ("greedy", "ngram-fixed", "ngram-adaptive"):
            print(f"  {mode} ...")
            r = _run_arm(packed, manifest, ids, eos, budget_bytes, args.max_tokens,
                         mode, args.k_cap, args.min_draft_tokens)
            arms[mode] = r
            m_str = ""
            if "spec_stats" in r:
                m_str = f", M={r['spec_stats']['multiplier']:.2f}"
            k_str = ""
            if "k_min_seen" in r:
                k_str = f", k={r['k_min_seen']}-{r['k_max_seen']}"
            print(f"    {r['tok_s']:.2f} tok/s, {r['tokens_generated']} tokens{m_str}{k_str}")

        ok = (arms["greedy"]["out_ids"] == arms["ngram-fixed"]["out_ids"] ==
              arms["ngram-adaptive"]["out_ids"])
        if not ok:
            all_ok = False
            print("  MISMATCH: arms disagree on output token ids!")
        else:
            print("  bit-identical across all three arms: OK")

        runs.append({
            "label": label,
            "prompt_tokens": len(ids),
            "bit_identical": ok,
            "arms": {
                mode: {k: v for k, v in r.items() if k != "out_ids"}
                for mode, r in arms.items()
            },
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "packed": str(packed),
            "budget_bytes": budget_bytes,
            "k_cap": args.k_cap,
            "min_draft_tokens": args.min_draft_tokens,
            "max_tokens": args.max_tokens,
            "all_bit_identical": all_ok,
        },
        "runs": runs,
    }, indent=2))
    print(f"\nwrote {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
