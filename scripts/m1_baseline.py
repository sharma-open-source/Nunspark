#!/usr/bin/env python
"""M1 task 4 (Plan 4, docs/plan4-moe-streaming.md): baseline runner.

Runs 3 built-in prompts (labelled code / prose / reasoning -- workload styles
matching report.md) through greedy decode, and (when --draft is given)
speculative decode, over a packed model. For every run it records into ONE
JSON file: tok/s (decode phase, excluding prefill), tokens generated,
acceptance stats (spec mode only, from generate.SpecStats), a PieceCache.stats()
snapshot (hits/misses/bytes loaded per piece class -- dense/core/expert),
bytes-read-per-token, peak unified memory, and wall time. When --trace-dir is
given, each run also gets its own expert-trace JSONL (via the engine's
`expert_trace` kwarg) at <trace-dir>/<label>-<mode>.jsonl.

Each run builds a FRESH StreamingEngine (and lets generate() manage its own
throwaway KVStore) so PieceCache stats / peak memory / the expert trace are
independent per run, with no explicit cache-reset API needed. engine.close()
always runs (try/finally) so a trace file is flushed and closed even on error.

Usage (synthetic smoke test against a tiny packed MoE model):
  uv run python scripts/m1_baseline.py --packed ./packed/tiny-moe \\
      --max-tokens 20 --out /tmp/m1_smoke.json --trace-dir /tmp/m1_traces

Real Qwen3-30B-A3B baseline, greedy only:
  uv run python scripts/m1_baseline.py --packed ./packed/qwen3-30b \\
      --budget 8GB --max-tokens 500 \\
      --out scripts/results/m1_baseline.json \\
      --trace-dir scripts/results/m1_traces

Real Qwen3-30B-A3B baseline, greedy + speculative (Qwen3-0.6B draft, K=24):
  uv run python scripts/m1_baseline.py --packed ./packed/qwen3-30b \\
      --draft Qwen/Qwen3-0.6B --draft-tokens 24 --budget 8GB --max-tokens 500 \\
      --out scripts/results/m1_baseline.json \\
      --trace-dir scripts/results/m1_traces
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.engine import StreamingEngine
from nunspark.generate import stream_generate, speculative_generate, SpecStats

_UNITS = [("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000),
          ("T", 1000**4), ("G", 1000**3), ("M", 1000**2), ("K", 1000), ("B", 1)]


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mult in _UNITS:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


# code / prose / reasoning -- report.md's three workload styles.
PROMPTS = [
    ("code",
     "Implement an LRU cache in TypeScript with O(1) get and put operations. "
     "Provide the full class definition and briefly explain your design."),
    ("prose",
     "Write a design essay proposing a SaaS billing system: pricing tiers, "
     "metered usage, invoicing, and how you would handle failed payments and "
     "proration. Reason through the tradeoffs -- don't just list features."),
    ("reasoning",
     "A train leaves station A at 60 mph heading toward station B, which is "
     "300 miles away. Thirty minutes later, a second train leaves station B "
     "heading toward station A at 90 mph. How far from station A do the two "
     "trains meet, and how long after the first train departed? Show your "
     "work step by step."),
]


def _load_tokenizer(packed: Path, tokenizer_model: str | None):
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    if tokenizer_model:
        p = Path(tokenizer_model)
        if p.exists():
            return load_tokenizer(p)
        from huggingface_hub import snapshot_download
        local = snapshot_download(tokenizer_model)
        return load_tokenizer(Path(local))
    return load_tokenizer(packed)


def _encode(tokenizer, prompt: str) -> list[int]:
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        if isinstance(ids, str):
            ids = tokenizer.encode(ids)
    except Exception:
        ids = tokenizer.encode(prompt)
    return list(ids)


def _run_one(packed: Path, manifest: Manifest, ids: list[int], eos, label: str, mode: str,
            args, draft_model=None) -> dict:
    """Build a FRESH engine for this run (independent cache stats / peak memory /
    trace file), stream up to max_tokens greedy or speculative tokens via the
    existing generate() entry points (no hand-rolled decode loop), and return a
    metrics dict."""
    trace_path = None
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_path = str(Path(args.trace_dir) / f"{label}-{mode}.jsonl")

    engine = StreamingEngine(
        packed, manifest, budget_bytes=args.budget_bytes,
        prefetch=not args.no_prefetch, io_threads=args.io_threads,
        warm_window=args.warm_window, expert_trace=trace_path,
        expert_cache_frac=args.expert_cache_frac,
        expert_prefetch=args.expert_prefetch,
    )
    try:
        mx.reset_peak_memory()
        spec_stats = SpecStats() if mode == "spec" else None

        if mode == "spec":
            gen = speculative_generate(
                engine, draft_model, ids, max_tokens=args.max_tokens,
                num_draft_tokens=args.draft_tokens, accept_top_k=1,
                eos_id=eos, stats=spec_stats,
            )
        else:
            gen = stream_generate(engine, ids, max_tokens=args.max_tokens, temp=0.0)

        # Time-to-first-token includes prefill; everything after is decode-only.
        t0 = time.perf_counter()
        first = next(gen)
        t_prefill = time.perf_counter() - t0
        out_ids = [first]

        t1 = time.perf_counter()
        if first != eos:
            for tok in gen:
                out_ids.append(tok)
                if tok == eos:
                    break
        t_decode = time.perf_counter() - t1

        decode_tokens = len(out_ids) - 1
        tok_s_decode = decode_tokens / t_decode if t_decode > 0 and decode_tokens > 0 else 0.0
        peak_bytes = mx.get_peak_memory()
        cache_stats = engine.cache.stats()
        prefetch_stats = engine.prefetch_stats()
        bytes_loaded_total = sum(cache_stats["bytes_loaded"].values())
        tokens_total = len(out_ids)
        bytes_per_token = bytes_loaded_total / tokens_total if tokens_total else 0.0
    finally:
        engine.close()  # flushes + closes the expert trace, if any

    result = {
        "label": label,
        "mode": mode,
        "prompt_tokens": len(ids),
        "tokens_generated": len(out_ids),
        "decode_tokens": decode_tokens,
        "wall_time_prefill_s": t_prefill,
        "wall_time_decode_s": t_decode,
        "wall_time_total_s": t_prefill + t_decode,
        "tok_s_decode": tok_s_decode,
        "peak_memory_gb": peak_bytes / 1e9,
        "cache_stats": cache_stats,
        "prefetch_stats": prefetch_stats,
        "bytes_loaded_total": bytes_loaded_total,
        "bytes_per_token": bytes_per_token,
        "budget_bytes": args.budget_bytes,
        "expert_prefetch": args.expert_prefetch,
        "trace_path": trace_path,
    }
    if spec_stats is not None:
        result["spec_stats"] = {
            "target_passes": spec_stats.target_passes,
            "tokens_emitted": spec_stats.tokens_emitted,
            "draft_tokens_proposed": spec_stats.draft_tokens_proposed,
            "accepted_total": spec_stats.accepted_total,
            "accepted_offpath": spec_stats.accepted_offpath,
            "multiplier": spec_stats.multiplier,
            "deviation_rate": spec_stats.deviation_rate,
            "draft_tokens_per_sweep": args.draft_tokens,
        }
    return result


def _print_summary(results: list[dict]) -> None:
    print(f"\n{'label':>10} | {'mode':>6} | {'tok/s':>7} | {'peak GB':>8} | "
          f"{'B/tok':>10} | {'hit%':>6} | {'exp%':>6} | {'stall s':>8} | "
          f"{'spec u/w%':>9} | {'M':>5}")
    print("-" * 100)
    for r in results:
        cs = r["cache_stats"]
        hits = sum(cs["hits"].values())
        misses = sum(cs["misses"].values())
        hit_rate = hits / (hits + misses) if (hits + misses) else 0.0
        eh, em = cs["hits"]["expert"], cs["misses"]["expert"]
        exp_hit = eh / (eh + em) if (eh + em) else 0.0
        spec = cs.get("speculative", {"issued": 0, "used": 0, "wasted_bytes": 0})
        used_rate = spec["used"] / spec["issued"] if spec["issued"] else 0.0
        exp_bytes = cs["bytes_loaded"]["expert"]
        wasted_rate = spec["wasted_bytes"] / exp_bytes if exp_bytes else 0.0
        stall = r.get("prefetch_stats", {}).get("stall_seconds", 0.0)
        m = r["spec_stats"]["multiplier"] if "spec_stats" in r else 0.0
        print(f"{r['label']:>10} | {r['mode']:>6} | {r['tok_s_decode']:>7.2f} | "
              f"{r['peak_memory_gb']:>8.2f} | {r['bytes_per_token']:>10.0f} | "
              f"{hit_rate:>6.1%} | {exp_hit:>6.1%} | {stall:>8.2f} | "
              f"{used_rate:>4.0%}/{wasted_rate:>3.0%} | {m:>5.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", required=True, help="path to a packed NunSpark model dir")
    ap.add_argument("--draft", default=None,
                    help="HF repo id or local path of a draft model sharing the target's "
                         "tokenizer/vocab. When given, also runs speculative decoding "
                         "alongside greedy for every prompt.")
    ap.add_argument("--draft-tokens", type=int, default=24,
                    help="draft tokens proposed per target sweep (default 24)")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--budget", default="8GB", help="PieceCache byte budget, e.g. 512MB, 8GB")
    ap.add_argument("--out", default="results.json", help="path to write the JSON results")
    ap.add_argument("--trace-dir", default=None,
                    help="directory for per-run expert-trace JSONL files "
                         "(<label>-<mode>.jsonl); omit to disable tracing")
    ap.add_argument("--tokenizer-model", default=None,
                    help="HF repo id or local path for the tokenizer, if the packed dir "
                         "doesn't carry one (default: load from --packed)")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--io-threads", type=int, default=1)
    ap.add_argument("--warm-window", type=int, default=1)
    ap.add_argument("--expert-cache-frac", type=float, default=0.9,
                    help="fraction of --budget for the LRU expert region (plan4 M2)")
    ap.add_argument("--expert-prefetch", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="temporal expert prefetch (plan4 M3a); --no-expert-prefetch "
                         "for the A/B no-prefetch stall baseline")
    args = ap.parse_args()

    packed = Path(args.packed)
    manifest = Manifest.load(packed / "manifest.json")
    args.budget_bytes = _parse_size(args.budget)

    tokenizer = _load_tokenizer(packed, args.tokenizer_model)
    eos = getattr(tokenizer, "eos_token_id", None)

    draft_model = None
    if args.draft:
        from mlx_lm import load as load_full
        print(f"Loading draft {args.draft} ...")
        draft_model, draft_tok = load_full(args.draft)
        try:
            probe = "The quick brown fox jumps over the lazy dog 0123456789."
            if list(tokenizer.encode(probe)) != list(draft_tok.encode(probe)):
                print("WARNING: draft/target tokenizers disagree on a probe string -- "
                      "they do not share a vocab; acceptance will likely be ~0.")
        except Exception:
            pass

    results = []
    for label, prompt in PROMPTS:
        ids = _encode(tokenizer, prompt)
        print(f"\n=== {label} ({len(ids)} prompt tokens) ===")

        print("  greedy ...")
        r = _run_one(packed, manifest, ids, eos, label, "greedy", args)
        results.append(r)
        print(f"    {r['tok_s_decode']:.2f} tok/s, peak {r['peak_memory_gb']:.2f} GB, "
              f"{r['tokens_generated']} tokens")

        if draft_model is not None:
            print(f"  spec (K={args.draft_tokens}) ...")
            r = _run_one(packed, manifest, ids, eos, label, "spec", args,
                        draft_model=draft_model)
            results.append(r)
            print(f"    {r['tok_s_decode']:.2f} tok/s, M={r['spec_stats']['multiplier']:.2f}, "
                  f"peak {r['peak_memory_gb']:.2f} GB, {r['tokens_generated']} tokens")

    out_path = Path(args.out)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "packed": str(packed),
            "draft": args.draft,
            "draft_tokens": args.draft_tokens,
            "max_tokens": args.max_tokens,
            "budget": args.budget,
            "budget_bytes": args.budget_bytes,
        },
        "runs": results,
    }, indent=2))
    print(f"\nwrote {out_path}")

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
