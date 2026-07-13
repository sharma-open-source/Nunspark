"""A/B the page-cache warmer on a real packed model (naive streaming, no draft).

Each config runs on a FRESH engine (cold cache) so the comparison is fair.
Reports tok/s and engine.cache.{hits,misses}. Bit-identical is asserted across
configs (greedy, temp=0).

Usage:
  env -u VIRTUAL_ENV uv run python scripts/io_warm_bench.py \
      --packed-dir /Users/ssathananthan/Project/Expirments/qwen2-test2-32b/packed \
      --tokenizer-model mlx-community/Qwen2.5-32B-4bit \
      --budget 512MB --max-tokens 24
"""
import argparse, time
from pathlib import Path

import mlx.core as mx
from mlx_lm.tokenizer_utils import load as load_tokenizer

from nunspark.manifest import Manifest
from nunspark.engine import StreamingEngine
from nunspark.generate import generate as run_generate


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    for suf, mult in [("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000), ("B", 1)]:
        if s.endswith(suf):
            return int(float(s[: -len(suf)]) * mult)
    return int(s)


def run(packed, manifest, budget, prompt_ids, max_tokens, io_threads, warm_window):
    engine = StreamingEngine(packed, manifest, budget_bytes=budget,
                             io_threads=io_threads, warm_window=warm_window)
    try:
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = run_generate(engine, prompt_ids, max_tokens=max_tokens, temp=0.0)
        dt = time.perf_counter() - t0
        return {
            "tok_s": max_tokens / dt, "secs": dt, "tokens": out,
            "hits": engine.cache.hits, "misses": engine.cache.misses,
            "peak_gb": mx.get_peak_memory() / 1e9,
        }
    finally:
        engine.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", required=True)
    ap.add_argument("--tokenizer-model", required=True)
    ap.add_argument("--prompt", default="Explain streaming LLM inference in one sentence.")
    ap.add_argument("--budget", default="512MB")
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    packed = Path(args.packed_dir)
    manifest = Manifest.load(packed / "manifest.json")
    budget = _parse_size(args.budget)
    # Load ONLY the tokenizer (never the 32B target — that is what we stream).
    tp = Path(args.tokenizer_model)
    if not tp.exists():
        from huggingface_hub import snapshot_download
        tp = Path(snapshot_download(args.tokenizer_model, local_files_only=True))
    tokenizer = load_tokenizer(tp)
    prompt_ids = tokenizer.encode(args.prompt)

    configs = [
        ("warmer OFF (baseline)", 1, 1),
        ("K=4 W=4", 4, 4),
        ("K=8 W=4", 8, 4),
        ("K=8 W=8", 8, 8),
    ]
    print(f"pack={packed.name} layers={manifest.num_layers} budget={args.budget} "
          f"max_tokens={args.max_tokens}\n")
    print(f"{'config':<24} {'tok/s':>7} {'secs':>7} {'misses':>7} {'peakGB':>7}")
    baseline_tokens = None
    for name, k, w in configs:
        r = run(packed, manifest, budget, prompt_ids, args.max_tokens, k, w)
        if baseline_tokens is None:
            baseline_tokens = r["tokens"]
        else:
            assert r["tokens"] == baseline_tokens, f"{name} output diverged (not lossless!)"
        print(f"{name:<24} {r['tok_s']:>7.3f} {r['secs']:>7.2f} "
              f"{r['misses']:>7} {r['peak_gb']:>7.2f}")
    print("\nbit-identical across all configs: OK")


if __name__ == "__main__":
    main()
