#!/usr/bin/env python
"""Measure speculative-decoding acceptance for a (small draft, bigger target) pair.

The deciding number for "speculative streaming": how many tokens we accept per
TARGET forward pass. In the disk-streaming regime one target pass == one full
weight read of the target, so:

    streaming tok/s  ~=  M  x  (naive-stream tok/s)        where  M = tokens / target_passes

M is intrinsic to the (draft, target) pair + prompt distribution and is the SAME
whether the target is resident or streamed -- so we measure it cheaply with both
models resident.

We sweep the draft depth K (num_draft_tokens). The "deep speculation is nearly
free when disk-bound" thesis predicts M should keep rising with K (each extra
verified token is ~free per target pass) until draft divergence caps it. If M
barely moves above K=2, or M<3 even at best, the strategy is not worth building.
"""
from __future__ import annotations

import argparse
import time

from mlx_lm import load, stream_generate


PROMPTS = [
    "Explain how streaming LLM inference works in two sentences.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "Summarize the causes of the French Revolution.",
    "What is the capital of France, and name three landmarks there?",
]


def measure(target, tok, draft, prompt: str, k: int, max_tokens: int):
    """Return (n_tokens, n_target_passes, wall_s) for one prompt at depth K."""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    n_tokens = 0
    n_target_passes = 0  # each from_draft=False token == one target verification pass
    t0 = time.perf_counter()
    for resp in stream_generate(target, tok, text, max_tokens=max_tokens,
                                draft_model=draft, num_draft_tokens=k):
        n_tokens += 1
        if not resp.from_draft:
            n_target_passes += 1
    wall = time.perf_counter() - t0
    return n_tokens, n_target_passes, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/Qwen3-8B-6bit")
    ap.add_argument("--draft", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--depths", default="2,4,6,8",
                    help="comma-sep num_draft_tokens values to sweep")
    args = ap.parse_args()

    print(f"Loading target {args.target} ...")
    target, tok = load(args.target)
    print(f"Loading draft  {args.draft} ...")
    draft, _ = load(args.draft)
    depths = [int(x) for x in args.depths.split(",")]

    print(f"\nPrompts: {len(PROMPTS)}, max_tokens={args.max_tokens} each\n")
    print(f"{'K':>3} {'tokens':>7} {'tgt_pass':>9} {'M=tok/pass':>11} {'tok/s(res)':>11}")
    print("-" * 46)

    results = {}
    for k in depths:
        tot_tok = tot_pass = 0
        tot_wall = 0.0
        for p in PROMPTS:
            nt, npass, wall = measure(target, tok, draft, p, k, args.max_tokens)
            tot_tok += nt
            tot_pass += npass
            tot_wall += wall
        M = tot_tok / max(tot_pass, 1)
        res_tps = tot_tok / tot_wall
        results[k] = M
        print(f"{k:>3} {tot_tok:>7} {tot_pass:>9} {M:>11.2f} {res_tps:>11.2f}")

    best_k = max(results, key=results.get)
    best_M = results[best_k]
    print("\n=== verdict input ===")
    print(f"best multiplier M = {best_M:.2f}  at K={best_k}")
    print("Projected speculative-streaming tok/s on this 16GB machine "
          "(naive-stream baseline x M):")
    # naive-stream baselines from measured ~2.8 GB/s disk for 4-bit targets
    for name, base in [("70B-4bit", 0.08), ("30B-4bit", 0.17), ("26B-4bit", 0.20)]:
        print(f"  {name:>9}: {base:.2f} -> {base*best_M:.2f} tok/s")
    print("\nDecision gate: M>=3 -> worth building; M~1.5-2 -> physics wins, drop.")
    print("Caveat: measured with an 8B target; a 70B target + 0.6B draft will")
    print("likely accept somewhat LESS (larger capability gap). Treat M as an")
    print("optimistic anchor; the real 70B pair is the confirmatory next step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
