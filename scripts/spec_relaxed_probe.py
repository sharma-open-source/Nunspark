#!/usr/bin/env python
"""Measure the RELAXED-acceptance ceiling for speculative decoding.

Lossless greedy spec accepts a draft token only if it equals the target argmax.
A lossy 'fast mode' accepts it if it's merely PLAUSIBLE under the target — here:
the draft token is within the target's top-k. k=1 reproduces lossless greedy.

For each top-k we report:
  M             = tokens / target_passes (the streaming multiplier; higher = faster)
  deviation%    = fraction of ACCEPTED draft tokens that were NOT the target argmax
                  (i.e. the lossiness — tokens a lossless run would have rejected)
  mean_accept_p = mean target-probability of accepted tokens (plausibility; ~1 good)

M is intrinsic to the (draft,target,K,criterion) and transfers to streaming, so we
measure it cheaply with both models resident (use a same-family proxy target).
KV bookkeeping mirrors generate.speculative_generate exactly.
"""
from __future__ import annotations
import argparse
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    "Explain how streaming LLM inference works in two sentences.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "Summarize the causes of the French Revolution.",
    "What is the capital of France, and name three landmarks there?",
]


def run(target, tok, draft, prompt, K, topk, max_tokens):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text)
    tc = make_prompt_cache(target)
    dc = make_prompt_cache(draft)
    tl = target(mx.array(ids)[None], cache=tc)[:, -1, :]
    draft(mx.array(ids)[None], cache=dc)
    b = int(mx.argmax(tl, -1).item())

    tokens = 1          # the bootstrap token b
    passes = 0
    acc_draft = 0       # accepted draft tokens (excludes bonus)
    devs = 0            # accepted draft tokens that != target argmax (lossy)
    psum = 0.0
    while tokens < max_tokens:
        q = []
        di = mx.array([b])[None]
        for _ in range(K):
            dl = draft(di, cache=dc)[:, -1, :]
            nx = int(mx.argmax(dl, -1).item())
            q.append(nx)
            di = mx.array([nx])[None]

        vlog = target(mx.array([b] + q)[None], cache=tc)[0]   # [K+1, V]
        passes += 1

        m = 0
        for i in range(K):
            row = vlog[i]
            am = int(mx.argmax(row).item())
            if topk == 1:
                ok = (q[i] == am)
            else:
                tk = set(mx.argpartition(row, -topk)[-topk:].tolist())
                ok = q[i] in tk
            if not ok:
                break
            m += 1
            acc_draft += 1
            psum += float(mx.softmax(row)[q[i]].item())
            if q[i] != am:
                devs += 1
        bonus = int(mx.argmax(vlog[m]).item())

        # rollback (mirror speculative_generate)
        for c in tc:
            c.trim(K - m)
        if m < K:
            drop = (K - 1) - m
            for c in dc:
                c.trim(drop)
        else:
            draft(mx.array([q[K - 1]])[None], cache=dc)

        tokens += m + 1
        b = bonus
    return tokens, passes, acc_draft, devs, psum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/Qwen2.5-7B-4bit")
    ap.add_argument("--draft", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--k", type=int, default=24, help="draft depth (num_draft_tokens)")
    ap.add_argument("--topks", default="1,2,3,5,10", help="acceptance top-k sweep (1=lossless)")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    print(f"Loading target {args.target} ...")
    target, tok = load(args.target)
    print(f"Loading draft  {args.draft} ...")
    draft, _ = load(args.draft)
    topks = [int(x) for x in args.topks.split(",")]

    print(f"\ntarget={args.target.split('/')[-1]} draft={args.draft.split('/')[-1]} "
          f"K={args.k} prompts={len(PROMPTS)} max_tokens={args.max_tokens}\n")
    print(f"{'top-k':>6} {'M':>6} {'vs k=1':>7} {'deviation%':>11} {'mean_accept_p':>14}")
    print("-" * 50)
    base_M = None
    for tk in topks:
        TOK = PASS = ACC = DEV = 0
        PS = 0.0
        for p in PROMPTS:
            t, pa, a, d, ps = run(target, tok, draft, p, args.k, tk, args.max_tokens)
            TOK += t; PASS += pa; ACC += a; DEV += d; PS += ps
        M = TOK / max(PASS, 1)
        if base_M is None:
            base_M = M
        dev = 100.0 * DEV / max(ACC, 1)
        mp = PS / max(ACC, 1)
        print(f"{tk:>6} {M:>6.2f} {M/base_M:>6.2f}x {dev:>10.1f}% {mp:>14.3f}")
    print("\nk=1 is lossless (deviation 0). Higher k = more M but more deviation from")
    print("the greedy choice. The M-vs-deviation curve is the lossy 'fast mode' ceiling.")


if __name__ == "__main__":
    raise SystemExit(main())
