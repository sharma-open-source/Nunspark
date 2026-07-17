"""Decode demand-parallel warm probe (community suggestion, 2026-07-17).

Question: does extending the post-router warm_bulk to SINGLE-token decode
passes (engine decode_bulk_warm=True) speed up streaming decode, or does the
Phase-1 "warming is net-negative for decode" verdict hold for the exact-miss
variant too?

Measures one run: short prefill (untimed), then N greedy decode tokens.
Reports decode tok/s, expert stall, bytes/misses deltas for the decode
segment only, and the generated token ids (arms must match byte-for-byte —
the warm is page-cache-only and must not change numerics).

Usage: python decode_bulkwarm_probe.py <packed_root> <budget_gb> <decode_tokens> <tag> [warm]
  warm  -> engine decode_bulk_warm=True (experiment arm); absent = control.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from nunspark.bench import _load_tokenizer, _encode
from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest

PROMPT = ("Explain, step by step, how a modern operating system schedules "
          "threads across performance and efficiency cores, and what a "
          "userspace developer can do to cooperate with the scheduler.")


def main() -> None:
    root = Path(sys.argv[1])
    budget = int(float(sys.argv[2]) * (1 << 30))
    n_decode = int(sys.argv[3])
    tag = sys.argv[4]
    warm = "warm" in sys.argv[5:]

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    engine = StreamingEngine(root, manifest, budget_bytes=budget,
                             decode_bulk_warm=warm)
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
            "arm": "warm" if warm else "control",
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
            "token_ids": out_ids,
        }
        print(json.dumps(out, indent=2))
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()


if __name__ == "__main__":
    main()
