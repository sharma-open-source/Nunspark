"""Plan 7 M3 gate: online router-lookahead prefetch A/B (one arm per invocation).

Same harness as decode_bulkwarm_probe.py: short prefill (untimed), N greedy
decode tokens timed. The experiment arm builds the engine with
lookahead_prefetch=True (depth/topn engine defaults: 1/12). Token ids are
emitted so the outer harness can assert byte-identity between arms — the
lookahead is prefetch-only and must never change numerics.

Usage: python lookahead_ab_probe.py <packed_root> <budget_gb> <decode_tokens> <tag> [lookahead] [topn=N]
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
    lookahead = "lookahead" in sys.argv[5:]
    topn = next((int(a.split("=")[1]) for a in sys.argv[5:]
                 if a.startswith("topn=")), 12)

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    engine = StreamingEngine(root, manifest, budget_bytes=budget,
                             lookahead_prefetch=lookahead, lookahead_topn=topn)
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        logits = _prefill(engine, ids, kv, 1024)   # already last-position logits
        tok = int(mx.argmax(logits, axis=-1).item())
        out_ids = [tok]

        mx.reset_peak_memory()
        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        spec0 = (engine.cache.speculative_issued, engine.cache.speculative_used,
                 engine.cache.speculative_wasted_bytes)
        t0 = time.monotonic()
        for _ in range(n_decode - 1):
            logits = engine.forward(mx.array([[tok]]), kv=kv)
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            out_ids.append(tok)
        decode_s = time.monotonic() - t0
        s1 = engine.cache.stats()
        stall1 = engine._stall_seconds
        spec1 = (engine.cache.speculative_issued, engine.cache.speculative_used,
                 engine.cache.speculative_wasted_bytes)

        bytes_delta = {k: s1["bytes_loaded"][k] - s0["bytes_loaded"][k]
                       for k in s1["bytes_loaded"]}
        misses = {k: s1["misses"][k] - s0["misses"][k] for k in s1["misses"]}
        total_gb = sum(bytes_delta.values()) / (1 << 30)
        n_timed = len(out_ids) - 1
        out = {
            "tag": tag,
            "arm": f"lookahead-top{topn}" if lookahead else "control",
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
            "lookahead_issued": engine.lookahead_issued,
            "lookahead_skipped_core_missing": engine.lookahead_skipped_core_missing,
            "speculative_issued": spec1[0] - spec0[0],
            "speculative_used": spec1[1] - spec0[1],
            "speculative_wasted_bytes": spec1[2] - spec0[2],
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
