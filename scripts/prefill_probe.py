"""Backlog #1 probe: time PREFILL (TTFT) on a packed MoE model and attribute it.

Measures one prefill forward over an ~N-token prompt:
  - wall time of engine.forward(prompt)  (== TTFT minus sampling epsilon)
  - engine._stall_seconds delta          (expert-load wait inside the pass)
  - cache stats deltas (bytes loaded per class, hits/misses)

Usage: python prefill_probe.py <packed_root> <budget_gb> <target_prompt_tokens> <tag>
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from nunspark.bench import _load_tokenizer, _encode
from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store
from nunspark.manifest import Manifest

BASE = ("Write a design essay proposing a SaaS billing system: pricing tiers, "
        "metered usage, invoicing, failed payments, proration, tax handling, "
        "refunds, chargebacks, dunning, revenue recognition, and reporting. ")


def main() -> None:
    root = Path(sys.argv[1])
    budget = int(float(sys.argv[2]) * (1 << 30))
    target_tokens = int(sys.argv[3])
    tag = sys.argv[4]
    chunk = None
    for extra in sys.argv[5:]:
        if extra == "nowarm":
            # Control arm: disable the bulk warm so both arms run the same binary.
            from nunspark.piece_cache import PieceCache
            PieceCache.warm_bulk = lambda self, pids: None
        elif extra.startswith("chunk="):
            chunk = int(extra.split("=", 1)[1])

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)

    # Grow the prompt by repetition until ~target_tokens after chat templating.
    text = BASE
    ids = _encode(tokenizer, text)
    while len(ids) < target_tokens:
        text += BASE
        ids = _encode(tokenizer, text)

    engine = StreamingEngine(root, manifest, budget_bytes=budget)
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        mx.reset_peak_memory()
        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        t0 = time.monotonic()
        if chunk is not None:
            from nunspark.generate import _prefill
            logits = _prefill(engine, ids, kv, chunk)
        else:
            logits = engine.forward(mx.array(ids)[None], kv=kv)[:, -1, :]
        first = int(mx.argmax(logits, axis=-1).item())  # forces the graph
        prefill_s = time.monotonic() - t0
        s1 = engine.cache.stats()
        stall1 = engine._stall_seconds

        bytes_delta = {k: s1["bytes_loaded"][k] - s0["bytes_loaded"][k]
                       for k in s1["bytes_loaded"]}
        total_gb = sum(bytes_delta.values()) / (1 << 30)
        out = {
            "tag": tag,
            "prompt_tokens": len(ids),
            "budget_gb": budget / (1 << 30),
            "prefill_s": round(prefill_s, 2),
            "expert_stall_s": round(stall1 - stall0, 2),
            "bytes_loaded_gb": {k: round(v / (1 << 30), 3)
                                for k, v in bytes_delta.items()},
            "effective_read_mb_s": round(
                (total_gb * 1024) / prefill_s, 1) if prefill_s else None,
            "misses": {k: s1["misses"][k] - s0["misses"][k] for k in s1["misses"]},
            "hits": {k: s1["hits"][k] - s0["hits"][k] for k in s1["hits"]},
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
            "first_token": first,
        }
        print(json.dumps(out, indent=2))
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()


if __name__ == "__main__":
    main()
