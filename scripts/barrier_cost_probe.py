"""Probe 1 (Plan-8 follow-on / #16 verification): is barrier reduction a REAL
decode lever, or is the attributed sync time actually drained compute?

#16 attributed router_sync=19% and layer_sync=19% of decode to host<->device
sync points (inds.tolist() and mx.eval). But in a lazy/async MLX engine a "sync"
is where queued GPU work DRAINS -- so much of that time can be compute measured
at the barrier, not removable overhead. This probe separates the two by pricing
the barriers directly:

  1. L_eval   -- per-mx.eval round-trip floor (a trivial dependent op, so the
                 dispatch+sync cost is isolated from any real compute).
  2. L_tolist -- per host read-back of a top-k-sized int array (the router
                 inds.tolist(), the router_sync mechanism).
  3. count    -- ACTUAL host<->device syncs per token during a short real decode
                 (mx.eval wrapped + counted; the per-MoE-layer router tolist is
                 structural = n_moe_layers/token; +1 sample).

Then: floor_ms = evals/tok*L_eval + tolist/tok*L_tolist  vs the #16 buckets.
  floor << (router_sync + layer_sync)  => those buckets are drained COMPUTE, not
      sync overhead -> barrier reduction is NOT the lever. Stop.
  floor comparable                     => the round-trips are real -> merging /
      pinning barriers can pay; size it with Probe 2 (barrier_headroom_probe).

Usage: python barrier_cost_probe.py <packed_root> [n_decode]
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest

# #16 wired-baseline decode split (scripts/results/decode_attribution_wired.json),
# for the verdict comparison. ms/token at 147.1 ms/token total.
SIXTEEN = {"ms_per_token": 147.1, "router_sync_ms": 0.19 * 147.1,
           "layer_sync_ms": 0.19 * 147.1}
PROMPT = [3, 7, 42, 1, 9, 2, 5, 11]


def _round_trip_floor(reps: int = 2000) -> dict:
    """Cost of forcing a host<->device sync with ~no work to drain."""
    x = mx.zeros((1,), dtype=mx.float32)
    mx.eval(x)
    # L_eval: a fresh trivial dependent op per iter (x+1), eval'd -> dispatch+sync
    t0 = time.monotonic()
    for i in range(reps):
        mx.eval(x + 1)
    l_eval = (time.monotonic() - t0) / reps

    # L_tolist: read a top-k-sized int array back to host (router inds mechanism)
    inds = mx.arange(8, dtype=mx.int32)
    mx.eval(inds)
    t0 = time.monotonic()
    for i in range(reps):
        _ = (inds + (i & 1)).reshape(-1).tolist()
    l_tolist = (time.monotonic() - t0) / reps

    # L_item: scalar read-back (the sampler's host read)
    t0 = time.monotonic()
    for i in range(reps):
        _ = (x + i).item()
    l_item = (time.monotonic() - t0) / reps
    return {"L_eval_ms": l_eval * 1e3, "L_tolist_ms": l_tolist * 1e3,
            "L_item_ms": l_item * 1e3, "reps": reps}


def _count_evals(engine, tokens, n_decode: int) -> int:
    """Count mx.eval calls during n_decode single-token steps (post-prefill)."""
    import tempfile
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_barrier_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    real_eval = mx.eval
    n = {"c": 0}

    def counting_eval(*a, **k):
        n["c"] += 1
        return real_eval(*a, **k)

    try:
        logits = _prefill(engine, tokens, kv, 1024)
        tok = int(mx.argmax(logits, axis=-1).item())
        mx.eval = counting_eval           # count decode-only
        for _ in range(n_decode):
            logits = engine.forward(mx.array([[tok]]), kv=kv)[:, -1, :]
            mx.eval(logits)
            tok = int(mx.argmax(logits, axis=-1).item())
    finally:
        mx.eval = real_eval
        kv.close()
        tmp.cleanup()
    return n["c"]


def main() -> None:
    root = Path(sys.argv[1])
    n_decode = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    manifest = Manifest.load(root / "manifest.json")

    floor = _round_trip_floor()

    engine = StreamingEngine(root, manifest, budget_bytes=10 * (1 << 30))
    n_layers = manifest.num_layers
    evals_total = _count_evals(engine, PROMPT, n_decode)
    engine.close()

    evals_per_tok = evals_total / n_decode
    # router tolist is 1 per MoE layer (structural, engine.py:734); +1 sample/tok.
    tolist_per_tok = n_layers
    floor_ms = (evals_per_tok * floor["L_eval_ms"]
                + tolist_per_tok * floor["L_tolist_ms"]
                + floor["L_item_ms"])
    barrier_bucket = SIXTEEN["router_sync_ms"] + SIXTEEN["layer_sync_ms"]

    verdict = ("floor << bucket: router_sync+layer_sync is drained COMPUTE, not "
               "removable sync overhead -> barrier reduction is NOT the lever"
               if floor_ms < 0.5 * barrier_bucket else
               "floor comparable to bucket: the round-trips are real -> barrier "
               "merge/pin can pay; size it with Probe 2")

    out = {
        "probe": "barrier cost / round-trip floor (Plan-8 follow-on, #16 verify)",
        "model": str(root),
        "n_layers": n_layers,
        "n_decode": n_decode,
        "round_trip_floor": floor,
        "evals_per_token": round(evals_per_tok, 2),
        "tolist_per_token_structural": tolist_per_tok,
        "sync_floor_ms_per_token": round(floor_ms, 2),
        "sixteen_buckets_ms": {k: round(v, 2) for k, v in SIXTEEN.items()},
        "router_plus_layer_sync_ms": round(barrier_bucket, 2),
        "floor_as_frac_of_bucket": round(floor_ms / barrier_bucket, 3),
        "verdict": verdict,
    }
    print(json.dumps(out, indent=2))
    with open("scripts/results/barrier_cost_probe.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
