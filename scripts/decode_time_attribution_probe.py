"""Decode time attribution (community follow-up to the get_many refutation).

Question: expert-load wait is only ~18-19% of decode wall time at 30B@8GB —
where do the other ~80% go? Buckets every host-BLOCKING point per token:

  stall_s        demand-load wait inside _scatter_experts (engine counter)
  scatter_s      _scatter_experts wall MINUS stall (slot update + force-read)
  router_sync_s  the per-layer `inds.tolist()` host sync — drains the lazy
                 graph up to and including this layer's attention + router
  layer_sync_s   _sync_layer's mx.eval — drains this layer's expert mix
  head_sample_s  final norm/head logits + argmax .item() per token
  other_s        wall - all of the above (python orchestration, cache lookups,
                 prefetch enqueue, embed, ...)

Lazy-eval caveat (deliberate): a bucket is "where the host blocks", not "what
the GPU computes" — e.g. router_sync_s includes the attention compute drained
by that sync. That is exactly the orchestration-vs-I/O split the question
needs.

Usage: python decode_time_attribution_probe.py <packed_root> <budget_gb> <decode_tokens> [tag]
"""
import json
import sys
import tempfile
import time
import types
from pathlib import Path

import mlx.core as mx

from nunspark.bench import _load_tokenizer, _encode
from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest

PROMPT = ("Explain, step by step, how a modern operating system schedules "
          "threads across performance and efficiency cores, and what a "
          "userspace developer can do to cooperate with the scheduler.")

BUCKETS = {"router_sync_s": 0.0, "scatter_wall_s": 0.0, "layer_sync_s": 0.0}


def _timed_moe_attn_and_mix(self, slot, layer, h, mask, cache):
    # Verbatim replica of StreamingEngine._moe_attn_and_mix with host-sync
    # timing added (probe-only).
    r = slot.self_attn(slot.input_layernorm(h), mask, cache)
    h = h + r
    x = slot.post_attention_layernorm(h)
    gate_logits = getattr(slot.mlp, self._router_attr)(x)
    inds, scores = self._moe_route(self.args, gate_logits)
    t0 = time.monotonic()
    fired = sorted({int(e) for e in inds.reshape(-1).tolist()})
    BUCKETS["router_sync_s"] += time.monotonic() - t0
    batch_tokens = inds.shape[0] * inds.shape[1]
    if self._trace_fh is not None:
        self._trace_moe(layer, fired, batch_tokens)
    if self._cur_pass_multi or self._decode_bulk_warm:
        self.cache.warm_bulk(
            Manifest.layer_expert_piece_id(layer, e) for e in fired)
    t0 = time.monotonic()
    self._scatter_experts(slot, layer, fired)
    BUCKETS["scatter_wall_s"] += time.monotonic() - t0
    if self._expert_prefetch:
        self._fired_history[layer] = (fired, batch_tokens > 1)
    y = getattr(slot.mlp, self._expert_attr)(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return h + y


def _timed_sync_layer(self, h):
    if not self._fully_resident:
        t0 = time.monotonic()
        mx.eval(h)
        BUCKETS["layer_sync_s"] += time.monotonic() - t0
    return h


def main() -> None:
    root = Path(sys.argv[1])
    budget = int(float(sys.argv[2]) * (1 << 30))
    n_decode = int(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "attribution"

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    engine = StreamingEngine(root, manifest, budget_bytes=budget)
    engine._moe_attn_and_mix = types.MethodType(_timed_moe_attn_and_mix, engine)
    engine._sync_layer = types.MethodType(_timed_sync_layer, engine)
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        logits = _prefill(engine, ids, kv, 1024)
        tok = int(mx.argmax(logits, axis=-1).item())

        for k in BUCKETS:
            BUCKETS[k] = 0.0
        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        head_s = 0.0
        t_decode0 = time.monotonic()
        for _ in range(n_decode - 1):
            logits = engine.forward(mx.array([[tok]]), kv=kv)
            t0 = time.monotonic()
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            head_s += time.monotonic() - t0
        decode_s = time.monotonic() - t_decode0
        s1 = engine.cache.stats()
        stall_s = engine._stall_seconds - stall0

        n = n_decode - 1
        misses = {k: s1["misses"][k] - s0["misses"][k] for k in s1["misses"]}
        accounted = (BUCKETS["router_sync_s"] + BUCKETS["scatter_wall_s"]
                     + BUCKETS["layer_sync_s"] + head_s)
        out = {
            "tag": tag,
            "budget_gb": budget / (1 << 30),
            "decode_tokens_timed": n,
            "decode_s": round(decode_s, 2),
            "decode_tok_s": round(n / decode_s, 3),
            "ms_per_token": {
                "total": round(1000 * decode_s / n, 1),
                "stall_demand_load": round(1000 * stall_s / n, 1),
                "scatter_minus_stall": round(
                    1000 * (BUCKETS["scatter_wall_s"] - stall_s) / n, 1),
                "router_sync": round(1000 * BUCKETS["router_sync_s"] / n, 1),
                "layer_sync": round(1000 * BUCKETS["layer_sync_s"] / n, 1),
                "head_sample": round(1000 * head_s / n, 1),
                "other_python": round(
                    1000 * (decode_s - accounted) / n, 1),
            },
            "expert_misses_per_token": round(misses["expert"] / n, 2),
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        print(json.dumps(out, indent=2))
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()


if __name__ == "__main__":
    main()
