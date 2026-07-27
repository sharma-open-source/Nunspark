"""Decode time attribution on the WIRED baseline (backlog #14 follow-up).

Why this exists: the last attribution probe (decode_time_attribution_probe.py,
2026-07-17) predates BOTH the #10 scatter fix AND the #14 wiring win. It found
scatter-rebuild = 59-70% and expert-stall = 17-30% of decode. Both of those
buckets have since been attacked:

  - #10 persistent scatter buffers killed the full-buffer rebuild (scatter_wall
    minus stall should now be small).
  - #14 set_wired_limit killed the compressor fight; the only post-wiring number
    on record is the #14.5 note "expert stall is only ~14% of decode".

So on the shipped wired 10 GB baseline (~8 tok/s), roughly ~86% of a decode
token is currently UNATTRIBUTED. Every algorithmic decode lever (deep-K spec,
router-lookahead, startup/decode warming) is measured dead on this hardware, so
the next win — if there is one on 16 GB — is hiding in that gap, exactly as the
scatter and wiring wins were. This probe finds the new dominant bucket.

Buckets (each is "where the HOST blocks", not "what the GPU computes" — a lazy
sync drains everything enqueued up to it; that is the orchestration-vs-compute
split the question needs):

  stall_demand_load    demand-load wait inside _scatter_experts (engine counter)
  scatter_minus_stall  _scatter_experts wall MINUS stall (slot update / setitem
                       scatter / force-read) — the #10 fix should have gutted this
  router_sync          per-layer `inds.reshape(-1).tolist()` host sync — drains
                       the lazy graph up to and incl. this layer's attention+router
  layer_sync           _sync_layer's mx.eval — drains this layer's expert mix
  head_sample          final norm/head logits + argmax .item() per token
  other_python         wall - accounted (python orchestration, cache lookups,
                       prefetch enqueue, embed, lookahead issue, ...)

Validity: vm_stat is sampled around the decode segment. The #15 ambient-
compressor-churn sightings show some runs collapse to ~3 tok/s with 2-4x the
compression on byte-identical cache work. Clean wired runs compress ~15-20 GB;
if compressed_gb_during_run is much higher, the run is CONTAMINATED — re-run it
before trusting the attribution (the buckets will be smeared by swap).

Usage (on the Mac under test — wired is the shipped default, so this measures
the real baseline; close heavy apps, run nothing else disk-heavy alongside):

  uv run python scripts/decode_time_attribution_wired.py ./packed/qwen3-30b
  uv run python scripts/decode_time_attribution_wired.py ./packed/qwen3-30b \
      --budget 10 --tokens 200 --flush-file ./probe_flush.bin

  --budget 10       GB (default 10 = the measured 16 GB wired optimum)
  --tokens 200      greedy decode tokens (first is prefill's, timing covers N-1)
  --no-wire         measure the UNWIRED control instead (for an A vs #14 delta)
  --flush-file P    stream this file first to evict the model's pages (cold-disk
                    start); omit to attribute a warm-start steady state
  --out P           default scripts/results/decode_attribution_wired.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import tempfile
import time
import types
from pathlib import Path

import mlx.core as mx

from nunspark.bench import _encode, _load_tokenizer
from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest


# House-style memory instrumentation, copied verbatim from wired_limit_probe.py
# (scripts/ is not a package, so each probe keeps its own copy — same measured
# helpers, same schema).
def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def _vm_stat() -> dict:
    """Parse `vm_stat` into {metric: pages}, plus page_size_bytes."""
    raw = _sh(["vm_stat"])
    out: dict = {}
    m = re.search(r"page size of (\d+) bytes", raw)
    out["page_size_bytes"] = int(m.group(1)) if m else 16384
    for line in raw.splitlines():
        m = re.match(r'^"?([A-Za-z -]+)"?:\s+([\d.]+)\.?$', line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            out[key] = int(float(m.group(2)))
    return out


def _vm_delta(before: dict, after: dict) -> dict:
    """Deltas of the counters that evidence compressor/swap activity."""
    page = after.get("page_size_bytes", 16384)
    keys = ("pages_occupied_by_compressor", "compressions", "decompressions",
            "swapins", "swapouts", "pageins", "pageouts")
    d = {}
    for k in keys:
        if k in before and k in after:
            d[k] = after[k] - before[k]
    if "compressions" in d:
        d["compressed_gb_during_run"] = round(d["compressions"] * page / (1 << 30), 2)
    if "pages_occupied_by_compressor" in d:
        d["compressor_pages_net_gb"] = round(
            d["pages_occupied_by_compressor"] * page / (1 << 30), 2)
    return d


def _flush_page_cache(path: Path) -> None:
    """Evict the model's pages by streaming a dummy file through the page cache
    (house method — no sudo/purge needed)."""
    t0 = time.monotonic()
    with open(path, "rb") as f:
        while f.read(256 << 20):
            pass
    print(f"  [flush {time.monotonic() - t0:.0f}s]", flush=True)
    time.sleep(2)


PROMPT =("Explain, step by step, how a modern operating system schedules "
          "threads across performance and efficiency cores, and what a "
          "userspace developer can do to cooperate with the scheduler.")

BUCKETS = {"router_sync_s": 0.0, "scatter_wall_s": 0.0, "layer_sync_s": 0.0}


def _timed_moe_attn_and_mix(self, slot, layer, h, mask, cache):
    """Verbatim replica of StreamingEngine._moe_attn_and_mix (engine.py, kept in
    sync with lines 662-722) with two host-sync timers inserted. Keep this in
    lockstep with the engine: the shared-experts add, the fp32 mix cast, and the
    single-token lookahead issue are all part of the real decode cost and must
    stay here or the attribution lies."""
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
    mix = (y * scores[..., None]).sum(axis=-2)
    if self._moe_mix_cast:
        mix = mix.astype(y.dtype)
    if self._shared_experts_attr is not None:
        mix = mix + getattr(slot.mlp, self._shared_experts_attr)(x)
    out = h + mix
    if self._lookahead_prefetch and batch_tokens == 1:
        self._issue_lookahead(layer, out)
    return out


def _timed_sync_layer(self, h):
    if not self._fully_resident:
        t0 = time.monotonic()
        mx.eval(h)
        BUCKETS["layer_sync_s"] += time.monotonic() - t0
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_root", type=Path)
    ap.add_argument("--budget", type=float, default=10.0,
                    help="GB (default 10 = measured 16 GB wired optimum)")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--no-wire", action="store_true",
                    help="measure the UNWIRED control instead of the shipped default")
    ap.add_argument("--flush-file", type=Path, default=None,
                    help="stream this file first to evict the model's pages (cold start)")
    ap.add_argument("--out", type=Path,
                    default=Path("scripts/results/decode_attribution_wired.json"))
    args = ap.parse_args()

    root = args.packed_root
    budget = int(args.budget * (1 << 30))
    n_decode = args.tokens
    wired = not args.no_wire

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    if args.flush_file is not None:
        print(f"flushing page cache via {args.flush_file} ...", flush=True)
        _flush_page_cache(args.flush_file)

    engine = StreamingEngine(root, manifest, budget_bytes=budget, wire_limit=wired)
    engine._moe_attn_and_mix = types.MethodType(_timed_moe_attn_and_mix, engine)
    engine._sync_layer = types.MethodType(_timed_sync_layer, engine)
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        logits = _prefill(engine, ids, kv, 1024)
        tok = int(mx.argmax(logits, axis=-1).item())

        # Reset all counters AFTER prefill so we attribute the decode segment only.
        for k in BUCKETS:
            BUCKETS[k] = 0.0
        s0 = engine.cache.stats()
        stall0 = engine._stall_seconds
        head_s = 0.0
        vm0 = _vm_stat()
        t_decode0 = time.monotonic()
        for _ in range(n_decode - 1):
            logits = engine.forward(mx.array([[tok]]), kv=kv)
            t0 = time.monotonic()
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            head_s += time.monotonic() - t0
        decode_s = time.monotonic() - t_decode0
        vm1 = _vm_stat()
        s1 = engine.cache.stats()
        stall_s = engine._stall_seconds - stall0

        n = n_decode - 1
        misses = {k: s1["misses"][k] - s0["misses"][k] for k in s1["misses"]}
        scatter_minus_stall_s = BUCKETS["scatter_wall_s"] - stall_s
        accounted = (stall_s + scatter_minus_stall_s + BUCKETS["router_sync_s"]
                     + BUCKETS["layer_sync_s"] + head_s)
        vm = _vm_delta(vm0, vm1)
        compressed_gb = vm.get("compressed_gb_during_run", 0.0)

        buckets_ms = {
            "stall_demand_load": round(1000 * stall_s / n, 1),
            "scatter_minus_stall": round(1000 * scatter_minus_stall_s / n, 1),
            "router_sync": round(1000 * BUCKETS["router_sync_s"] / n, 1),
            "layer_sync": round(1000 * BUCKETS["layer_sync_s"] / n, 1),
            "head_sample": round(1000 * head_s / n, 1),
            "other_python": round(1000 * (decode_s - accounted) / n, 1),
        }
        total_ms = 1000 * decode_s / n
        buckets_pct = {k: round(100 * v / total_ms, 1) for k, v in buckets_ms.items()}
        dominant = max(buckets_ms, key=buckets_ms.get)
        # Clean wired runs compress ~15-20 GB (backlog #15); flag contamination.
        contaminated = wired and compressed_gb > 30.0

        out = {
            "tag": "decode_attribution_wired" if wired else "decode_attribution_unwired",
            "date": datetime.date.today().isoformat(),
            "wired": wired,
            "wired_limit_gb": (round(engine.wired_limit_bytes / (1 << 30), 2)
                               if engine.wired_limit_bytes else None),
            "budget_gb": budget / (1 << 30),
            "cold_start": args.flush_file is not None,
            "decode_tokens_timed": n,
            "decode_s": round(decode_s, 2),
            "decode_tok_s": round(n / decode_s, 3),
            "ms_per_token_total": round(total_ms, 1),
            "buckets_ms_per_token": buckets_ms,
            "buckets_pct_of_decode": buckets_pct,
            "dominant_bucket": dominant,
            "expert_misses_per_token": round(misses["expert"] / n, 2),
            "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
            "vm_delta": vm,
            "compressed_gb_during_run": compressed_gb,
            "contaminated": contaminated,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))

        print(json.dumps(out, indent=2))
        print(f"\n  decode: {out['decode_tok_s']} tok/s "
              f"({out['ms_per_token_total']} ms/token), wired={wired}")
        print("  attribution (share of decode):")
        for k, v in sorted(buckets_ms.items(), key=lambda kv: -kv[1]):
            print(f"    {k:22s} {v:7.1f} ms/tok  {buckets_pct[k]:5.1f}%"
                  f"{'   <-- dominant' if k == dominant else ''}")
        if contaminated:
            print(f"\n  !! CONTAMINATED: {compressed_gb} GB compressed this run "
                  f"(clean wired ~15-20). Re-run before trusting the buckets (#15).")
        else:
            print(f"\n  compressor: {compressed_gb} GB compressed "
                  f"(clean wired baseline). Buckets are trustworthy.")
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()


if __name__ == "__main__":
    main()
