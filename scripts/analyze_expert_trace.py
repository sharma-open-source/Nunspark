"""Analyze an expert-trace JSONL (StreamingEngine's `expert_trace` kwarg) to
answer M1's locality questions before any cache/prefetch engine code is written.

Input: one JSON object per line, `{"t": int, "layer": int, "fired": [int, ...],
"batch_tokens": int}` (see StreamingEngine._trace_moe / engine.py's
_moe_layer_forward). `t` is a global, strictly increasing call counter -- records
are processed in that order per layer.

Computes, per layer and aggregate:
  - expert firing frequency distribution + skew (top-10%-of-experts' share of
    all firings -- high share = a small hot set worth caching/pinning),
  - mean token-to-token Jaccard overlap of consecutive fired sets (high overlap
    => temporal prefetch (M3a) captures most of the win),
  - reuse-distance histogram (calls since an expert last fired, per layer),
  - offline cache simulation: expert-piece hit rate vs cache capacity (in
    experts, swept 8..num_experts) for LRU and LFU-with-decay, using
    --expert-bytes piece sizes if given (else uniform unit weight),
  - fired-union size vs sliding-window length K in {1,4,8,16,25,32,64} (mean
    union of fired sets over K consecutive calls per layer -- decides whether
    tree/window verification (G3/M4) is worth building selective support for).

Usage:
  uv run python scripts/analyze_expert_trace.py trace.jsonl --json out.json
  uv run python scripts/analyze_expert_trace.py trace.jsonl --expert-bytes sizes.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

WINDOW_KS = (1, 4, 8, 16, 25, 32, 64)
REUSE_BUCKET_EDGES = (1, 2, 4, 8, 16, 32, 64, 128)


def load_trace(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["t"])
    return records


def group_by_layer(records: list[dict]) -> dict[int, list[dict]]:
    layers: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        layers[r["layer"]].append(r)
    for recs in layers.values():
        recs.sort(key=lambda r: r["t"])
    return dict(sorted(layers.items()))


def firing_frequency(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for r in records:
        counts.update(r["fired"])
    return counts


def skew_summary(counts: Counter, num_experts: int) -> dict:
    total = sum(counts.values())
    if total == 0 or num_experts == 0:
        return {"top10pct_experts": 0, "top10pct_share": 0.0}
    sizes = sorted(counts.values(), reverse=True)
    sizes += [0] * max(0, num_experts - len(sizes))
    top_n = max(1, num_experts // 10)
    return {
        "top10pct_experts": top_n,
        "top10pct_share": sum(sizes[:top_n]) / total,
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def mean_consecutive_jaccard(records: list[dict]) -> float | None:
    if len(records) < 2:
        return None
    vals = [
        jaccard(set(records[i - 1]["fired"]), set(records[i]["fired"]))
        for i in range(1, len(records))
    ]
    return sum(vals) / len(vals)


def _reuse_bucket(distance: int) -> str:
    for edge in REUSE_BUCKET_EDGES:
        if distance <= edge:
            return f"<={edge}"
    return f">{REUSE_BUCKET_EDGES[-1]}"


def reuse_distance_histogram(records: list[dict]) -> dict[str, int]:
    """Calls-since-last-fired for each expert, within this layer's call order."""
    last_seen: dict[int, int] = {}
    hist: Counter = Counter()
    for i, r in enumerate(records):
        for e in r["fired"]:
            if e in last_seen:
                hist[_reuse_bucket(i - last_seen[e])] += 1
            last_seen[e] = i
    return dict(hist)


def union_size_by_window(records: list[dict], ks=WINDOW_KS) -> dict[int, float]:
    fired_sets = [set(r["fired"]) for r in records]
    out: dict[int, float] = {}
    for k in ks:
        if not fired_sets:
            out[k] = 0.0
            continue
        if len(fired_sets) < k:
            out[k] = float(len(set().union(*fired_sets)))
            continue
        sizes = []
        for i in range(len(fired_sets) - k + 1):
            u: set = set()
            for j in range(i, i + k):
                u |= fired_sets[j]
            sizes.append(len(u))
        out[k] = sum(sizes) / len(sizes)
    return out


def _load_expert_bytes(path: str | None) -> dict:
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def _sizeof(expert_bytes: dict, layer: int, expert: int, default: float) -> float:
    for key in (f"{layer}:{expert}", str(expert)):
        if key in expert_bytes:
            return float(expert_bytes[key])
    return default


def _capacities(max_experts: int, min_c: int = 8, num_points: int = 6) -> list[int]:
    if max_experts <= min_c:
        return [max_experts]
    step = max(1, (max_experts - min_c) // (num_points - 1))
    caps = sorted(set(range(min_c, max_experts, step)) | {max_experts})
    return caps


def simulate_lru(events: list[tuple[int, int]], capacity_bytes: float,
                 sizeof) -> float:
    resident: "OrderedDict[tuple[int, int], float]" = OrderedDict()
    used = 0.0
    hits = misses = 0
    for key in events:
        if key in resident:
            resident.move_to_end(key)
            hits += 1
            continue
        misses += 1
        sz = sizeof(key)
        resident[key] = sz
        used += sz
        while used > capacity_bytes and resident:
            _, v = resident.popitem(last=False)
            used -= v
    return hits / (hits + misses) if (hits + misses) else 0.0


def simulate_lfu_decay(events: list[tuple[int, int]], capacity_bytes: float,
                       sizeof, decay: float = 0.98) -> float:
    resident: dict[tuple[int, int], float] = {}
    freq: dict[tuple[int, int], float] = {}
    last_t: dict[tuple[int, int], int] = {}
    used = 0.0
    hits = misses = 0

    def decayed(key: tuple[int, int], t: int) -> float:
        return freq.get(key, 0.0) * (decay ** (t - last_t.get(key, t)))

    for t, key in enumerate(events):
        freq[key] = decayed(key, t) + 1.0
        last_t[key] = t
        if key in resident:
            hits += 1
            continue
        misses += 1
        sz = sizeof(key)
        resident[key] = sz
        used += sz
        while used > capacity_bytes and resident:
            evict_key = min(resident, key=lambda k: decayed(k, t))
            used -= resident.pop(evict_key)
            freq.pop(evict_key, None)
            last_t.pop(evict_key, None)
    return hits / (hits + misses) if (hits + misses) else 0.0


def cache_sweep(records: list[dict], num_experts: int, expert_bytes: dict,
                capacities: list[int] | None) -> dict:
    events = [(r["layer"], e) for r in records for e in r["fired"]]
    keys = sorted(set(events))
    if not keys:
        return {"capacities": [], "lru_hit_rate": [], "lfu_decay_hit_rate": []}
    have_bytes = bool(expert_bytes)
    default_size = 1.0
    if have_bytes:
        sizes = [_sizeof(expert_bytes, l, e, 1.0) for l, e in keys]
        default_size = sum(sizes) / len(sizes)

    def sizeof(key: tuple[int, int]) -> float:
        l, e = key
        return _sizeof(expert_bytes, l, e, default_size) if have_bytes else 1.0

    caps = capacities or _capacities(len(keys))
    lru_rates, lfu_rates = [], []
    for cap in caps:
        capacity_bytes = cap * default_size if have_bytes else float(cap)
        lru_rates.append(simulate_lru(events, capacity_bytes, sizeof))
        lfu_rates.append(simulate_lfu_decay(events, capacity_bytes, sizeof))
    return {
        "capacities": caps,
        "lru_hit_rate": lru_rates,
        "lfu_decay_hit_rate": lfu_rates,
        "bytes_mode": "per-expert" if have_bytes else "count",
    }


def analyze(records: list[dict], expert_bytes: dict,
           capacities: list[int] | None) -> dict:
    layers = group_by_layer(records)
    num_experts = max((e for r in records for e in r["fired"]), default=-1) + 1

    per_layer = {}
    for layer, recs in layers.items():
        counts = firing_frequency(recs)
        per_layer[layer] = {
            "num_calls": len(recs),
            "firing_frequency": dict(counts),
            "skew": skew_summary(counts, num_experts),
            "mean_consecutive_jaccard": mean_consecutive_jaccard(recs),
            "reuse_distance_histogram": reuse_distance_histogram(recs),
            "union_by_window": union_size_by_window(recs),
        }

    agg_counts = firing_frequency(records)
    agg_jaccards = [
        v for l in layers.values()
        if (v := mean_consecutive_jaccard(l)) is not None
    ]
    agg_reuse: Counter = Counter()
    for stats in per_layer.values():
        agg_reuse.update(stats["reuse_distance_histogram"])
    agg_union: dict[int, list[float]] = defaultdict(list)
    for stats in per_layer.values():
        for k, v in stats["union_by_window"].items():
            agg_union[k].append(v)

    aggregate = {
        "num_records": len(records),
        "num_layers": len(layers),
        "num_experts": num_experts,
        "firing_frequency": dict(agg_counts),
        "skew": skew_summary(agg_counts, num_experts),
        "mean_consecutive_jaccard": (
            sum(agg_jaccards) / len(agg_jaccards) if agg_jaccards else None
        ),
        "reuse_distance_histogram": dict(agg_reuse),
        "union_by_window": {
            k: sum(vs) / len(vs) for k, vs in sorted(agg_union.items())
        },
        "cache_sweep": cache_sweep(records, num_experts, expert_bytes, capacities),
    }
    return {"per_layer": per_layer, "aggregate": aggregate}


def _print_summary(results: dict) -> None:
    agg = results["aggregate"]
    print(f"records={agg['num_records']}  layers={agg['num_layers']}  "
          f"experts={agg['num_experts']}")
    print()
    print("-- aggregate --")
    skew = agg["skew"]
    print(f"top-10% experts ({skew['top10pct_experts']}) share of firings: "
          f"{skew['top10pct_share']:.1%}")
    mj = agg["mean_consecutive_jaccard"]
    print(f"mean consecutive-call Jaccard overlap: "
          f"{mj:.3f}" if mj is not None else "mean consecutive-call Jaccard overlap: n/a")
    print("reuse-distance histogram (calls since expert last fired):")
    for bucket in [f"<={e}" for e in REUSE_BUCKET_EDGES] + [f">{REUSE_BUCKET_EDGES[-1]}"]:
        n = agg["reuse_distance_histogram"].get(bucket, 0)
        if n:
            print(f"  {bucket:>6}: {n}")
    print("fired-union size vs window K (mean over sliding windows, per-layer avg):")
    for k, v in sorted(agg["union_by_window"].items()):
        print(f"  K={k:<3}: {v:.2f} experts")
    sweep = agg["cache_sweep"]
    if sweep["capacities"]:
        print(f"offline cache sim (mode={sweep['bytes_mode']}):")
        print(f"  {'capacity':>9} | {'LRU hit%':>9} | {'LFU-decay hit%':>15}")
        for cap, lru, lfu in zip(sweep["capacities"], sweep["lru_hit_rate"],
                                 sweep["lfu_decay_hit_rate"]):
            print(f"  {cap:>9} | {lru:>8.1%} | {lfu:>14.1%}")
    print()
    print("-- per layer --")
    for layer, stats in sorted(results["per_layer"].items()):
        mj = stats["mean_consecutive_jaccard"]
        mj_str = f"{mj:.3f}" if mj is not None else "n/a"
        print(f"  layer {layer:>3}: calls={stats['num_calls']:<6} "
              f"top10%_share={stats['skew']['top10pct_share']:.1%}  "
              f"consec_jaccard={mj_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to an expert-trace JSONL file")
    ap.add_argument("--json", default=None, help="write full results as JSON to this path")
    ap.add_argument("--expert-bytes", default=None,
                    help="JSON file mapping 'layer:expert' or 'expert' -> piece byte size; "
                         "default is uniform (count-based) cache simulation")
    ap.add_argument("--capacities", default=None,
                    help="comma-separated cache capacities (in experts) to sweep "
                         "(default: ~6 points from 8 to num_experts)")
    args = ap.parse_args()

    records = load_trace(args.trace)
    if not records:
        print("empty trace")
        return 1

    expert_bytes = _load_expert_bytes(args.expert_bytes)
    capacities = (
        [int(x) for x in args.capacities.split(",")] if args.capacities else None
    )

    results = analyze(records, expert_bytes, capacities)
    _print_summary(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
