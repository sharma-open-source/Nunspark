"""Backlog #13 (C2) offline go/no-go: does a pinnable expert hot set exist?

Colibrì persists per-workload expert-usage counts and pins the hottest experts
across sessions. Before building anything online, measure — from the existing
M1 decode traces (Qwen3-30B, 128 experts/layer, top-8) — whether the fired
distribution is concentrated enough for frequency pinning to beat what plain
LRU already achieves, and whether a hot set learned on one workload transfers
to another (the cross-session claim).

For each workload's greedy trace (decode passes only, batch_tokens == 1):
  1. Count fires per (layer, expert).
  2. Coverage curve: fraction of all decode expert-loads that land inside the
     top-N (layer, expert) pairs by frequency, for N as a fraction of the
     total expert population. Uniform baseline = that same fraction.
  3. Transfer: rank pairs on workload A, score coverage on workload B.

Pure JSON analysis — no model, no timing, page cache irrelevant.

Usage: python expert_hotset_offline_probe.py [traces_dir]
"""
import json
import sys
from collections import Counter
from pathlib import Path

WORKLOADS = ("prose", "code", "reasoning")
# Pin-set sizes as a fraction of the full (layer, expert) population; on 30B
# (48 layers x 128 experts, ~34 MB/expert-piece) 0.05 ~= 10 GB of pins, so
# only the small fractions are realistic on 16 GB machines.
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)


def decode_counts(path: Path) -> tuple[Counter, int]:
    """Fires per (layer, expert) over single-token decode passes."""
    counts: Counter = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["batch_tokens"] != 1:
                continue
            for e in rec["fired"]:
                counts[(rec["layer"], e)] += 1
                total += 1
    return counts, total


def coverage(rank_source: Counter, eval_counts: Counter, eval_total: int,
             top_n: int) -> float:
    hot = {p for p, _ in rank_source.most_common(top_n)}
    return sum(c for p, c in eval_counts.items() if p in hot) / eval_total


def main() -> None:
    traces = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/results/m1_traces")
    data = {}
    pop = set()
    for w in WORKLOADS:
        counts, total = decode_counts(traces / f"{w}-greedy.jsonl")
        data[w] = (counts, total)
        pop |= set(counts)
    n_pop = len(pop)

    out = {
        "probe": "offline expert hot-set concentration + transfer (backlog #13 C2)",
        "traces": str(traces),
        "population_layer_expert_pairs": n_pop,
        "decode_loads_per_workload": {w: t for w, (_c, t) in data.items()},
        "self_coverage": {},    # ranked and scored on the same workload (ceiling)
        "transfer_coverage": {},  # ranked on A, scored on B (the pinning claim)
    }
    for frac in FRACTIONS:
        top_n = max(1, int(n_pop * frac))
        key = f"pin_top_{frac:.0%}"
        out["self_coverage"][key] = {
            w: round(coverage(c, c, t, top_n), 4) for w, (c, t) in data.items()
        }
        out["transfer_coverage"][key] = {
            f"{a}->{b}": round(coverage(data[a][0], *data[b], top_n), 4)
            for a in WORKLOADS for b in WORKLOADS if a != b
        }
        # uniform baseline == frac by construction; a hot set only matters if
        # coverage >> frac.

    print(json.dumps(out, indent=2))
    with open("scripts/results/expert_hotset_offline.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
