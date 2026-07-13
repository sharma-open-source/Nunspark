"""Compare eviction policies on the engine's cyclic layer-scan WITH prefetch.

Pure-Python simulation (no MLX). Every token scans layers 0..N-1 in order.
The engine prefetches layer+1 while computing layer, so we model each step as:
  get(layer); prefetch(layer+1)
Both get-misses and prefetch-loads can evict. We count disk reads (loads) per
policy across budgets to choose the most robust eviction variant.

Policies:
  lru            : evict least-recently-used   (current code -- the bug)
  mru_touch      : evict most-recently-used, move_to_end on hit (true LRU order)
  mru_notouch    : evict most-recently-INSERTED, no move_to_end (stable prefix)
"""
from __future__ import annotations

from collections import OrderedDict

N = 24
TOKENS = 8
PREFETCH = True


def simulate(budget: int, policy: str) -> int:
    resident: "OrderedDict[int, None]" = OrderedDict()
    reads = 0

    def evict(protect: int) -> None:
        order = list(resident) if policy == "lru" else list(reversed(list(resident)))
        for v in order:
            if len(resident) <= budget:
                break
            if v == protect:
                continue
            resident.pop(v)

    def load(layer: int, protect_against: int) -> None:
        nonlocal reads
        if layer in resident:
            return
        reads += 1
        resident[layer] = None              # inserted at MRU/newest end
        evict(protect=protect_against)

    for _ in range(TOKENS):
        for layer in range(N):
            # get(layer)
            if layer in resident:
                if policy == "mru_touch":
                    resident.move_to_end(layer)
            else:
                load(layer, protect_against=layer)
            # prefetch(layer+1): worker loads it, protecting the prefetched id
            if PREFETCH and layer + 1 < N:
                load(layer + 1, protect_against=layer + 1)
    return reads


def main() -> None:
    print(f"{N} layers, {TOKENS} tokens, prefetch={PREFETCH}")
    print(f"ideal lower bound ~= N + (TOKENS-1)*(N-budget)\n")
    print(f"{'budget':>6} | {'lru':>6} | {'mru_touch':>10} | "
          f"{'mru_notouch':>12} | {'ideal':>6}")
    print("-" * 56)
    for budget in [4, 8, 12, 16, 20, 24]:
        lru = simulate(budget, "lru")
        mt = simulate(budget, "mru_touch")
        nt = simulate(budget, "mru_notouch")
        ideal = N + (TOKENS - 1) * max(N - budget, 0)
        print(f"{budget:>6} | {lru:>6} | {mt:>10} | {nt:>12} | {ideal:>6}")


if __name__ == "__main__":
    main()
