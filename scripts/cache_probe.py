"""Probe: does increasing the PieceCache budget reduce disk reads?

Simulates the StreamingEngine access pattern WITHOUT a model: every token is a
full cyclic scan of layers 0..N-1. We count real disk loads (loader calls) and
the cache's own hits/misses across a range of budgets, with prefetch off so the
numbers reflect the eviction policy alone (not prefetch's miss accounting).
"""
from __future__ import annotations

import mlx.core as mx
from nunspark.piece_cache import PieceCache

N_LAYERS = 24
TOKENS = 6
LAYER_BYTES = 1  # we fake size via a 1-element array; budget is in "layers" below


def run(budget_layers: int) -> tuple[int, int, int]:
    loads: list[str] = []

    def loader(pid: str) -> dict:
        loads.append(pid)
        # one float32 scalar == 4 bytes; we scale the budget in units of 4 bytes
        return {"w": mx.zeros((1,), dtype=mx.float32)}

    one = 4  # bytes per piece
    cache = PieceCache(loader, budget_bytes=one * budget_layers)
    for _ in range(TOKENS):
        for layer in range(N_LAYERS):
            cache.get(f"layer_{layer:03d}")   # no prefetch -> pure get/LRU path
    cache.close()
    return len(loads), cache.hits, cache.misses


def main() -> None:
    total_accesses = N_LAYERS * TOKENS
    print(f"model = {N_LAYERS} layers, {TOKENS} tokens, "
          f"{total_accesses} layer-accesses total")
    print(f"{'budget (layers)':>16} | {'disk loads':>10} | {'hits':>6} | "
          f"{'misses':>6} | {'hit rate':>8}")
    print("-" * 60)
    for budget_layers in [1, 4, 8, 12, 18, 23, 24, 26]:
        loads, hits, misses = run(budget_layers)
        rate = hits / total_accesses
        note = "  <- whole model fits" if budget_layers >= N_LAYERS else ""
        print(f"{budget_layers:>16} | {loads:>10} | {hits:>6} | "
              f"{misses:>6} | {rate:>7.0%}{note}")


if __name__ == "__main__":
    main()
