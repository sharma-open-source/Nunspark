"""Probe real KVStore behavior under a realistic growing cyclic decode scan.

Question: does MRU eviction help KVStore the way it helps PieceCache, or do
KVStore's differences (evicts on every hit; KVCache pre-allocates buffers in
256-token steps; offloaded caches reload trimmed) change the picture?

We simulate decode: each step, for every layer, get(layer) then append ONE token
(update_and_fetch) -- exactly what the engine does. KV grows over steps. We count
disk reloads (misses beyond the seeding baseline) under the current LRU code and
under a monkeypatched MRU variant.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from nunspark import kv_store
from nunspark.kv_store import KVStore

N_KV_HEADS, HEAD_DIM = 8, 128          # realistic-ish per-layer KV width
N_LAYERS = 8
STEPS = 20                              # decode steps (KV grows each step)


def _one_token():
    k = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM))
    v = mx.random.normal((1, N_KV_HEADS, 1, HEAD_DIM))
    return k, v


def _per_layer_nbytes_after(steps: int) -> int:
    c = KVCache()
    for _ in range(steps):
        c.update_and_fetch(*_one_token())
    mx.eval(c.keys, c.values)
    return int(c.nbytes)


def run(budget_layers: float, mru: bool) -> tuple[int, int]:
    """Return (scan_reloads, peak_resident_layers) for a growing cyclic scan."""
    # Monkeypatch eviction direction.
    orig = KVStore._evict_locked
    if mru:
        def mru_evict(self, protect):
            total = sum(int(c.nbytes) for c in self._resident.values())
            self.peak_bytes = max(self.peak_bytes, total)
            evicted = False
            for layer in reversed(list(self._resident)):
                if total <= self._budget:
                    break
                if layer == protect:
                    continue
                cache = self._resident[layer]
                nb = int(cache.nbytes)
                if nb == 0:
                    self._resident.pop(layer)
                    continue
                keys, values = cache.state
                mx.save_safetensors(str(self._file(layer)), {"keys": keys, "values": values})
                self._resident.pop(layer)
                self._offloaded.add(layer)
                total -= nb
                evicted = True
            if evicted:
                mx.clear_cache()
        KVStore._evict_locked = mru_evict
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # size the budget against a mid-run per-layer cache size
            nb_mid = _per_layer_nbytes_after(STEPS // 2)
            store = KVStore(d, budget_bytes=int(nb_mid * budget_layers), prefetch=False)
            for step in range(STEPS):
                for layer in range(N_LAYERS):
                    c = store.get(layer)
                    c.update_and_fetch(*_one_token())   # decode appends one token
                    mx.eval(c.keys, c.values)
            reloads = store.misses
            peak = store.peak_bytes
            store.close()
            return reloads, int(peak / max(nb_mid, 1))
    finally:
        KVStore._evict_locked = orig


def main() -> None:
    print(f"{N_LAYERS} layers, {STEPS} decode steps (KV grows each step), "
          f"budget sized in mid-run layer-equivalents\n")
    print(f"{'budget(layers)':>14} | {'LRU reloads':>12} | {'MRU reloads':>12}")
    print("-" * 46)
    print(f"(N*STEPS = {N_LAYERS * STEPS} == full thrash)")
    for b in [2, 6]:
        lru, _ = run(b, mru=False)
        mru, _ = run(b, mru=True)
        print(f"{b:>14} | {lru:>12} | {mru:>12}")


if __name__ == "__main__":
    main()
