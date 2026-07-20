from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import mlx.core as mx
from mlx_lm.models.cache import CacheList, KVCache, QuantizedKVCache

from .archspec import KVQuant, make_cache


def _cache_nbytes(cache) -> int:
    """cache.nbytes, but 0 for a cache that has no buffers yet —
    QuantizedKVCache.nbytes raises AttributeError before the first append.
    A CacheList (Paired kind) reports the sum of its children."""
    if isinstance(cache, CacheList):
        return sum(_cache_nbytes(c) for c in cache.caches)
    if getattr(cache, "keys", None) is None:
        return 0
    return int(cache.nbytes)


class KVStore:
    """Byte-budget store of per-layer KV caches with disk offload + prefetch.

    Mirrors PieceCache, but the cached objects are mutable KVCache instances
    that grow one token per decode step. Layers resident up to budget_bytes
    stay live in memory (zero disk cost); overflow layers are serialized to
    per-layer safetensors scratch files (via KVCache.state) and reloaded on
    next access. A background worker reloads offloaded layers ahead of use so
    the disk read overlaps GPU compute.

    Because KV caches grow in place, the resident byte total is RECOMPUTED on
    each eviction decision (never tracked incrementally). Eviction serializes
    synchronously; async write-back is a deferred optimization (Plan 4b).
    """

    def __init__(self, scratch_dir, budget_bytes: int, prefetch: bool = True,
                 cache_kinds: list | None = None,
                 kv_quant: KVQuant | None = None):
        self._dir = Path(scratch_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._budget = int(budget_bytes)
        self._cache_kinds = cache_kinds
        self._kv_quant = kv_quant
        self._prefetch_enabled = prefetch
        self._resident: "OrderedDict[int, KVCache]" = OrderedDict()
        self._offloaded: set[int] = set()      # layer -> current state lives on disk
        self._inflight: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self._queue: "queue.Queue[int | None]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---- public API ----
    def get(self, layer: int) -> KVCache:
        with self._lock:
            if layer in self._resident:
                self._resident.move_to_end(layer)
                self.hits += 1
                cache = self._resident[layer]
                # Enforce the budget on EVERY main-thread access, not just misses:
                # the worker inserts prefetched layers without evicting (it must not
                # call mx.save_safetensors off the main thread), so a run of hits
                # would otherwise let residency grow past the budget unbounded.
                # NOTE: _evict_locked may call mx.save_safetensors while holding
                # _lock (prefetch worker can't evict from its thread). This is
                # correct but means the hit path can block on disk I/O under
                # eviction pressure.
                self._evict_locked(protect=layer)
                return cache
            ev = self._inflight.get(layer)
            if ev is None:
                ev = threading.Event()
                self._inflight[layer] = ev   # reserve under lock; we own the load
                self.misses += 1
                owner = True
            else:
                owner = False
        if not owner:
            ev.wait()
            with self._lock:
                if layer in self._resident:
                    self._resident.move_to_end(layer)
                    self.hits += 1
                    cache = self._resident[layer]
                    self._evict_locked(protect=layer)   # enforce budget; may do disk I/O (see above)
                    return cache
            return self.get(layer)           # owner failed + cleaned up; retry
        return self._materialize(layer, ev)

    def prefetch(self, layers: Iterable[int]) -> None:
        if not self._prefetch_enabled:
            return
        with self._lock:
            for layer in layers:
                # Only offloaded layers need a real disk reload; a never-offloaded
                # layer is created fresh+empty instantly by get(), nothing to overlap.
                if layer not in self._offloaded:
                    continue
                if layer in self._resident or layer in self._inflight:
                    continue
                self._inflight[layer] = threading.Event()
                self.misses += 1
                self._queue.put(layer)

    def truncate(self, n: int) -> None:
        """Drop the last `n` appended tokens from every layer's KV cache.

        Rolls back rejected speculative tokens. Offloaded layers are reloaded so
        their on-disk (un-trimmed) state can't resurrect dropped tokens, then
        trimmed in place. KVCache.trim only decrements offset; the pre-allocated
        key/value buffer is reused on the next append, so peak_bytes won't drop
        instantly (cosmetic, not a leak). Rare on the common path: KV is tiny and
        the default budget keeps every layer resident.
        """
        if n <= 0:
            return
        with self._lock:
            for layer in list(self._offloaded):
                cache = self._load_cache(layer)   # layer still marked offloaded -> reads disk
                self._resident[layer] = cache
                self._offloaded.discard(layer)
            for cache in self._resident.values():
                cache.trim(n)

    def close(self) -> None:
        if not self._worker.is_alive():
            return                           # safe to call twice
        self._queue.put(None)                # drains queued prefetches, then stops
        self._worker.join()
        for f in self._dir.glob("kv_layer_*.safetensors"):
            f.unlink()                       # scratch state is disposable

    @property
    def resident_ids(self) -> list[int]:
        with self._lock:
            return list(self._resident)

    # ---- internals ----
    def _file(self, layer: int) -> Path:
        return self._dir / f"kv_layer_{layer:03d}.safetensors"

    def _run(self) -> None:
        while True:
            layer = self._queue.get()
            if layer is None:
                return
            with self._lock:
                if layer in self._resident:
                    ev = self._inflight.pop(layer, None)
                    if ev:
                        ev.set()
                    continue
                ev = self._inflight.get(layer)
                if ev is None:
                    continue
            try:
                self._materialize(layer, ev, evict=False)
            except Exception:
                pass                         # failed prefetch must not kill the worker

    def _materialize(self, layer: int, ev: threading.Event, evict: bool = True) -> KVCache:
        try:
            cache = self._load_cache(layer)
            with self._lock:
                self._resident[layer] = cache
                self._offloaded.discard(layer)
                # Eviction calls mx.save_safetensors, which must only run on the
                # main thread. Worker-driven prefetch loads never evict; the next
                # main-thread get() call will handle any budget overflow.
                if evict:
                    self._evict_locked(protect=layer)
            return cache
        finally:
            with self._lock:
                self._inflight.pop(layer, None)
            ev.set()

    def _new_cache(self, layer: int):
        """A fresh, empty cache of the configured kind for `layer`."""
        kind = self._cache_kinds[layer] if self._cache_kinds is not None else None
        return make_cache(kind, self._kv_quant)

    def _is_rotating(self, layer: int) -> bool:
        from .archspec import Rotating
        return (self._cache_kinds is not None
                and isinstance(self._cache_kinds[layer], Rotating))

    def _load_cache(self, layer: int) -> KVCache:
        if layer in self._offloaded:
            d = mx.load(str(self._file(layer)))
            if "_list_n" in d:                   # a Paired (CacheList) layer
                cache = self._new_cache(layer)   # CacheList of the right arity
                state = []
                for i in range(int(d["_list_n"][0])):
                    k = d[f"list{i}_keys"]
                    v = d.get(f"list{i}_values")
                    if v is None:                # zero-width values (indexer cache)
                        v = mx.zeros(tuple(d[f"list{i}_vshape"].tolist()), dtype=k.dtype)
                    state.append((k, v))
                cache.state = state
                mx.eval(*(c.keys for c in cache.caches),
                        *(c.values for c in cache.caches))
            elif "_qmeta" in d:                  # a quantized layer
                cache = QuantizedKVCache()
                cache.state = (
                    (d["keys_q"], d["keys_scales"], d["keys_biases"]),
                    (d["values_q"], d["values_scales"], d["values_biases"]),
                )
                # meta restores (offset, group_size, bits); the state setter
                # does NOT set offset for QuantizedKVCache.
                cache.meta_state = tuple(str(int(x)) for x in d["_qmeta"].tolist())
                mx.eval(*cache.keys, *cache.values)
            elif "_meta" in d:                   # a rotating layer
                cache = self._new_cache(layer)   # RotatingKVCache(max_size, keep)
                cache.state = (d["keys"], d["values"])
                cache.meta_state = tuple(str(int(x)) for x in d["_meta"].tolist())
                mx.eval(cache.keys, cache.values)   # force the disk read off-thread
            else:
                cache = KVCache()
                cache.state = (d["keys"], d["values"])
                mx.eval(cache.keys, cache.values)   # force the disk read off-thread
            return cache
        return self._new_cache(layer)            # never offloaded -> fresh + empty of the right kind

    def _evict_locked(self, protect: int) -> None:
        total = sum(_cache_nbytes(c) for c in self._resident.values())   # recompute: caches grow
        self.peak_bytes = max(self.peak_bytes, total)
        evicted = False
        # MRU (scan-resistant) eviction: iterate most-recently-used first so the cyclic
        # decode scan keeps an early-layer prefix resident (mirrors PieceCache).
        for layer in reversed(list(self._resident)):
            if total <= self._budget:
                break
            if layer == protect:
                continue
            cache = self._resident[layer]
            nb = _cache_nbytes(cache)
            if nb == 0:
                self._resident.pop(layer)        # empty: drop; reloads as fresh+empty
                continue
            if isinstance(cache, CacheList):
                # Paired kind: children are plain KVCaches. Flatten each child's
                # (keys, values) with an index prefix; "_list_n" marks the file
                # so _load_cache rebuilds the right shape. A child's values may
                # be zero-width (the DSA indexer cache stores no values) — that
                # round-trips through safetensors as an ordinary 0-dim-axis array.
                payload = {"_list_n": mx.array([len(cache.caches)])}
                for i, c in enumerate(cache.caches):
                    k, v = c.state
                    payload[f"list{i}_keys"] = k
                    if v.size > 0:
                        payload[f"list{i}_values"] = v
                    else:
                        # safetensors refuses empty arrays; keep the shape so
                        # _load_cache can rebuild the zero-width placeholder.
                        payload[f"list{i}_vshape"] = mx.array(v.shape)
            elif isinstance(cache, QuantizedKVCache):
                (kq, ks, kb), (vq, vs, vb) = cache.state
                payload = {
                    "keys_q": kq, "keys_scales": ks, "keys_biases": kb,
                    "values_q": vq, "values_scales": vs, "values_biases": vb,
                    "_qmeta": mx.array([int(x) for x in cache.meta_state]),
                }
            else:
                keys, values = cache.state
                payload = {"keys": keys, "values": values}
                if self._is_rotating(layer):
                    # RotatingKVCache.meta_state == (keep, max_size, offset, _idx) as strs.
                    payload["_meta"] = mx.array([int(x) for x in cache.meta_state])
            mx.save_safetensors(str(self._file(layer)), payload)
            self._resident.pop(layer)
            self._offloaded.add(layer)
            total -= nb
            evicted = True
        if evicted:
            mx.clear_cache()                     # reclaim Metal buffers we freed
