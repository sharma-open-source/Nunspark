import tempfile

from mlx_lm.models.cache import KVCache, RotatingKVCache

from nunspark.archspec import KV, Rotating
from nunspark.kv_store import KVStore


def test_default_cache_kinds_are_all_kvcache():
    with tempfile.TemporaryDirectory() as d:
        kv = KVStore(d, budget_bytes=10**12)
        try:
            assert isinstance(kv.get(0), KVCache)
            assert isinstance(kv.get(5), KVCache)
        finally:
            kv.close()


def test_mixed_cache_kinds_build_correct_types():
    # layer 0 global -> KVCache; layer 1 sliding -> RotatingKVCache(window)
    with tempfile.TemporaryDirectory() as d:
        kv = KVStore(d, budget_bytes=10**12, cache_kinds=[KV, Rotating(window=4)])
        try:
            assert isinstance(kv.get(0), KVCache)
            c1 = kv.get(1)
            assert isinstance(c1, RotatingKVCache)
            assert c1.max_size == 4
        finally:
            kv.close()


import mlx.core as mx


def _feed(cache, n_tokens, head_dim=2, kv_heads=1):
    """Append n_tokens single-token steps to a cache; return final fetched keys."""
    keys = None
    for t in range(n_tokens):
        k = mx.full((1, kv_heads, 1, head_dim), float(t + 1))
        v = mx.full((1, kv_heads, 1, head_dim), float(t + 1))
        keys, _ = cache.update_and_fetch(k, v)
    mx.eval(keys)
    return keys


def test_rotating_cache_survives_eviction_roundtrip():
    # Reference: a standalone RotatingKVCache fed 10 single-token steps, window=4.
    from mlx_lm.models.cache import RotatingKVCache
    ref = RotatingKVCache(max_size=4, keep=0)
    ref_keys = _feed(ref, 10)

    # KVStore-managed: layer 0 is Rotating(4), layer 1 is plain KV (a second
    # layer is required: get() always protects the layer it returns from
    # eviction, so a single-layer store can never actually evict anything).
    # With a 1-byte budget, touching layer 1 between layer-0 steps forces
    # layer 0 out of residency -> serialized to disk -> reloaded on next get().
    with tempfile.TemporaryDirectory() as d:
        kv = KVStore(d, budget_bytes=1, cache_kinds=[Rotating(window=4), KV])
        try:
            keys = None
            for t in range(10):
                c = kv.get(0)                       # may reload from disk each step
                k = mx.full((1, 1, 1, 2), float(t + 1))
                v = mx.full((1, 1, 1, 2), float(t + 1))
                keys, _ = c.update_and_fetch(k, v)
                mx.eval(keys)

                c1 = kv.get(1)                       # touch a second layer ->
                z = mx.zeros((1, 1, 1, 2))            # forces layer 0 out of
                _ = c1.update_and_fetch(z, z)          # residency under budget=1
                mx.eval(_[0])
            # After 10 steps through eviction, the fetched window must match the
            # non-evicted reference exactly (same windowed contents + order).
            assert 0 in kv._offloaded or kv.misses > 10  # sanity: eviction actually happened
            assert keys.shape == ref_keys.shape
            assert mx.array_equal(keys, ref_keys)
        finally:
            kv.close()
