import time
import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from nunspark.archspec import KVQuant
from nunspark.kv_store import KVStore


def _fill(cache: KVCache, ntok: int) -> tuple[mx.array, mx.array]:
    """Append `ntok` random tokens (B=1, n_kv_heads=2, head_dim=4) to a cache."""
    k = mx.random.normal((1, 2, ntok, 4))
    v = mx.random.normal((1, 2, ntok, 4))
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return cache.state  # exact (keys, values) of length == offset


def _probe_nbytes(ntok: int = 8) -> int:
    c = KVCache()
    _fill(c, ntok)
    return int(c.nbytes)


def test_reuse_returns_same_live_object(tmp_path):
    store = KVStore(tmp_path, budget_bytes=10**12)
    a = store.get(0)
    a.update_and_fetch(mx.random.normal((1, 2, 3, 4)), mx.random.normal((1, 2, 3, 4)))
    b = store.get(0)
    store.close()
    assert b is a                 # same live cache, mutations preserved
    assert b.offset == 3
    assert store.hits == 1 and store.misses == 1


def test_fresh_layer_starts_empty(tmp_path):
    store = KVStore(tmp_path, budget_bytes=10**12)
    c = store.get(2)
    store.close()
    assert isinstance(c, KVCache)
    assert c.offset == 0          # never offloaded -> fresh + empty


def test_overflow_offloads_to_disk_and_reloads_bit_identical(tmp_path):
    nb = _probe_nbytes(8)
    store = KVStore(tmp_path, budget_bytes=int(nb * 1.5), prefetch=False)  # ~1 cache fits
    saved = {}
    for layer in range(4):
        c = store.get(layer)
        saved[layer] = _fill(c, 8)            # each grows to nb bytes
    store.get(99)                             # one more miss -> forces an eviction pass

    assert len(store.resident_ids) <= 2       # budget bounds residency (1 + protect)
    assert list(tmp_path.glob("kv_layer_*.safetensors"))  # overflow went to disk

    for layer in range(4):                    # every layer round-trips (hit or reload)
        bk, bv = store.get(layer).state
        sk, sv = saved[layer]
        assert float(mx.max(mx.abs(bk - sk))) == 0.0   # exact round-trip
        assert float(mx.max(mx.abs(bv - sv))) == 0.0
    store.close()


def test_peak_bytes_tracks_resident_total(tmp_path):
    nb = _probe_nbytes(8)
    store = KVStore(tmp_path, budget_bytes=10**12)  # everything resident
    for layer in range(3):
        _fill(store.get(layer), 8)
    store.get(99)                             # triggers a recompute of the total
    store.close()
    assert store.peak_bytes >= nb * 3         # saw all three resident at once


def _evict_one(store):
    """Fill layers 0 and 1, then force one to disk. Return (evicted_layer, saved)."""
    saved = {}
    for layer in (0, 1):
        saved[layer] = _fill(store.get(layer), 8)
    store.get(2)                              # miss -> eviction pass offloads one layer
    resident = set(store.resident_ids)
    evicted = next(l for l in (0, 1) if l not in resident)
    return evicted, saved[evicted]


def test_prefetch_makes_offloaded_layer_resident(tmp_path):
    nb = _probe_nbytes(8)
    store = KVStore(tmp_path, budget_bytes=int(nb * 1.5))
    layer, _ = _evict_one(store)
    assert layer not in store.resident_ids and (tmp_path / f"kv_layer_{layer:03d}.safetensors").exists()

    hits0 = store.hits
    store.prefetch([layer])
    for _ in range(200):                      # wait for the worker to reload it
        if layer in store.resident_ids:
            break
        time.sleep(0.005)
    assert layer in store.resident_ids        # prefetched into residency off-thread
    store.get(layer)
    store.close()
    assert store.hits == hits0 + 1            # the get() was a hit, not a blocking load


def test_reload_failure_surfaces_and_worker_survives(tmp_path):
    nb = _probe_nbytes(8)
    store = KVStore(tmp_path, budget_bytes=int(nb * 1.5))
    layer, _ = _evict_one(store)
    (tmp_path / f"kv_layer_{layer:03d}.safetensors").write_bytes(b"garbage")  # corrupt it

    store.prefetch([layer])                   # worker tries, fails silently
    with pytest.raises(Exception):
        store.get(layer)                      # owner retries -> surfaces the error

    # Worker still alive: a fresh offloaded layer still reloads correctly.
    s3 = _fill(store.get(3), 8)
    store.get(4); store.get(5)                # pressure -> evict layer 3 to disk
    back = store.get(3)
    bk, bv = back.state
    store.close()
    assert float(mx.max(mx.abs(bk - s3[0]))) == 0.0


def test_kv_cyclic_scan_reuse_scales_with_budget(tmp_path):
    # Realistic decode: each step touches layers 0..N-1 in order AND appends one
    # token to each, so KV grows over steps. (A one-shot fill won't do: an
    # offloaded cache reloads TRIMMED to its actual token count, so tiny reloaded
    # caches all fit any budget and hide the effect. Growth keeps eviction biting.)
    # Realistic per-layer width (8 kv-heads x 128 head_dim) avoids the same
    # buffer-pre-allocation artifact. Under LRU the cyclic scan reloads every
    # offloaded layer every step (full thrash == N*STEPS); a scan-resistant policy
    # keeps an early-layer prefix resident, so more kv-budget => fewer reloads.
    N, STEPS = 8, 20

    def one_token():
        return (mx.random.normal((1, 8, 1, 128)), mx.random.normal((1, 8, 1, 128)))

    probe = KVCache()                          # size the budget against a mid-run layer
    for _ in range(STEPS // 2):
        probe.update_and_fetch(*one_token())
    mx.eval(probe.keys, probe.values)
    nb_mid = int(probe.nbytes)

    def reloads(budget_layers, dirname):
        store = KVStore(tmp_path / dirname, budget_bytes=int(nb_mid * budget_layers),
                        prefetch=False)
        for _ in range(STEPS):
            for layer in range(N):
                c = store.get(layer)
                c.update_and_fetch(*one_token())   # decode appends one token
                mx.eval(c.keys, c.values)
        out = store.misses
        store.close()
        return out

    small = reloads(2, "small")
    big = reloads(6, "big")
    assert small < N * STEPS    # MRU reuses even at a modest budget (LRU: == N*STEPS)
    assert big < small          # more kv-budget -> strictly fewer reloads (LRU: equal)


def _fill_q(cache, ntok: int):
    k = mx.random.normal((1, 2, ntok, 64))   # head_dim 64: divisible by group_size
    v = mx.random.normal((1, 2, ntok, 64))
    cache.update_and_fetch(k, v)
    mx.eval(*cache.keys, *cache.values)
    return cache.state   # ((kq, ks, kb), (vq, vs, vb)), sliced to offset


def test_kv_quant_new_cache_is_quantized(tmp_path):
    store = KVStore(tmp_path, budget_bytes=10**12, kv_quant=KVQuant(bits=8))
    c = store.get(0)
    store.close()
    assert isinstance(c, QuantizedKVCache)
    assert c.bits == 8 and c.group_size == 64


def test_kv_quant_empty_cache_nbytes_safe(tmp_path):
    # QuantizedKVCache.nbytes raises before the first append; the budget
    # recompute in _evict_locked must treat empty caches as 0 bytes.
    store = KVStore(tmp_path, budget_bytes=1, prefetch=False,
                    kv_quant=KVQuant(bits=8))
    store.get(0)            # fresh empty quantized cache
    store.get(1)            # second access triggers an eviction pass
    store.close()           # reaching here without AttributeError is the test


def test_kv_quant_offload_roundtrip_bit_identical(tmp_path):
    probe = KVStore(tmp_path / "probe", budget_bytes=10**12,
                    kv_quant=KVQuant(bits=8))
    _fill_q(probe.get(0), 8)
    nb = int(probe.get(0).nbytes)
    probe.close()

    store = KVStore(tmp_path / "s", budget_bytes=int(nb * 1.5), prefetch=False,
                    kv_quant=KVQuant(bits=8))
    saved = {}
    for layer in range(4):
        saved[layer] = _fill_q(store.get(layer), 8)
    store.get(99)                                  # force eviction pass
    assert list((tmp_path / "s").glob("kv_layer_*.safetensors"))

    for layer in range(4):
        got = store.get(layer)
        assert isinstance(got, QuantizedKVCache)
        assert got.offset == 8
        for g3, s3 in zip(got.state, saved[layer]):     # keys triple, values triple
            for g, s in zip(g3, s3):
                assert float(mx.max(mx.abs(g - s))) == 0.0
    store.close()


def test_kv_quant_truncate_works(tmp_path):
    store = KVStore(tmp_path, budget_bytes=10**12, kv_quant=KVQuant(bits=8))
    _fill_q(store.get(0), 8)
    store.truncate(3)
    assert store.get(0).offset == 5
    store.close()
