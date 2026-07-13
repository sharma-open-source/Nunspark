import mlx.core as mx

from nunspark.kv_store import KVStore


def _append(cache, n_tokens, n_kv_heads=2, head_dim=4):
    k = mx.zeros((1, n_kv_heads, n_tokens, head_dim))
    v = mx.zeros((1, n_kv_heads, n_tokens, head_dim))
    cache.update_and_fetch(k, v)


def test_truncate_rewinds_all_layers(tmp_path):
    kv = KVStore(tmp_path / "kv", budget_bytes=10**12)
    try:
        for layer in (0, 1, 2):
            _append(kv.get(layer), 5)
        assert [kv.get(l).offset for l in (0, 1, 2)] == [5, 5, 5]
        kv.truncate(2)
        assert [kv.get(l).offset for l in (0, 1, 2)] == [3, 3, 3]
    finally:
        kv.close()


def test_truncate_noop_and_clamp(tmp_path):
    kv = KVStore(tmp_path / "kv", budget_bytes=10**12)
    try:
        _append(kv.get(0), 3)
        kv.truncate(0)                 # non-positive -> no-op
        assert kv.get(0).offset == 3
        kv.truncate(10)                # more than present -> clamps to 0
        assert kv.get(0).offset == 0
    finally:
        kv.close()
