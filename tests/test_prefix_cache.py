import mlx.core as mx

from nunspark.archspec import KV, Rotating
from nunspark.prefix_cache import PrefixCache


def _append(kv, layers: int, ntok: int):
    """Simulate a forward: append ntok tokens to each layer's cache."""
    for l in range(layers):
        kv.get(l).update_and_fetch(
            mx.random.normal((1, 2, ntok, 4)), mx.random.normal((1, 2, ntok, 4)))


def _pc(tmp_path, cache_kinds=None):
    return PrefixCache(tmp_path / "pc", kv_budget=10**12, prefetch=False,
                       cache_kinds=cache_kinds)


def test_cold_start_returns_full_prompt(tmp_path):
    pc = _pc(tmp_path)
    kv, suffix = pc.begin([1, 2, 3, 4])
    assert suffix == [1, 2, 3, 4]
    assert kv.get(0).offset == 0
    pc.close()


def test_warm_hit_returns_suffix_only(tmp_path):
    pc = _pc(tmp_path)
    kv, suffix = pc.begin([1, 2, 3, 4])
    _append(kv, 2, len(suffix))            # "prefill" 4 tokens
    _append(kv, 2, 2)                      # "generate" tokens 9, 10 (last not in KV)
    pc.commit([1, 2, 3, 4, 9, 10, 11])     # claims 7; actual KV len is 6

    kv2, suffix2 = pc.begin([1, 2, 3, 4, 9, 10, 99, 100])
    assert kv2 is kv                       # same live store
    assert suffix2 == [99, 100]            # common prefix [1,2,3,4,9,10] reused
    assert kv2.get(0).offset == 6
    pc.close()


def test_divergent_prompt_trims_kv(tmp_path):
    pc = _pc(tmp_path)
    kv, suffix = pc.begin([1, 2, 3, 4, 5, 6])
    _append(kv, 2, 6)
    pc.commit([1, 2, 3, 4, 5, 6])

    kv2, suffix2 = pc.begin([1, 2, 3, 99])
    assert suffix2 == [99]
    assert kv2.get(0).offset == 3          # trimmed back to the common prefix
    pc.close()


def test_fully_cached_prompt_refeeds_last_token(tmp_path):
    pc = _pc(tmp_path)
    kv, _ = pc.begin([1, 2, 3, 4])
    _append(kv, 2, 4)
    pc.commit([1, 2, 3, 4])

    kv2, suffix2 = pc.begin([1, 2, 3, 4])  # identical prompt
    assert suffix2 == [4]                  # at least one token must be fed
    assert kv2.get(0).offset == 3
    pc.close()


def test_rotated_window_falls_back_to_cold(tmp_path):
    kinds = [KV, Rotating(window=4)]
    pc = _pc(tmp_path, cache_kinds=kinds)
    kv, _ = pc.begin([1, 2, 3, 4, 5, 6])
    for l, ntok in ((0, 6), (1, 6)):
        kv.get(l).update_and_fetch(
            mx.random.normal((1, 2, ntok, 4)), mx.random.normal((1, 2, ntok, 4)))
    pc.commit([1, 2, 3, 4, 5, 6])

    # 6 > window 4: the rotating layer has rotated; trimming back to a
    # 3-token prefix is impossible -> cold start.
    kv2, suffix2 = pc.begin([1, 2, 3, 99])
    assert suffix2 == [1, 2, 3, 99]
    assert kv2.get(0).offset == 0
    pc.close()


def test_no_overlap_resets(tmp_path):
    pc = _pc(tmp_path)
    kv, _ = pc.begin([1, 2, 3])
    _append(kv, 1, 3)
    pc.commit([1, 2, 3])
    kv2, suffix2 = pc.begin([7, 8, 9])
    assert suffix2 == [7, 8, 9]
    assert kv2.get(0).offset == 0
    pc.close()


def test_commit_truncates_overhang(tmp_path):
    """Generation aborted mid-speculation can leave MORE tokens in KV than
    were emitted; commit must trim the store down to the claimed tokens."""
    pc = _pc(tmp_path)
    kv, _ = pc.begin([1, 2, 3])
    _append(kv, 1, 3)
    _append(kv, 1, 5)                      # 8 in KV
    pc.commit([1, 2, 3, 4, 5])             # only 5 claimed
    assert kv.get(0).offset == 5
    pc.close()


def test_all_rotating_layers_disables_reuse(tmp_path):
    """When every layer is sliding-window the true length is unknowable from
    the caches, so reuse is disabled: begin() always cold-starts and commit()
    is a safe no-op (the probe layer is None, never read)."""
    kinds = [Rotating(window=4), Rotating(window=4)]
    pc = _pc(tmp_path, cache_kinds=kinds)
    kv, suffix = pc.begin([1, 2, 3])
    assert suffix == [1, 2, 3]
    _append(kv, 2, 3)
    pc.commit([1, 2, 3])                    # no-op, must not raise
    kv2, suffix2 = pc.begin([1, 2, 3, 4])   # would be a warm hit if reuse were on
    assert suffix2 == [1, 2, 3, 4]          # still cold
    assert kv2.get(0).offset == 0
    pc.close()


def test_commit_overhang_on_rotated_cache_drops_slot(tmp_path):
    """If an overhang must be trimmed but a rotating layer has already
    rotated, trimming would corrupt it — drop the slot so the next request
    prefills cold rather than reusing a corrupt cache."""
    kinds = [KV, Rotating(window=4)]
    pc = _pc(tmp_path, cache_kinds=kinds)
    kv, _ = pc.begin([1, 2, 3, 4, 5, 6])
    for l, ntok in ((0, 6), (1, 6)):
        kv.get(l).update_and_fetch(
            mx.random.normal((1, 2, ntok, 4)), mx.random.normal((1, 2, ntok, 4)))
    pc.commit([1, 2, 3, 4])                # claim 4, actual 6, rotating rotated

    kv2, suffix2 = pc.begin([1, 2, 3, 4])
    assert suffix2 == [1, 2, 3, 4]         # cold: slot was dropped
    assert kv2.get(0).offset == 0
    pc.close()
