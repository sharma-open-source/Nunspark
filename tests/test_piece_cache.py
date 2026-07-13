import time
import mlx.core as mx
import pytest
from nunspark.piece_cache import PieceCache


def _arr_loader(nbytes_each):
    # 256x256 float32 = 262144 bytes; returns a fresh dict each call
    n = nbytes_each
    side = 256
    loads = []

    def loader(pid):
        loads.append(pid)
        return {"w": mx.zeros((side, side), dtype=mx.float32)}

    return loader, loads, side * side * 4


def test_reuse_avoids_reload():
    loader, loads, _ = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=10**9)
    cache.get("x")
    cache.get("x")
    cache.get("x")
    cache.close()
    assert loads == ["x"]            # loaded once, reused twice
    assert cache.hits == 2 and cache.misses == 1


def test_mru_eviction_by_bytes():
    # Scan-resistant policy: when over budget, evict the MOST-recently-used
    # unpinned piece (not the least). Accessing a, b, c with room for 2 keeps the
    # early "a" resident and evicts the most-recent prior piece "b".
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=int(one * 2.5))  # fits 2 pieces
    cache.get("a")
    cache.get("b")
    cache.get("c")                   # over budget -> evict MRU ("b"), keep "a"
    resident = set(cache.resident_ids)
    cache.close()
    assert resident == {"a", "c"}
    assert cache.peak_bytes <= one * 2.5 + one   # transient: current + one in-flight


def test_cyclic_scan_reuse_scales_with_budget():
    # The engine's access pattern: every "token" scans pieces 0..N-1 in order.
    # Under LRU this thrashes (every access misses) until the whole set fits, so
    # raising the budget below that line changes nothing. A scan-resistant policy
    # keeps an early-piece prefix resident, so more budget => fewer disk loads.
    N, TOKENS, one = 12, 5, 256 * 256 * 4   # _arr_loader piece size

    def run(budget_pieces):
        loader, loads, _ = _arr_loader(0)
        cache = PieceCache(loader, budget_bytes=one * budget_pieces)
        for _ in range(TOKENS):
            for i in range(N):
                cache.get(f"p{i:02d}")
        cache.close()
        return len(loads)

    small = run(3)
    big = run(9)
    assert small < N * TOKENS    # even a modest budget yields reuse (LRU: == N*TOKENS)
    assert big < small           # more budget -> strictly fewer disk loads (LRU: equal)


def test_pinned_piece_survives_eviction():
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=int(one * 1.5), pinned=["keep"])
    cache.get("keep")
    cache.get("a")
    cache.get("b")
    resident = set(cache.resident_ids)
    cache.close()
    assert "keep" in resident        # pinned never evicted


def test_prefetch_overlaps_loading():
    events = []

    def slow_loader(pid):
        events.append(("start", pid, time.perf_counter()))
        time.sleep(0.05)
        events.append(("end", pid, time.perf_counter()))
        return {"w": mx.zeros((8, 8))}

    cache = PieceCache(slow_loader, budget_bytes=10**9)
    cache.prefetch(["a", "b"])       # background loads begin
    time.sleep(0.07)                 # main thread "computes" while loads run
    t = time.perf_counter()
    cache.get("a")                   # already prefetched -> returns fast
    dt = time.perf_counter() - t
    cache.close()
    assert dt < 0.02                 # did NOT block on the 50ms load
    assert cache.hits >= 1


def test_prefetch_and_get_dedup_to_single_load():
    loader, loads, _ = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=10**9)
    cache.prefetch(["z"])
    cache.get("z")                   # may race the worker; must load only once
    cache.close()
    assert loads.count("z") == 1


def test_loader_failure_does_not_hang_or_kill_worker():
    state = {"fail": True}

    def flaky_loader(pid):
        if state["fail"]:
            raise RuntimeError("boom")
        return {"w": mx.zeros((4, 4))}

    cache = PieceCache(flaky_loader, budget_bytes=10**9)
    cache.prefetch(["a"])             # background load will fail; must not hang/kill
    with pytest.raises(RuntimeError):
        cache.get("a")                # surfaces the error instead of hanging forever

    # Worker survived: a later successful prefetch is still processed.
    state["fail"] = False
    cache.prefetch(["b"])
    got = cache.get("b")
    cache.close()
    assert "w" in got and "b" in cache.resident_ids


def test_warmer_single_threaded_never_calls_pather(tmp_path):
    # io_threads=1 => no warmer pool => pather is never used; behavior unchanged.
    warmed = []
    loader, loads, _ = _arr_loader(0)
    cache = PieceCache(
        loader, budget_bytes=10**9, io_threads=1,
        pather=lambda pid: warmed.append(pid) or (tmp_path / f"{pid}.bin"),
    )
    cache.prefetch(["a", "b"])
    cache.get("a")
    cache.get("b")
    cache.close()
    assert warmed == []                       # pather untouched when single-threaded
    assert sorted(loads) == ["a", "b"]


def test_warmer_parallel_warms_each_file_once_and_dedups_loads(tmp_path):
    files = {}
    for pid in ("a", "b", "c"):
        f = tmp_path / f"{pid}.bin"
        f.write_bytes(b"\x00" * (1 << 20))    # 1 MiB so _warm has real bytes to read
        files[pid] = f
    warmed = []
    loader, loads, _ = _arr_loader(0)

    def pather(pid):
        warmed.append(pid)
        return files[pid]

    cache = PieceCache(loader, budget_bytes=10**9, io_threads=4, pather=pather)
    # NOTE: dedup (prefetch-then-get loads once) is verified here indirectly; the
    # warm pool typically wins the race for these tiny files, which is acceptable.
    cache.prefetch(["a", "b", "c"])
    got = {pid: cache.get(pid) for pid in ("a", "b", "c")}
    cache.close()

    assert all("w" in g for g in got.values())
    assert sorted(loads) == ["a", "b", "c"]   # each materialized exactly once
    assert sorted(warmed) == ["a", "b", "c"]  # each warmed via pather exactly once


def test_warmer_close_is_clean_and_idempotent(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"\x00" * (1 << 20))
    loader, _loads, _ = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=10**9, io_threads=4, pather=lambda pid: f)
    cache.prefetch([f"p{i}" for i in range(8)])
    cache.close()                             # must drain warms + join worker, no hang
    cache.close()                             # idempotent
    assert cache._warm_pool is None


def test_warmer_missing_file_does_not_break_load(tmp_path):
    # Warming is best-effort: a bad path must not stop the piece from materializing.
    loader, loads, _ = _arr_loader(0)
    cache = PieceCache(
        loader, budget_bytes=10**9, io_threads=4,
        pather=lambda pid: tmp_path / "does_not_exist.bin",
    )
    cache.prefetch(["a"])
    got = cache.get("a")                      # falls back to a cold load, still correct
    cache.close()
    assert "w" in got and loads == ["a"]


def test_warm_does_not_change_tracked_bytes(tmp_path):
    # The warmer reads into the OS page cache; it must NOT allocate mx arrays or
    # touch the byte-budget accounting (spec memory-bound invariant).
    f = tmp_path / "x.bin"
    f.write_bytes(b"\x00" * (1 << 20))
    loader, _loads, _ = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=10**9, io_threads=4, pather=lambda pid: f)
    cache._warm("p")  # call the raw I/O step in isolation (bypassing the normal flow on purpose)
    assert cache._bytes == 0
    assert cache.resident_ids == []
    cache.close()
