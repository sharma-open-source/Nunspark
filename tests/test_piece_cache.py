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


def test_stats_classify_pids_by_class():
    # "expert" if "_expert_" in pid, "core" if pid endswith "_core", else "dense".
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=10**9)
    cache.get("layer_000_core")
    cache.get("layer_000_expert_3")
    cache.get("layer_001")            # dense whole-layer piece
    cache.get("layer_000_core")       # hit
    cache.close()

    stats = cache.stats()
    assert stats["hits"] == {"dense": 0, "core": 1, "expert": 0}
    assert stats["misses"] == {"dense": 1, "core": 1, "expert": 1}
    assert stats["bytes_loaded"] == {"dense": one, "core": one, "expert": one}
    # existing global counters keep working exactly as before, as sums of the classes
    assert cache.hits == 1 and cache.misses == 3


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


# ---- two-region policy (plan4 M2): MRU main region + LRU expert region ----

def _E(i):
    return f"layer_000_expert_{i}"


def test_expert_region_evicts_lru_first():
    # Expert region cap = 4*one * 0.625 = 2.5 pieces. Touch e1 again before
    # loading e3: the least-recently-used expert (e2) must be the one evicted.
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.625)
    cache.get(_E(1))
    cache.get(_E(2))
    cache.get(_E(1))                  # hit -> e1 becomes most-recent
    cache.get(_E(3))                  # over expert cap -> evict LRU (e2)
    resident = set(cache.resident_ids)
    cache.close()
    assert resident == {_E(1), _E(3)}
    assert loads.count(_E(1)) == 1    # e1 was a hit, never reloaded


def test_dense_pressure_never_evicts_experts():
    # frac=0.5: expert cap = 2*one, main cap = 2*one once the split is active.
    # Filling the expert region, then hammering dense pieces, must only evict
    # dense pieces (per the main region's MRU policy).
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.5)
    cache.get(_E(1))
    cache.get(_E(2))
    for pid in ("a", "b", "c", "d", "e"):
        cache.get(pid)
    resident = set(cache.resident_ids)
    cache.close()
    assert {_E(1), _E(2)} <= resident            # experts untouched
    assert sum(p in resident for p in "abcde") == 2   # main capped at 2 pieces
    assert loads.count(_E(1)) == 1 and loads.count(_E(2)) == 1


def test_expert_pressure_never_evicts_dense():
    # Dense pieces loaded before any expert stay resident (main cap = 2*one
    # after the split activates, and a+b fit exactly); expert churn beyond the
    # expert cap must evict only experts.
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.5)
    cache.get("a")
    cache.get("b")
    for i in range(1, 6):
        cache.get(_E(i))
    resident = set(cache.resident_ids)
    cache.close()
    assert {"a", "b"} <= resident
    experts = {p for p in resident if "_expert_" in p}
    assert experts == {_E(4), _E(5)}             # LRU kept the 2 most recent
    assert loads.count("a") == 1 and loads.count("b") == 1


def test_stats_report_per_region_occupancy():
    loader, _loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.5)
    cache.get("a")
    cache.get(_E(1))
    cache.get(_E(2))
    stats = cache.stats()
    cache.close()
    assert stats["resident_bytes"] == {"main": one, "expert": 2 * one}


def test_pinned_core_survives_both_regions_pressure():
    # A pinned core piece must never be evicted, even when the split activates
    # and shrinks the main region below what is resident.
    loader, _loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.5,
                       pinned=["layer_000_core"])
    cache.get("layer_000_core")
    for pid in ("a", "b", "c"):
        cache.get(pid)               # main at 4*one (full budget; split inactive)
    for i in range(1, 5):
        cache.get(_E(i))             # split activates: main cap -> 2*one
    resident = set(cache.resident_ids)
    cache.close()
    assert "layer_000_core" in resident


# ---- two-tier prefetch queue (plan4 M3b): demand beats speculative ----

def test_demand_prefetch_drains_before_speculative():
    # Occupy the single worker with a blocking load, enqueue speculative pids
    # first and demand pids second, then release: the worker must materialize ALL
    # demand pids before ANY speculative pid (tier beats FIFO seq order).
    import threading
    order = []
    started = threading.Event()
    gate = threading.Event()

    def loader(pid):
        if pid == "block":
            started.set()
            gate.wait()
        order.append(pid)
        return {"w": mx.zeros((2, 2))}

    cache = PieceCache(loader, budget_bytes=10**9)
    cache.prefetch(["block"])                       # seizes the worker
    assert started.wait(1)
    cache.prefetch([_E(1), _E(2), _E(3)], speculative=True)  # low tier, enqueued first
    cache.prefetch(["d1", "d2"])                    # demand tier, enqueued after
    gate.set()
    cache.close()                                   # drains all queued, then stops

    assert order[0] == "block"
    demand_last = max(order.index("d1"), order.index("d2"))
    spec_first = min(order.index(_E(1)), order.index(_E(2)), order.index(_E(3)))
    assert demand_last < spec_first                 # every demand pid before any spec


def test_speculative_dedup_and_used_counter():
    loader, loads, _one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=10**9)
    cache.prefetch([_E(1)], speculative=True)
    cache.prefetch([_E(1)])                # already in flight -> deduped, not re-issued
    cache.prefetch([_E(1)], speculative=True)  # dedup again
    got = cache.get(_E(1))                 # hit (waits on the in-flight spec load) -> used
    cache.close()

    st = cache.stats()["speculative"]
    assert st["issued"] == 1               # only the first reservation issued a load
    assert st["used"] == 1                 # the get() hit the speculatively-loaded piece
    assert st["wasted_bytes"] == 0
    assert loads.count(_E(1)) == 1 and "w" in got   # materialized exactly once


def _wait_staged(cache, pid):
    for _ in range(2000):
        if pid in cache._staging:
            return
        time.sleep(0.001)
    raise AssertionError(f"{pid} never staged")


def test_speculative_load_lands_in_staging_not_expert_lru():
    # v3: speculative loads land in the STAGING buffer, never the expert LRU.
    # Staging shares the expert budget (occupied staging squeezes the LRU tail so
    # total bytes never grow past the expert share), so a blast far bigger than
    # the region costs demand at most the staging cap — never the whole set.
    # Expert share = 8*one*0.5 = 4*one; staging cap 1*one -> 3 demand experts
    # must survive any blast.
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 8, expert_frac=0.5,
                       spec_staging_bytes=int(one))
    cache.begin_pass()
    for i in (1, 2, 3):                    # demand working set (3*one of 4*one share)
        cache.get(_E(i))
    for i in range(10, 20):                # speculative blast >> expert region
        cache.prefetch([_E(i)], speculative=True)
    cache.close()                          # drains all queued spec loads into staging

    resident = set(cache.resident_ids)
    assert {_E(1), _E(2), _E(3)} <= resident                  # demand experts survived
    assert not any(_E(i) in resident for i in range(10, 20))  # no spec in the LRU
    stats = cache.stats()
    # expert LRU + staging never exceed the expert share despite the blast
    assert stats["resident_bytes"]["expert"] + cache._staging_bytes <= 4 * one
    assert cache._staging_bytes <= one                        # staging under its cap


def test_staging_cap_drops_oldest_and_counts_wasted():
    # Staging cap = 2*one -> holds 2 staged pieces; a 3rd drops the OLDEST staged
    # entry (and counts its bytes wasted), never touching the resident regions.
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.5,
                       spec_staging_bytes=int(one * 2))
    cache.begin_pass()
    for i in (1, 2, 3):
        cache.prefetch([_E(i)], speculative=True)
        _wait_staged(cache, _E(i))
    staged = set(cache._staging)
    cache.close()

    assert staged == {_E(2), _E(3)}        # oldest (_E(1)) dropped to make room
    st = cache.stats()["speculative"]
    assert st["issued"] == 3
    assert st["used"] == 0
    assert st["wasted_bytes"] == one       # _E(1) dropped by cap pressure


def test_staged_piece_demanded_migrates_to_expert_lru_as_demand():
    # A demand get() on a staged piece counts used ONCE, moves it out of staging
    # into the expert LRU, and from then on it behaves as a plain demand piece
    # (a later plain hit does not re-count used; normal LRU eviction is not wasted).
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.625,  # expert cap 2
                       spec_staging_bytes=int(one * 2))
    cache.begin_pass()
    cache.prefetch([_E(1)], speculative=True)
    _wait_staged(cache, _E(1))
    got = cache.get(_E(1))                 # staging hit -> used, migrate to expert LRU
    assert "w" in got
    assert _E(1) not in cache._staging
    assert _E(1) in cache.resident_ids     # now a normal demand piece
    st = cache.stats()["speculative"]
    assert st["used"] == 1 and st["wasted_bytes"] == 0

    cache.get(_E(1))                       # plain hit: used must NOT increment again
    assert cache.stats()["speculative"]["used"] == 1

    cache.get(_E(2))
    cache.get(_E(3))                       # over expert cap -> LRU-evict _E(1) as demand
    resident = set(cache.resident_ids)
    cache.close()

    assert _E(1) not in resident
    assert cache.stats()["speculative"]["wasted_bytes"] == 0  # used, so never wasted
    assert loads.count(_E(1)) == 1         # loaded exactly once


def test_staged_piece_expires_after_one_pass_grace():
    # A staged piece survives its insertion pass plus exactly one more (begin_pass
    # grace); if still undemanded it is expired at begin_pass and counted wasted.
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4, expert_frac=0.5,
                       spec_staging_bytes=int(one * 4))
    cache.begin_pass()                     # epoch 1
    cache.prefetch([_E(1)], speculative=True)
    _wait_staged(cache, _E(1))             # staged at epoch 1

    cache.begin_pass()                     # epoch 2: within grace -> survives
    assert _E(1) in cache._staging
    assert cache.stats()["speculative"]["wasted_bytes"] == 0

    cache.begin_pass()                     # epoch 3: past grace -> expired
    assert _E(1) not in cache._staging
    st = cache.stats()["speculative"]
    cache.close()
    assert st["used"] == 0
    assert st["wasted_bytes"] == one       # never demanded within its window


def test_demand_get_joining_inflight_speculative_load_counts_used():
    # A demand get() that joins a still-in-flight speculative load (waits on the
    # dedup Event) must count as `used` — the guess was right before the load
    # even finished — and the piece lands as a normal hot demand piece (never
    # later counted wasted).
    import threading
    started = threading.Event()
    gate = threading.Event()

    def loader(pid):
        started.set()
        gate.wait()
        return {"w": mx.zeros((2, 2))}

    cache = PieceCache(loader, budget_bytes=10**9)
    cache.prefetch([_E(1)], speculative=True)
    assert started.wait(1)                 # spec load is in flight (blocked)
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("w", cache.get(_E(1))))
    t.start()
    for _ in range(2000):                  # used is counted at JOIN time,
        if cache.speculative_used == 1:    # before the load materializes
            break
        time.sleep(0.001)
    assert cache.speculative_used == 1
    gate.set()
    t.join(5)
    cache.close()

    st = cache.stats()["speculative"]
    assert st["issued"] == 1 and st["used"] == 1 and st["wasted_bytes"] == 0
    assert "w" in got["w"]                 # the joining get() got the weights
    assert _E(1) in cache.resident_ids     # landed resident as a demand piece


def test_split_inactive_dense_only_keeps_full_budget():
    # With the DEFAULT expert_frac, a dense-only workload must still use the
    # whole budget (the split only activates on the first expert insert) —
    # dense models see zero behavior change from the two-region policy.
    loader, loads, one = _arr_loader(0)
    cache = PieceCache(loader, budget_bytes=one * 4)     # default expert_frac
    for pid in ("a", "b", "c", "d"):
        cache.get(pid)
    for pid in ("a", "b", "c", "d"):
        cache.get(pid)               # all hits: nothing was evicted
    resident = set(cache.resident_ids)
    cache.close()
    assert resident == {"a", "b", "c", "d"}
    assert len(loads) == 4
