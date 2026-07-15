import threading
import time

import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.piece_cache import PieceCache


def _arr_loader():
    # Mirrors test_piece_cache._arr_loader: 256x256 float32 pieces, records loads.
    loads = []
    side = 256

    def loader(pid):
        loads.append(pid)
        return {"w": mx.zeros((side, side), dtype=mx.float32)}

    return loader, loads, side * side * 4


def _E(i):
    return f"layer_000_expert_{i}"


def _wait(pred, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.001)
    return False


def test_warm_bulk_reads_only_non_resident_files_and_leaves_counters(tmp_path):
    # warm_bulk raw-reads the files of pids that need a disk read, skipping any
    # that are resident / staged / in-flight, and never touches hits/misses/
    # bytes_loaded or residency (pure page-cache populate).
    files = {}
    for i in (1, 2, 3, 4):
        f = tmp_path / f"{_E(i)}.bin"
        f.write_bytes(b"\x00" * (1 << 20))    # 1 MiB so _warm has real bytes to read
        files[_E(i)] = f
    warmed = []

    def pather(pid):
        warmed.append(pid)
        return files[pid]

    loader, loads, one = _arr_loader()
    cache = PieceCache(loader, budget_bytes=10**9, pather=pather)
    cache.get(_E(1))                          # make _E(1) resident (a demand miss)
    before = cache.stats()

    cache.warm_bulk([_E(1), _E(2), _E(3), _E(4)])
    # _E(1) is resident -> skipped; the other three get raw-read via pather.
    assert _wait(lambda: set(warmed) == {_E(2), _E(3), _E(4)})
    cache.close()

    assert _E(1) not in warmed                # resident pid never warmed
    after = cache.stats()
    assert after["hits"] == before["hits"]
    assert after["misses"] == before["misses"]
    assert after["bytes_loaded"] == before["bytes_loaded"]
    # only the single explicit get() ever materialized / became resident
    assert loads == [_E(1)]
    assert cache.resident_ids == [_E(1)]


def test_warm_bulk_skips_inflight(tmp_path):
    # A pid with an in-flight load (reserved by prefetch) is skipped: a read is
    # already happening, so warm_bulk must not double-read it.
    f = tmp_path / "x.bin"
    f.write_bytes(b"\x00" * (1 << 20))
    warmed = []
    gate = threading.Event()

    def loader(pid):
        gate.wait()                           # hold the load in flight
        return {"w": mx.zeros((4, 4))}

    def pather(pid):
        warmed.append(pid)
        return f

    cache = PieceCache(loader, budget_bytes=10**9, pather=pather)
    cache.prefetch([_E(9)])                    # reserves _E(9) in _inflight
    assert _wait(lambda: _E(9) in cache._inflight)
    cache.warm_bulk([_E(9)])                    # in-flight -> skipped
    time.sleep(0.05)
    assert warmed == []
    gate.set()
    cache.close()


def test_warm_bulk_noop_without_pather():
    loader, loads, _one = _arr_loader()
    cache = PieceCache(loader, budget_bytes=10**9)   # no pather
    cache.warm_bulk([_E(1), _E(2)])            # must be a silent no-op
    cache.close()
    assert loads == []
    assert cache._bulk_warm_pool is None       # pool never created


def test_warm_bulk_close_is_clean(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"\x00" * (1 << 20))
    loader, _loads, _one = _arr_loader()
    cache = PieceCache(loader, budget_bytes=10**9, pather=lambda pid: f)
    cache.warm_bulk([f"p{i}" for i in range(8)])
    cache.close()                              # must not hang or raise
    cache.close()                              # idempotent
    assert cache._bulk_warm_pool is None


def test_warm_bulk_missing_file_swallowed(tmp_path):
    loader, _loads, _one = _arr_loader()
    cache = PieceCache(
        loader, budget_bytes=10**9,
        pather=lambda pid: tmp_path / "does_not_exist.bin",
    )
    cache.warm_bulk([_E(1)])                    # bad path -> per-file exception swallowed
    cache.close()                              # no raise, no hang


# ---- engine-level: multi-token hook is a no-op on numerics; decode never calls it ----


def test_multitoken_forward_bit_identical_with_and_without_warm_bulk(
        tiny_qwen3_moe_quant_model_dir, tmp_path):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = mx.array([3, 7, 42, 1, 9])[None]   # multi-token -> hook fires

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got_hook = engine.forward(tokens)
        mx.eval(got_hook)
    finally:
        engine.close()

    engine2 = StreamingEngine(out, manifest, budget_bytes=10**9)
    engine2.cache.warm_bulk = lambda *a, **k: None   # neutralize the hook
    try:
        got_noop = engine2.forward(tokens)
        mx.eval(got_noop)
    finally:
        engine2.close()

    # warm_bulk is a pure page-cache populate -> must be numerically invisible.
    assert float(mx.max(mx.abs(got_hook - got_noop))) == 0.0


def test_single_token_forward_never_calls_warm_bulk(
        tiny_qwen3_moe_quant_model_dir, tmp_path):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    calls = []
    orig = engine.cache.warm_bulk
    def spy(pids):
        calls.append(list(pids))
        return orig(pids)
    engine.cache.warm_bulk = spy
    try:
        engine.forward(mx.array([[7]]))         # single token -> _cur_pass_multi False
    finally:
        engine.close()

    assert calls == []                          # decode must never bulk-warm
