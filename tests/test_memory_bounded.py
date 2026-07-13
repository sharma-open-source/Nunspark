import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.piece_store import PieceStore
from nunspark.engine import StreamingEngine
from nunspark.generate import generate


def _layer_bytes(packed_dir, manifest):
    store = PieceStore(packed_dir, manifest)
    w = store.load("layer_000")
    mx.eval(list(w.values()))
    return sum(int(v.nbytes) for v in w.values())


def test_peak_bytes_bounded_by_budget(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    one = _layer_bytes(out, manifest)
    budget = one * 2  # room for ~2 layers

    engine = StreamingEngine(out, manifest, budget_bytes=budget)
    try:
        generate(engine, [3, 7, 1], max_tokens=5, temp=0.0)
        # Resident layer bytes never exceed budget by more than one in-flight layer.
        assert engine.cache.peak_bytes <= budget + 2 * one
    finally:
        engine.close()


def test_hot_set_reuses_layers_across_tokens(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    one = _layer_bytes(out, manifest)
    big_budget = one * (manifest.num_layers + 2)  # fits the whole model

    engine = StreamingEngine(out, manifest, budget_bytes=big_budget)
    try:
        # prefill + several decode steps: each of the 4 layers should load ONCE.
        generate(engine, [3, 7, 1], max_tokens=6, temp=0.0)
        # 4 layers, each a single disk load despite 7 forward passes.
        assert engine.cache.misses == manifest.num_layers
        assert engine.cache.hits > engine.cache.misses
    finally:
        engine.close()


def test_partial_budget_reduces_disk_reads(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    one = _layer_bytes(out, manifest)
    n = manifest.num_layers                    # tiny model has 4 layers

    def disk_reads(budget):
        # prefetch off so cache.misses == actual disk loads (no prefetch counting)
        engine = StreamingEngine(out, manifest, budget_bytes=budget, prefetch=False)
        try:
            generate(engine, [3, 7, 1], max_tokens=6, temp=0.0, prefetch=False)
            return engine.cache.misses
        finally:
            engine.close()

    one_layer = disk_reads(one)                # K=1: no cross-token reuse possible
    most = disk_reads(one * (n - 1))           # K=n-1: early-layer prefix stays resident
    # Under LRU these are equal (cyclic thrash even at K=n-1); MRU keeps the prefix.
    assert one_layer > n                       # K=1 truly thrashes (>= one reload/layer)
    assert most < one_layer
