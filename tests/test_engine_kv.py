import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.kv_store import KVStore


def _two_step_logits(engine, prompt, kv):
    """Prefill the prompt, greedily take one token, decode it; return step-2 logits.

    Exercises KV growth across steps (and, with a tiny budget, offload+reload).
    """
    logits = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]
    nxt = int(mx.argmax(logits, axis=-1).item())
    logits2 = engine.forward(mx.array([nxt])[None], kv=kv)[:, -1, :]
    mx.eval(logits2)
    return logits2


def test_kv_offload_matches_resident_fp16(tiny_model_dir, tmp_path):
    _offload_parity(tiny_model_dir, tmp_path, "fp16")


def test_tiny_budget_bounds_residency_with_prefetch(tiny_model_dir, tmp_path):
    """A tiny budget must cap KV residency even when prefetch=True.

    The prefetch worker inserts reloaded layers without evicting (it can't write
    to disk off the main thread), so without main-thread eviction-on-every-access
    a run of prefetch hits would leave the whole model resident — defeating the
    budget. With a budget of 1 byte, at most the protected layer (plus a possible
    in-flight one) may stay resident after a forward.
    """
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    prompt = [3, 7, 42, 1, 9]

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    kv = KVStore(tmp_path / "kv", budget_bytes=1, prefetch=True)
    try:
        _two_step_logits(engine, prompt, kv)   # prefill + one decode step
        assert len(kv.resident_ids) < manifest.num_layers   # NOT all layers resident
        assert len(kv.resident_ids) <= 2                     # budget=1 -> protected (+inflight)
    finally:
        kv.close()
        engine.close()


def _offload_parity(model_dir, tmp_path, tag):
    out = tmp_path / f"{tag}.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    prompt = [3, 7, 42, 1, 9]

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        big = KVStore(tmp_path / f"{tag}_big", budget_bytes=10**12)
        small = KVStore(tmp_path / f"{tag}_small", budget_bytes=1)
        try:
            ref = _two_step_logits(engine, prompt, big)
            got = _two_step_logits(engine, prompt, small)
            assert got.shape == ref.shape
            assert float(mx.max(mx.abs(got - ref))) == 0.0
        finally:
            big.close()
            small.close()
    finally:
        engine.close()


def test_kv_offload_matches_resident_quant(tiny_quant_model_dir, tmp_path):
    _offload_parity(tiny_quant_model_dir, tmp_path, "q4")


def test_kv_offload_matches_resident_quant_untied(tiny_quant_untied_model_dir, tmp_path):
    _offload_parity(tiny_quant_untied_model_dir, tmp_path, "q4-untied")
