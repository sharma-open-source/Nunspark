"""Chunked prefill: the prompt is prefilled in fixed-size windows (bounding peak
activation memory) instead of one giant forward. Windowing must be LOSSLESS —
the cache offset supplies the correct positions for later windows, exactly like
the processed_tokens prefix-reuse path — so a tiny chunk must produce the
IDENTICAL token sequence as a single-pass prefill for every architecture,
including the gpt-oss sliding-window (RotatingKVCache) fixture where multi-token
appends past the window were a real bug class.
"""
import json

import mlx.core as mx
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel, ModelArgs as Qwen3MoeModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, stream_generate, ngram_speculative_generate
from nunspark.ngram_drafter import NGramDrafter


def _build_engine(model_dir, tmp_path):
    out = tmp_path / "packed.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


# ---- bit-identity: tiny chunk == single-pass, every architecture ----

def _assert_generate_chunk_identical(engine, prompt, max_tokens=12):
    # A single window (chunk >> len(prompt)) vs many tiny windows over the same
    # prompt must yield the exact same greedy sequence.
    single = generate(engine, prompt, max_tokens=max_tokens, temp=0.0,
                      prefill_chunk=10_000)
    chunked = generate(engine, prompt, max_tokens=max_tokens, temp=0.0,
                       prefill_chunk=3)
    assert chunked == single
    return single


def test_chunked_prefill_bit_identical_dense(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 9, 15, 2, 8, 4, 11]   # > 3 chunks of size 3
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        _assert_generate_chunk_identical(engine, prompt)
    finally:
        engine.close()


def test_chunked_prefill_bit_identical_quant(tiny_quant_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 9, 15, 2, 8, 4, 11]
    engine = _build_engine(tiny_quant_model_dir, tmp_path)
    try:
        _assert_generate_chunk_identical(engine, prompt)
    finally:
        engine.close()


def test_chunked_prefill_bit_identical_moe(tiny_qwen3_moe_quant_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 9, 15, 2, 8, 4, 11]
    engine = _build_engine(tiny_qwen3_moe_quant_model_dir, tmp_path)
    try:
        _assert_generate_chunk_identical(engine, prompt)
    finally:
        engine.close()


def test_chunked_prefill_bit_identical_gpt_oss_sliding_window(
    tiny_gpt_oss_model_dir, tmp_path
):
    # sliding_window=4; a 15-token prompt rotates every RotatingKVCache during
    # prefill, so tiny windows (size 3) exercise multi-token appends both up to
    # and past the window boundary. Must stay bit-identical to single-pass.
    prompt = [3, 7, 42, 1, 9] * 3
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)
    try:
        _assert_generate_chunk_identical(engine, prompt, max_tokens=16)
    finally:
        engine.close()


def test_chunked_prefill_stream_matches_generate(tiny_model_dir, tmp_path):
    # stream_generate with a tiny chunk must equal generate() single-pass.
    prompt = [3, 7, 42, 1, 9, 15, 2, 8, 4, 11]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0, prefill_chunk=10_000)
        got = list(stream_generate(engine, prompt, max_tokens=12, temp=0.0,
                                   prefill_chunk=3))
    finally:
        engine.close()
    assert got == ref


# ---- spec path composes with chunked prefill ----

def test_ngram_spec_chunked_prefill_matches_greedy(tiny_qwen3_moe_quant_model_dir, tmp_path):
    # ngram spec is lossless greedy; a tiny prefill_chunk must not change that.
    prompt = [3, 7, 42, 1, 9, 15, 2, 8, 4, 11]
    engine = _build_engine(tiny_qwen3_moe_quant_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=16, temp=0.0, prefill_chunk=10_000)
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=4)
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=16, prefill_chunk=3))
    finally:
        engine.close()
    assert got == ref


# ---- processed_tokens (prompt-prefix reuse) path ----

def test_chunked_prefill_processed_tokens(tiny_model_dir, tmp_path):
    # A borrowed kv already holding a prefix + chunked prefill of the suffix must
    # equal the unchunked suffix prefill. Both share one warmed prefix so only
    # the suffix-prefill path (the thing chunking touches) differs.
    import tempfile
    from nunspark.kv_store import KVStore

    prompt = [3, 7, 42, 1, 9, 15, 2, 8, 4, 11]
    processed = 4
    engine = _build_engine(tiny_model_dir, tmp_path)

    def _run(chunk):
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        kv = KVStore(tmp.name, budget_bytes=10**12, prefetch=True,
                     cache_kinds=engine.cache_kinds, kv_quant=None)
        try:
            # Warm the prefix into the borrowed kv exactly as a caller would.
            engine.forward(mx.array(prompt[:processed])[None], kv=kv)
            mx.eval(kv.get(0))
            return list(stream_generate(
                engine, prompt, max_tokens=12, temp=0.0, kv=kv,
                processed_tokens=processed, prefill_chunk=chunk))
        finally:
            kv.close()
            tmp.cleanup()

    try:
        big = _run(10_000)
        small = _run(3)
    finally:
        engine.close()
    assert small == big


# ---- edge: prompt shorter than one chunk == single forward ----

def test_chunked_prefill_short_prompt_single_forward(tiny_model_dir, tmp_path):
    # A prompt shorter than the chunk must behave identically to before (one
    # forward, no windowing effect).
    prompt = [3, 7, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        default = generate(engine, prompt, max_tokens=8, temp=0.0)
        big = generate(engine, prompt, max_tokens=8, temp=0.0, prefill_chunk=10_000)
    finally:
        engine.close()
    assert default == big
