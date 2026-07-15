import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, ngram_speculative_generate, SpecStats
from nunspark.ngram_drafter import NGramDrafter


# --- NGramDrafter unit tests -------------------------------------------------

def test_longest_suffix_preferred():
    # context ends in "9, 1, 2"; the trigram (9,1,2) occurred once earlier,
    # followed by 3,4,5. A shorter suffix (e.g. (1,2)) also occurs but the
    # longest match must win.
    ctx = [9, 1, 2, 3, 4, 5, 7, 1, 2, 8, 9, 1, 2]
    d = NGramDrafter(max_ngram=3, num_draft_tokens=4)
    # longest suffix (9,1,2) -> most recent prior occurrence is at index 0,
    # followed by 3,4,5,7.
    assert d.propose(ctx) == [3, 4, 5, 7]


def test_most_recent_occurrence_wins():
    # The bigram (1,2) appears twice before the suffix; the MOST RECENT prior
    # occurrence (index 5) should supply the proposal, not the earliest.
    ctx = [1, 2, 100, 101, 102, 1, 2, 200, 201, 1, 2]
    d = NGramDrafter(max_ngram=2, num_draft_tokens=2)
    # most recent prior (1,2) is at index 5, followed by 200, 201.
    assert d.propose(ctx) == [200, 201]


def test_falls_back_to_shorter_ngram():
    # The full trigram suffix has no prior match, but the unigram (5) does.
    ctx = [5, 42, 43, 44, 99, 98, 5]
    d = NGramDrafter(max_ngram=3, num_draft_tokens=2)
    # trigram (98,5) preceded by nothing matching; unigram (5) -> first 5 at
    # index 0 is followed by 42,43.
    assert d.propose(ctx) == [42, 43]


def test_no_match_returns_empty():
    # Every suffix token is unique -> no prior occurrence at any n.
    ctx = [10, 11, 12, 13, 14]
    d = NGramDrafter(max_ngram=3, num_draft_tokens=4)
    assert d.propose(ctx) == []


def test_k_truncation():
    # A long run of following tokens is truncated to K.
    ctx = [1, 2, 3, 4, 5, 6, 7, 8, 1]
    d = NGramDrafter(max_ngram=3, num_draft_tokens=2)
    # unigram (1) at index 0 followed by 2,3,4,5,...; K=2 -> [2, 3].
    assert d.propose(ctx) == [2, 3]


def test_too_short_context_returns_empty():
    d = NGramDrafter(max_ngram=3, num_draft_tokens=4)
    assert d.propose([]) == []
    assert d.propose([7]) == []


# --- end-to-end losslessness on a tiny packed model --------------------------

def _build_engine(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


def test_ngram_spec_matches_greedy(tiny_model_dir, tmp_path):
    # A repetitive prompt gives the n-gram drafter real matches to accept.
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=20, temp=0.0)
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=6)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=20, stats=stats))
    finally:
        engine.close()
    # Lossless: n-gram speculative output is bit-identical to greedy.
    assert got == ref
    assert len(got) == 20
    # Stats sanity: never accept more than proposed; passes account for tokens.
    assert stats.accepted_total <= stats.draft_tokens_proposed
    assert stats.accepted_offpath == 0          # lossless greedy
    assert stats.tokens_emitted == 20
    assert stats.target_passes >= 1


def test_ngram_spec_eos_stops(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=16, temp=0.0)
        eos = ref[4]
        expected = ref[: ref.index(eos) + 1]
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=4)
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=16, eos_id=eos))
    finally:
        engine.close()
    assert got == expected


def test_ngram_spec_gpt_oss_rotating_cache_matches_greedy(
    tiny_gpt_oss_model_dir, tmp_path
):
    # gpt-oss alternates sliding-window attention (RotatingKVCache, window 4
    # in this fixture) with full attention (KVCache). The 15-token prompt
    # rotates every sliding cache during prefill (offset 15 >> window 4) and
    # generation keeps rotating them. Regression for two defects:
    #   1. masks: PerLayer.build built ALL masks from layer 0's cache; layer 0
    #      is sliding, and RotatingKVCache.make_mask clamps offset to the
    #      window, so the "global" mask was too short for the full-attention
    #      layers' keys -> broadcast crash on the FIRST multi-token verify pass.
    #   2. rollback: RotatingKVCache.trim after rotation cannot restore evicted
    #      positions, so trim-based rollback silently desyncs. The verify pass
    #      now runs on ephemeral cache clones and commits only accepted tokens.
    prompt = [3, 7, 42, 1, 9] * 3
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=5)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=24, stats=stats))
    finally:
        engine.close()
    # Lossless past the rotation point: bit-identical to greedy.
    assert got == ref
    assert len(got) == 24
    assert stats.accepted_total <= stats.draft_tokens_proposed
    assert stats.accepted_offpath == 0


def test_ngram_spec_quantized_kv_matches_quantized_greedy(
    tiny_kvq_model_dir, tmp_path
):
    # With a quantized KV cache the reference is quantized greedy (same
    # quantization error on both sides); the ephemeral-verify/commit path must
    # reproduce it bit-identically — commit re-appends the RAW recorded rows
    # through the persistent QuantizedKVCache's own update_and_fetch.
    from nunspark.archspec import KVQuant
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    kvq = KVQuant(bits=8, group_size=32)
    engine = _build_engine(tiny_kvq_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=16, temp=0.0, kv_quant=kvq)
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=5)
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=16, kv_quant=kvq))
    finally:
        engine.close()
    assert got == ref


def test_ngram_spec_max_tokens_boundary(tiny_model_dir, tmp_path):
    # max_tokens not a multiple of any draft length -> must stop exactly.
    prompt = [3, 7, 42, 1, 3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=9, temp=0.0)
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=6)
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=9))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 9
