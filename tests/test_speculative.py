import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.models.gpt_oss import Model as GptOssModel, ModelArgs as GptOssModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, speculative_generate, SpecStats


def _build_engine(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


def _load_mlx_model(model_dir, seed=None):
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    if seed is None:
        # same weights as the packed target -> full acceptance
        model.load_weights(list(mx.load(str(model_dir / "model.safetensors")).items()))
    else:
        # fresh random weights (same shape/vocab) -> a deliberately poor draft
        mx.random.seed(seed)
    mx.eval(model.parameters())
    return model


def test_self_draft_matches_greedy(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        draft = _load_mlx_model(tiny_model_dir)
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=12, num_draft_tokens=4, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 12
    # identical weights -> every draft token accepted -> M close to K+1
    assert stats.multiplier >= 4.0


def test_imperfect_draft_still_matches_greedy(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        bad_draft = _load_mlx_model(tiny_model_dir, seed=123)  # random -> forces rejections
        got = list(speculative_generate(
            engine, bad_draft, prompt, max_tokens=12, num_draft_tokens=4))
    finally:
        engine.close()
    assert got == ref


def test_k1_matches_greedy(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=10, temp=0.0)
        draft = _load_mlx_model(tiny_model_dir)
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=10, num_draft_tokens=1))
    finally:
        engine.close()
    assert got == ref


def test_max_tokens_boundary_not_multiple(tiny_model_dir, tmp_path):
    # max_tokens=7 with K=4 forces emission to stop mid-round.
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=7, temp=0.0)
        draft = _load_mlx_model(tiny_model_dir)
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=7, num_draft_tokens=4))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 7


def test_eos_stops_mid_block(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        eos = ref[3]                              # stop when this token is produced
        expected = ref[: ref.index(eos) + 1]      # up to and including its first occurrence
        draft = _load_mlx_model(tiny_model_dir)
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=12, num_draft_tokens=4, eos_id=eos))
    finally:
        engine.close()
    assert got == expected


# A dense tiny Llama sharing the gpt-oss fixture's 320-token vocab: a resident
# draft whose full-attention KVCache trims exactly (the supported draft shape),
# driving a sliding-window (RotatingKVCache) streaming target.
_DENSE_DRAFT_CONFIG = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 320,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


def _random_dense_draft(seed: int):
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(_DENSE_DRAFT_CONFIG))
    mx.eval(model.parameters())
    return model


def test_gpt_oss_rotating_cache_imperfect_draft_matches_greedy(
    tiny_gpt_oss_model_dir, tmp_path
):
    # gpt-oss alternates sliding-window (RotatingKVCache, window 4 here) and
    # full attention. The 15-token prompt rotates every sliding cache during
    # prefill; generation keeps rotating them. Regression for the community
    # crash on gpt-oss-120b: (1) the global mask used to be built from layer
    # 0's rotating cache and came out clamped to the window (broadcast crash
    # on the first K+1-token verify pass), and (2) kv.truncate rollback is
    # unsound on a rotated RotatingKVCache — the verify pass now runs on
    # ephemeral cache clones and commits only the accepted prefix. A random
    # dense draft (same vocab, unrelated weights) forces rejections, so
    # commits happen at varying m, including m=0.
    prompt = [3, 7, 42, 1, 9] * 3
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        draft = _random_dense_draft(seed=123)
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=24, num_draft_tokens=4, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 24
    assert stats.accepted_offpath == 0      # lossless default path
    assert stats.accepted_total <= stats.draft_tokens_proposed


def test_gpt_oss_rotating_cache_self_draft_matches_greedy(
    tiny_gpt_oss_model_dir, tmp_path
):
    # Same target, but the draft IS the target model (resident mlx_lm copy):
    # every draft token is accepted, so every round commits the full K+1 block
    # (> window 4) into the rotated sliding caches — the m == K commit path.
    config = json.loads((tiny_gpt_oss_model_dir / "config.json").read_text())
    draft = GptOssModel(GptOssModelArgs.from_dict(config))
    draft.load_weights(
        list(mx.load(str(tiny_gpt_oss_model_dir / "model.safetensors")).items()))
    mx.eval(draft.parameters())

    prompt = [3, 7, 42, 1, 9] * 3
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=24, num_draft_tokens=4, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert stats.accepted_total > 0


def test_specstats_deviation_rate():
    s = SpecStats()
    assert s.deviation_rate == 0.0          # nothing accepted yet -> safe, no ZeroDivision
    s.accepted_total = 10
    s.accepted_offpath = 3
    assert s.deviation_rate == 0.3
    # explicit zero accepted_total stays safe
    assert SpecStats(accepted_total=0, accepted_offpath=0).deviation_rate == 0.0


def test_accept_top_k1_matches_greedy(tiny_model_dir, tmp_path):
    # The default lossless path: explicit accept_top_k=1 == plain greedy generate().
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        draft = _load_mlx_model(tiny_model_dir)
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=12, num_draft_tokens=4,
            accept_top_k=1, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert stats.accepted_offpath == 0      # nothing diverged from the target argmax
    assert stats.deviation_rate == 0.0


def test_relaxed_topk_accepts_offpath_tokens(tiny_model_dir, tmp_path):
    # TINY_CONFIG vocab_size is 320; accept_top_k >= V accepts every draft token,
    # so a deliberately-bad (random) draft forces off-path acceptances deterministically.
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        bad = _load_mlx_model(tiny_model_dir, seed=123)   # random -> disagrees with target
        s1 = SpecStats()
        out_k1 = list(speculative_generate(
            engine, bad, prompt, max_tokens=12, num_draft_tokens=4,
            accept_top_k=1, stats=s1))
        sV = SpecStats()
        out_kV = list(speculative_generate(
            engine, bad, prompt, max_tokens=12, num_draft_tokens=4,
            accept_top_k=320, stats=sV))
    finally:
        engine.close()
    # lossless path unchanged: still matches greedy, no divergence recorded
    assert out_k1 == ref
    assert s1.accepted_offpath == 0 and s1.deviation_rate == 0.0
    # relaxed (accept-all) path: off-path tokens accepted -> fewer passes, lossy output
    assert sV.target_passes < s1.target_passes
    assert sV.accepted_offpath > 0
    assert 0.0 < sV.deviation_rate <= 1.0
    assert out_kV != ref
