import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs

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
