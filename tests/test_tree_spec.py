# tests/test_tree_spec.py
import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate
from nunspark.kv_store import KVStore
from nunspark.tree_shape import TreeShape
from nunspark.tree_spec import tree_speculative_generate


def _engine(model_dir, tmp_path):
    out = tmp_path / "pk.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


def _draft(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    model.load_weights(list(mx.load(str(model_dir / "model.safetensors")).items()))
    mx.eval(model.parameters())
    return model


def test_temp0_tree_matches_greedy(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        draft = _draft(tiny_model_dir)
        got = list(tree_speculative_generate(
            engine, draft, prompt, shape=TreeShape([2, 2, 2]),
            max_tokens=12, temp=0.0))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 12


def test_temp0_imperfect_draft_still_greedy(tiny_model_dir, tmp_path):
    # A poor draft (random weights, same shape) must STILL yield exact greedy output.
    prompt = [3, 7, 42, 1]
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=10, temp=0.0)
        mx.random.seed(123)
        bad = Model(ModelArgs.from_dict(json.loads((tiny_model_dir / "config.json").read_text())))
        mx.eval(bad.parameters())
        got = list(tree_speculative_generate(
            engine, bad, prompt, shape=TreeShape([2, 2]), max_tokens=10, temp=0.0))
    finally:
        engine.close()
    assert got == ref


def test_eos_and_max_tokens(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        draft = _draft(tiny_model_dir)
        eos = ref[4]
        expected = ref[: ref.index(eos) + 1]
        got = list(tree_speculative_generate(
            engine, draft, prompt, shape=TreeShape([2, 2]),
            max_tokens=12, temp=0.0, eos_id=eos))
        # max_tokens truncation (no eos)
        got7 = list(tree_speculative_generate(
            engine, draft, prompt, shape=TreeShape([2, 2]),
            max_tokens=7, temp=0.0))
    finally:
        engine.close()
    assert got == expected
    assert got7 == ref[:7]


def test_temp_pos_distribution_matches_direct_target(tiny_model_dir, tmp_path):
    # First sampled token's distribution must match direct target sampling.
    # NOTE: max_tokens=1 exits before the tree rejection loop runs, so this exercises
    # only the prefill _sample_token. The rigorous rejection-sampling distribution test
    # (N=40000, TV<0.02) lives in tests/test_tree_sampling.py.
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        prompt = [3, 7, 42, 1]
        # target distribution for the first token, from a single streamed forward
        kv = KVStore(tmp_path / "ref_kv", budget_bytes=10**12)
        try:
            tl = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]
        finally:
            kv.close()
        p_target = mx.softmax(tl[0] / 0.8)

        draft = _draft(tiny_model_dir)
        mx.random.seed(0)
        vocab = p_target.shape[0]
        N, counts = 6000, [0] * vocab
        for _ in range(N):
            tok = next(tree_speculative_generate(
                engine, draft, prompt, shape=TreeShape([2, 2]),
                max_tokens=1, temp=0.8))
            counts[tok] += 1
        emp = mx.array([x / N for x in counts])
        tv = 0.5 * float(mx.sum(mx.abs(emp - p_target)).item())
        # TV threshold is set above the natural sampling noise floor (~0.073) for this
        # tiny near-uniform 320-vocab model with N=6000 draws; ideal direct sampling
        # itself yields ~0.0725, so 0.05 is unreachable. 0.10 gives ample headroom.
        assert tv < 0.10, f"TV={tv:.4f}"
    finally:
        engine.close()
