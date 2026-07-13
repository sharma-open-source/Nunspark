import json

import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, speculative_generate, SpecStats


def _build_engine(model_dir, tmp_path):
    out = tmp_path / "q3.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


def _load_draft(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    model.load_weights(list(mx.load(str(model_dir / "model.safetensors")).items()))
    mx.eval(model.parameters())
    return model


def test_qwen3_self_draft_matches_greedy(tiny_qwen3_model_dir, tmp_path):
    prompt = [3, 7, 42, 1]
    engine = _build_engine(tiny_qwen3_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=12, temp=0.0)
        draft = _load_draft(tiny_qwen3_model_dir)
        stats = SpecStats()
        got = list(speculative_generate(
            engine, draft, prompt, max_tokens=12, num_draft_tokens=4, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 12
    # identical draft+target weights -> every draft token accepted -> M == K at the cap
    assert stats.multiplier >= 4.0
