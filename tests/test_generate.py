import json

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate


def _reference_greedy(model_dir, prompt, n):
    config = json.loads((model_dir / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    model = Model(args)
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    cache = [KVCache() for _ in range(args.num_hidden_layers)]
    tokens = list(prompt)
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    for _ in range(n):
        nxt = int(mx.argmax(logits, axis=-1).item())
        tokens.append(nxt)
        logits = model(mx.array([nxt])[None], cache=cache)[:, -1, :]
    return tokens[len(prompt):]


def test_streamed_greedy_matches_full_load(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    prompt = [3, 7, 42, 1]
    ref = _reference_greedy(tiny_model_dir, prompt, n=8)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = generate(engine, prompt, max_tokens=8, temp=0.0)
    finally:
        engine.close()

    assert got == ref
