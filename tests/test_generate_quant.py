import mlx.core as mx
from mlx_lm.models.cache import KVCache

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate
from quant_helpers import load_quantized_model


def _full_load_greedy(model_dir, prompt, n):
    model, config = load_quantized_model(model_dir)
    cache = [KVCache() for _ in range(config["num_hidden_layers"])]
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :]
    out = []
    for _ in range(n):
        nxt = int(mx.argmax(logits, axis=-1).item())
        out.append(nxt)
        logits = model(mx.array([nxt])[None], cache=cache)[:, -1, :]
    return out


def test_quant_generate_matches_full_load(tiny_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-q4.nunspark"
    pack(tiny_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    prompt = [3, 7, 42, 1, 9]
    n = 12  # enough decode steps to exercise KV-cache reuse past the prompt
    ref = _full_load_greedy(tiny_quant_model_dir, prompt, n)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        got = generate(engine, prompt, max_tokens=n, temp=0.0)
    finally:
        engine.close()
    assert got == ref
