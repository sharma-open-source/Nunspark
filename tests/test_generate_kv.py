import json

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.archspec import KVQuant
from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate
from nunspark.kv_store import KVStore


def _engine(model_dir, tmp_path, budget=10**9):
    out = tmp_path / "pk.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=budget)


def _reference_greedy(model_dir, prompt, n):
    """Full-load fp16 mlx-lm greedy decode (RAM KV), as ground truth.

    fp16 only — does not apply quantization. Assumes a single-shard
    model.safetensors (true for all conftest tiny-model fixtures).
    """
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


def test_tiny_kv_budget_matches_full_load_tokens(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    prompt = [3, 7, 42, 1]
    ref = _reference_greedy(tiny_model_dir, prompt, n=8)

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        # kv_budget=1 forces every layer to offload+reload each step.
        got = generate(engine, prompt, max_tokens=8, temp=0.0, kv_budget=1)
    finally:
        engine.close()

    assert got == ref          # offloaded KV decode is token-for-token identical


def test_generate_with_external_kv_store(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    prompt = [3, 7, 42, 1]

    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    kv = KVStore(tmp_path / "kv", budget_bytes=1)
    try:
        got = generate(engine, prompt, max_tokens=4, temp=0.0, kv=kv)
    finally:
        kv.close()
        engine.close()
    assert len(got) == 4
    assert kv.hits + kv.misses > 0   # confirms kv was actually used


def test_generate_with_kv_quant_runs_and_is_close(tiny_kvq_model_dir, tmp_path):
    engine = _engine(tiny_kvq_model_dir, tmp_path)
    try:
        prompt = [1, 2, 3, 4]
        out_fp16 = generate(engine, prompt, max_tokens=8)
        # group_size=32 (mx.quantize's minimum) divides this model's head_dim
        # (hidden=128 / heads=4 = 32).
        out_q8 = generate(engine, prompt, max_tokens=8, kv_quant=KVQuant(bits=8, group_size=32))
        assert len(out_q8) == 8
        # 8-bit KV is near-lossless: on the tiny model greedy outputs match.
        assert out_q8 == out_fp16
    finally:
        engine.close()


def test_generate_kv_quant_rejected_for_sinks_arch(tiny_model_dir, tmp_path, monkeypatch):
    engine = _engine(tiny_model_dir, tmp_path)
    try:
        # engine._spec is a frozen dataclass shared by the registry, so patch
        # the engine property seam instead of the spec instance/class field.
        monkeypatch.setattr(type(engine), "supports_quantized_kv", property(lambda self: False))
        with pytest.raises(ValueError, match="quantized KV"):
            generate(engine, [1, 2, 3], max_tokens=2, kv_quant=KVQuant(bits=8))
    finally:
        engine.close()
