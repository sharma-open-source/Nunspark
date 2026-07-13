import pytest
from mlx_lm.models import gpt_oss, gemma3, gemma3_text, gemma4, gemma4_text, llama, qwen3, qwen3_moe

from nunspark.archspec import ArchSpec, DEFAULT_RUNNER, PerLayer
from nunspark.architectures import get_architecture, supported_model_types, GEMMA4_RUNNER

# Minimal configs for constructing a block from each factory.
_LLAMA_CFG = {
    "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 4,
    "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5, "vocab_size": 320, "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
_QWEN3_CFG = {**_LLAMA_CFG, "model_type": "qwen3", "head_dim": 16,
              "max_position_embeddings": 2048}


def test_get_architecture_returns_archspec():
    spec = get_architecture("llama")
    assert isinstance(spec, ArchSpec)
    assert spec.args_cls is llama.ModelArgs
    assert spec.layer_key_fn is None
    assert spec.selective_moe is False
    assert spec.layer_runner is DEFAULT_RUNNER


def test_llama_block_factory_builds_llama_block():
    spec = get_architecture("llama")
    block = spec.block_factory(spec.args_cls.from_dict(_LLAMA_CFG), 0)
    assert isinstance(block, llama.TransformerBlock)


def test_qwen3_block_factory_builds_qwen3_block():
    spec = get_architecture("qwen3")
    block = spec.block_factory(spec.args_cls.from_dict(_QWEN3_CFG), 0)
    assert isinstance(block, qwen3.TransformerBlock)


def test_qwen3_moe_is_selective_with_layer_key_fn():
    spec = get_architecture("qwen3_moe")
    assert spec.args_cls is qwen3_moe.ModelArgs
    assert spec.selective_moe is True
    assert callable(spec.layer_key_fn)
    # defaults match its switch_mlp/gate weight layout
    assert spec.expert_attr == "switch_mlp"
    assert spec.router_attr == "gate"
    assert spec.num_experts is None
    assert spec.moe_route is None


_GPT_OSS_CFG = {
    "model_type": "gpt_oss", "hidden_size": 64, "num_hidden_layers": 4,
    "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "num_local_experts": 8, "num_experts_per_tok": 2,
    "sliding_window": 4,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    "rms_norm_eps": 1e-5, "vocab_size": 320, "rope_theta": 10000.0,
}


def test_gpt_oss_is_selective_with_moe_shape_overrides():
    spec = get_architecture("gpt_oss")
    assert spec.args_cls is gpt_oss.ModelArgs
    assert spec.selective_moe is True
    assert spec.layer_key_fn(spec.args_cls.from_dict(_GPT_OSS_CFG), 0) == "moe"
    assert spec.layer_key_fn(spec.args_cls.from_dict(_GPT_OSS_CFG), 3) == "moe"
    # gpt_oss's SwitchGLU/router live at different attribute names than qwen3_moe
    assert spec.expert_attr == "experts"
    assert spec.router_attr == "router"
    args = spec.args_cls.from_dict(_GPT_OSS_CFG)
    assert spec.num_experts(args) == args.num_local_experts
    assert callable(spec.moe_route)


def test_gpt_oss_block_factory_builds_transformer_block():
    spec = get_architecture("gpt_oss")
    block = spec.block_factory(spec.args_cls.from_dict(_GPT_OSS_CFG), 0)
    assert isinstance(block, gpt_oss.TransformerBlock)


def test_gpt_oss_mask_and_cache_plans_alternate_global_and_sliding():
    spec = get_architecture("gpt_oss")
    args = spec.args_cls.from_dict(_GPT_OSS_CFG)
    plan = spec.mask_plan(args)
    assert isinstance(plan, PerLayer)
    # mirrors _GPT_OSS_CFG["layer_types"]: sliding, full, sliding, full
    assert plan.kinds == [("sliding", 4), "global", ("sliding", 4), "global"]

    cache_kinds = spec.cache_plan(args)
    from nunspark.archspec import KV, Rotating, make_cache
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    for kind, lt in zip(cache_kinds, _GPT_OSS_CFG["layer_types"]):
        cache = make_cache(kind)
        if lt == "full_attention":
            assert kind == KV
            assert isinstance(cache, KVCache)
        else:
            assert kind == Rotating(4)
            assert isinstance(cache, RotatingKVCache)


def test_dense_drop_ins_have_no_layer_key_fn_and_are_not_selective():
    for mt in ("apertus", "ernie4_5", "glm", "glm4", "helium",
               "hunyuan_v1_dense", "internlm3", "mimo", "olmo2", "phi3",
               "seed_oss", "telechat3", "youtu_llm", "mistral"):
        spec = get_architecture(mt)
        assert spec.layer_key_fn is None
        assert spec.selective_moe is False


def test_mistral_aliases_the_llama_impl():
    spec = get_architecture("mistral")
    assert spec.args_cls is llama.ModelArgs
    block = spec.block_factory(spec.args_cls.from_dict(_LLAMA_CFG), 0)
    assert isinstance(block, llama.TransformerBlock)


def test_unknown_model_type_raises_valueerror():
    with pytest.raises(ValueError, match="unsupported model_type 'gpt2'"):
        get_architecture("gpt2")


def test_supported_model_types_is_sorted_list():
    assert supported_model_types() == [
        "apertus", "ernie4_5", "gemma3", "gemma3_text", "gemma4", "gemma4_assistant",
        "gemma4_text", "glm", "glm4", "gpt_oss", "helium", "hunyuan_v1_dense",
        "internlm3", "llama", "mimo", "mistral", "olmo2", "phi3", "qwen2",
        "qwen3", "qwen3_moe", "seed_oss", "telechat3", "youtu_llm",
    ]


def test_gemma3_has_gemma_specific_features():
    from mlx_lm.models import gemma3_text
    spec = get_architecture("gemma3")
    # Both gemma3 and gemma3_text use gemma3_text.ModelArgs (base gemma3 is multimodal-only)
    assert spec.args_cls is gemma3_text.ModelArgs
    assert spec.embed_scale is not None
    assert spec.final_norm is not None
    assert spec.mask_plan is not None
    assert spec.cache_plan is not None
    assert spec.layer_key_fn is not None
    assert spec.layer_runner is DEFAULT_RUNNER


def test_gemma3_text_same_as_gemma3():
    from mlx_lm.models import gemma3_text
    spec = get_architecture("gemma3_text")
    assert spec.args_cls is gemma3_text.ModelArgs
    assert spec.embed_scale is not None
    assert spec.final_norm is not None
    assert spec.mask_plan is not None
    assert spec.cache_plan is not None


def test_gemma4_has_special_runner_and_features():
    from mlx_lm.models import gemma4_text
    from nunspark.architectures import GEMMA4_RUNNER
    spec = get_architecture("gemma4")
    # Both gemma4 and gemma4_text use gemma4_text.ModelArgs (base gemma4 is multimodal-only)
    assert spec.args_cls is gemma4_text.ModelArgs
    assert spec.layer_runner is GEMMA4_RUNNER
    assert spec.kv_sharing is not None
    assert spec.per_layer_inputs is not None
    assert spec.quant_predicate is not None
    assert spec.resident_extras is not None
    assert spec.logit_transform is not None
    assert spec.mask_plan is not None
    assert spec.cache_plan is not None


def test_gemma4_text_same_as_gemma4():
    from mlx_lm.models import gemma4_text
    from nunspark.architectures import GEMMA4_RUNNER
    spec = get_architecture("gemma4_text")
    assert spec.args_cls is gemma4_text.ModelArgs
    assert spec.layer_runner is GEMMA4_RUNNER
    assert spec.kv_sharing is not None
    assert spec.per_layer_inputs is not None


def test_gemma4_assistant_is_registered():
    from nunspark import gemma4_assistant
    spec = get_architecture("gemma4_assistant")
    assert spec.args_cls is gemma4_assistant.ModelArgs
    assert spec.layer_key_fn is not None
    # Assistant is not a standard streaming model
    assert spec.layer_runner is DEFAULT_RUNNER  # Uses default runner but called differently


def test_gemma4_store_full_length_kv_tgemma4_pack_shape():
    """tgemma4-pack: 60 layers, num_kv_shared_layers=0, sliding_window_pattern=5
    -> layer_types ends ...sliding,sliding,sliding,sliding,full (idx 58, 59).
    The last layer of each distinct type in the non-shared prefix (= every
    layer here) stores its K/V for the assistant drafter."""
    from mlx_lm.models import gemma4_text
    from nunspark.architectures import _gemma4_store_full_length_kv

    args = gemma4_text.ModelArgs.from_dict({
        "model_type": "gemma4_text",
        "num_hidden_layers": 60,
        "num_kv_shared_layers": 0,
        "sliding_window_pattern": 5,
    })
    assert _gemma4_store_full_length_kv(args) == {
        58: "sliding_attention",
        59: "full_attention",
    }


_GEMMA4_TINY_CFG = {
    "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 2,
    "intermediate_size": 64, "num_attention_heads": 4, "head_dim": 8,
    "global_head_dim": 8, "num_key_value_heads": 2, "num_global_key_value_heads": 2,
    "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
    "sliding_window": 16, "sliding_window_pattern": 5,
    "vocab_size": 100, "vocab_size_per_layer_input": 100,
}


def test_gemma4_layer_runner_ignores_stale_shared_kv_for_non_consumer_layers():
    """Regression test for the cross-call state bug: a layer that is NOT a
    kv_sharing consumer (kv_sharing[layer_idx] is None) must always run with
    shared_kv=None and recompute its own K/V, even if `lctx.shared_kv` already
    holds a stale entry under that layer's own index from a prior pass.

    Before the fix, the runner looked up `lctx.shared_kv.get(layer_idx)` (own
    index) instead of via `kv_sharing[layer_idx]` (producer index), so a
    leftover entry from the previous `forward()` call was fed back as
    `shared_kv`, causing `Attention.__call__` to skip recomputing K/V for the
    new token entirely.
    """
    import mlx.core as mx
    from mlx_lm.models import gemma4_text
    from mlx_lm.models.cache import KVCache
    from nunspark.archspec import LayerContext
    from nunspark.architectures import _gemma4_kv_sharing, _gemma4_store_full_length_kv

    args = gemma4_text.ModelArgs.from_dict(_GEMMA4_TINY_CFG)
    kv_sharing = _gemma4_kv_sharing(args)
    assert kv_sharing == [None, None]   # no kv-shared layers in this config

    layer0 = gemma4_text.DecoderLayer(args, 0)

    mx.random.seed(0)
    h = mx.random.normal((1, 1, args.hidden_size))

    # Baseline: empty lctx.shared_kv -> layer recomputes its own K/V fresh.
    lctx_clean = LayerContext(
        kv_sharing=kv_sharing,
        store_full_length_kv=_gemma4_store_full_length_kv(args),
    )
    out_clean = GEMMA4_RUNNER.run(lctx_clean, layer0, 0, h, None, KVCache())

    # Stale: lctx.shared_kv[0] holds a bogus (k, v) from a "previous pass"
    # that was never cleared. Since kv_sharing[0] is None, this must be
    # ignored entirely.
    bogus_kv = (mx.ones((1, 2, 1, 8)), mx.ones((1, 2, 1, 8)) * -7)
    lctx_stale = LayerContext(
        kv_sharing=kv_sharing,
        store_full_length_kv=_gemma4_store_full_length_kv(args),
    )
    lctx_stale.shared_kv[0] = (bogus_kv, mx.array(123))
    out_stale = GEMMA4_RUNNER.run(lctx_stale, layer0, 0, h, None, KVCache())

    assert mx.allclose(out_clean, out_stale)


def test_gemma4_runner_dequantizes_quantized_shared_kv():
    """A producer layer running on a QuantizedKVCache returns quantized
    triples as kv_out; the runner must stash dequantized fp16 arrays into
    lctx.shared_kv and lctx.target_kv_states so KV-shared consumers and the
    MTP drafter see plain arrays, not (q, scales, biases) triples."""
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache
    from nunspark.architectures import Gemma4LayerRunner
    from nunspark.archspec import LayerContext

    cache = QuantizedKVCache(group_size=64, bits=8)
    k = mx.random.normal((1, 2, 5, 64))
    v = mx.random.normal((1, 2, 5, 64))
    qkv = cache.update_and_fetch(k, v)      # (k_triple, v_triple)
    assert isinstance(qkv[0], tuple)        # sanity: cache really quantized

    class FakeSlot:
        def __call__(self, h, mask=None, cache=None, shared_kv=None,
                     offset=None, per_layer_input=None):
            return h, qkv, mx.array(0)

    lctx = LayerContext(
        kv_sharing=[None, 0],                # layer 1 consumes layer 0
        kv_producers=frozenset([0]),
        store_full_length_kv={0: "full_attention"},
    )
    runner = Gemma4LayerRunner()
    h = mx.zeros((1, 5, 8))
    runner.run(lctx, FakeSlot(), 0, h, mask=None, cache=cache)

    (sk, sv), _offset = lctx.shared_kv[0]
    tk, tv = lctx.target_kv_states["full_attention"]
    for arr in (sk, sv, tk, tv):
        assert isinstance(arr, mx.array)     # dequantized, not a triple
        assert arr.shape == (1, 2, 5, 64)
    # values reconstruct the quantized representation exactly
    ref_k = mx.dequantize(*qkv[0], group_size=64, bits=8)
    assert float(mx.max(mx.abs(sk - ref_k))) == 0.0


def test_gemma4_runner_fp16_kv_out_stashed_without_copy():
    """fp16 path is untouched: a plain (k, v) array pair must be stashed as
    the SAME array objects (no dequant, no copy)."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from nunspark.architectures import Gemma4LayerRunner
    from nunspark.archspec import LayerContext

    k = mx.random.normal((1, 2, 5, 64))
    v = mx.random.normal((1, 2, 5, 64))

    class FakeSlot:
        def __call__(self, h, mask=None, cache=None, shared_kv=None,
                     offset=None, per_layer_input=None):
            return h, (k, v), mx.array(0)

    lctx = LayerContext(
        kv_sharing=[None, 0],
        kv_producers=frozenset([0]),
        store_full_length_kv={0: "full_attention"},
    )
    runner = Gemma4LayerRunner()
    h = mx.zeros((1, 5, 8))
    runner.run(lctx, FakeSlot(), 0, h, mask=None, cache=KVCache())

    (sk, sv), _offset = lctx.shared_kv[0]
    tk, tv = lctx.target_kv_states["full_attention"]
    assert sk is k and sv is v       # same objects, no copy
    assert tk is k and tv is v


def test_gpt_oss_rejects_quantized_kv():
    assert get_architecture("gpt_oss").supports_quantized_kv is False
    assert get_architecture("llama").supports_quantized_kv is True
    assert get_architecture("gemma4").supports_quantized_kv is True
