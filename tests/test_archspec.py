import mlx.core as mx

from nunspark.archspec import (
    ArchSpec,
    DEFAULT_RUNNER,
    DefaultLayerRunner,
    LayerContext,
    UniformCausal,
)


def test_archspec_defaults_are_all_none_except_required():
    spec = ArchSpec(args_cls=dict, block_factory=lambda args, i=0: None)
    assert spec.layer_key_fn is None
    assert spec.selective_moe is False
    assert spec.expert_attr == "switch_mlp"
    assert spec.router_attr == "gate"
    assert spec.num_experts is None
    assert spec.moe_route is None
    assert spec.embed_scale is None
    assert spec.final_norm is None
    assert spec.logit_transform is None
    assert spec.mask_plan is None
    assert spec.cache_plan is None
    assert spec.per_layer_inputs is None
    assert spec.kv_sharing is None
    assert spec.quant_predicate is None
    assert spec.resident_extras is None
    assert spec.layer_runner is DEFAULT_RUNNER


def test_uniform_causal_returns_same_mask_for_every_layer():
    # build() makes one causal mask; indexing by layer always yields that mask.
    h = mx.zeros((1, 5, 8))
    index = UniformCausal().build(h, kv=None)
    m0 = index[0]
    assert index[3] is m0          # identical object for any layer
    assert m0 is not None          # a real causal mask for a 5-token prefill


def test_default_runner_calls_slot_with_mask_and_cache():
    calls = {}

    class FakeSlot:
        def __call__(self, h, mask=None, cache=None):
            calls["mask"] = mask
            calls["cache"] = cache
            return h + 1

    out = DefaultLayerRunner().run(
        LayerContext(), FakeSlot(), 0, mx.array([10]), mask="M", cache="C"
    )
    assert calls == {"mask": "M", "cache": "C"}
    assert out.tolist() == [11]


from mlx_lm.models.cache import KVCache, RotatingKVCache

from nunspark.archspec import KV, Rotating, make_cache


def test_make_cache_kv_default():
    assert isinstance(make_cache(KV), KVCache)
    assert isinstance(make_cache(None), KVCache)   # None == default KV


def test_make_cache_rotating():
    c = make_cache(Rotating(window=512))
    assert isinstance(c, RotatingKVCache)
    assert c.max_size == 512
    assert c.keep == 0


from nunspark.archspec import PerLayer


def test_perlayer_distinct_masks_per_kind_prefill():
    # 5-token prefill, no cache. Layer kinds: [global, sliding(2), global].
    h = mx.zeros((1, 5, 8))
    plan = PerLayer(["global", ("sliding", 2), "global"])
    index = plan.build(h, kv=None)
    # global layers share ONE object; the sliding layer is a different object.
    assert index[0] is index[2]            # both global -> same mask object
    assert index[1] is not index[0]        # sliding -> distinct mask
    # For an uncached prefill, the global mask is the "causal" string sentinel,
    # while sliding(window<N) is a real windowed array -> genuinely different.
    assert index[0] == "causal"
    assert isinstance(index[1], mx.array)


def test_perlayer_matches_stock_create_attention_mask():
    from mlx_lm.models.base import create_attention_mask
    h = mx.zeros((1, 6, 8))
    plan = PerLayer(["global", ("sliding", 3)])
    index = plan.build(h, kv=None)
    # global mirrors the plain causal sentinel ("causal" string for an uncached prefill)
    assert index[0] == create_attention_mask(h, None)
    # sliding(window<N) mirrors the windowed array
    ref = create_attention_mask(h, None, window_size=3)
    assert isinstance(ref, mx.array)
    assert mx.array_equal(index[1], ref)


def test_perlayer_rejects_unknown_kind():
    import pytest
    h = mx.zeros((1, 4, 8))
    with pytest.raises(ValueError, match="unknown mask kind"):
        PerLayer(["bogus"]).build(h, kv=None)


from mlx_lm.models.cache import QuantizedKVCache

from nunspark.archspec import KVQuant


def test_kvquant_validates_bits():
    import pytest
    with pytest.raises(ValueError):
        KVQuant(bits=3)
    assert KVQuant(bits=8).group_size == 64   # default
    assert KVQuant(bits=4, group_size=32).group_size == 32


def test_make_cache_quantized_kv():
    q = KVQuant(bits=8, group_size=64)
    c = make_cache(KV, q)
    assert isinstance(c, QuantizedKVCache)
    assert c.bits == 8 and c.group_size == 64
    c4 = make_cache(None, KVQuant(bits=4))
    assert c4.bits == 4


def test_make_cache_rotating_ignores_quant():
    c = make_cache(Rotating(window=16), KVQuant(bits=8))
    assert isinstance(c, RotatingKVCache)       # sliding windows stay fp16
    assert not hasattr(c, "bits")


def test_make_cache_no_quant_unchanged():
    assert isinstance(make_cache(None, None), KVCache)
    assert isinstance(make_cache(KV), KVCache)  # quant arg optional
