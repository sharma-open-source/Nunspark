def test_mlx_imports():
    import mlx.core as mx

    a = mx.array([1.0, 2.0, 3.0])
    assert float(mx.sum(a)) == 6.0


def test_mlx_lm_llama_imports():
    # The MVP reuses mlx-lm's Llama layer math. Pin the import surface we depend on.
    from mlx_lm.models.llama import Model, ModelArgs  # noqa: F401
    from mlx_lm.models.cache import KVCache  # noqa: F401
    from mlx_lm.models.base import create_attention_mask  # noqa: F401
