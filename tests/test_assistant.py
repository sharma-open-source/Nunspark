"""Tests for gemma4_assistant integration."""

import pytest
from nunspark import gemma4_assistant
from nunspark.architectures import get_architecture, supported_model_types


def test_gemma4_assistant_model_args_structure():
    """Test that gemma4_assistant ModelArgs can be constructed."""
    config = {
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 2560,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "use_ordered_embeddings": True,
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
        },
    }

    args = gemma4_assistant.ModelArgs.from_dict(config)
    assert args.model_type == "gemma4_assistant"
    assert args.backbone_hidden_size == 2560
    assert args.text_config["num_hidden_layers"] == 4


def test_gemma4_assistant_in_supported_types():
    """Test that gemma4_assistant is registered."""
    assert "gemma4_assistant" in supported_model_types()


def test_gemma4_assistant_archspec():
    """Test that gemma4_assistant has a valid ArchSpec."""
    spec = get_architecture("gemma4_assistant")
    assert spec.args_cls is gemma4_assistant.ModelArgs
    assert spec.layer_key_fn is not None


def test_gemma4_assistant_has_quant_predicate():
    """Test that Model has quant_predicate property."""
    config = {
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 2560,
        "text_config": {
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
        },
    }

    model = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(config))
    assert model.quant_predicate is not None

    # Test that centroids are excluded from quantization
    predicate = model.quant_predicate
    assert predicate("masked_embedding.centroids", None) == False
    assert predicate("model.layers.0.mlp.gate_proj", None) == True


def test_assistant_drafter_import():
    """Test that AssistantDrafter can be imported."""
    from nunspark.assistant import AssistantDrafter
    assert AssistantDrafter is not None


def test_assistant_drafter_computes_in_weight_dtype():
    """Regression test: the drafter must run in its own weight dtype even when
    the target engine hands it fp32 inputs.

    Without the cast at the AssistantDrafter boundary, MLX type-promotes every
    assistant matmul to fp32 — including the (vocab, hidden) tied-embedding
    logits matmul, which re-casts the whole table per draft step. At 31B scale
    that was ~1GB of transient allocation per step (~17GB per 16-token round)
    and a ~1000x drafting slowdown.
    """
    import mlx.core as mx
    from nunspark.assistant import AssistantDrafter

    config = {
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 64,
        "use_ordered_embeddings": False,
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "layer_types": ["sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
        },
    }
    model = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    drafter = AssistantDrafter(model)

    # fp32 inputs, exactly what the StreamingEngine produces.
    target_embed = mx.zeros((1, 1, 64), dtype=mx.float32)
    target_last_hidden = mx.zeros((1, 1, 64), dtype=mx.float32)
    kv = {
        "sliding_attention": (mx.zeros((1, 2, 3, 8), dtype=mx.float32),
                              mx.zeros((1, 2, 3, 8), dtype=mx.float32)),
        "full_attention": (mx.zeros((1, 2, 3, 8), dtype=mx.float32),
                           mx.zeros((1, 2, 3, 8), dtype=mx.float32)),
    }

    last_hidden, logits = drafter.draft_step(target_embed, target_last_hidden, kv)
    mx.eval(last_hidden, logits)
    assert last_hidden.dtype == mx.bfloat16
    assert logits.dtype == mx.bfloat16


def test_draft_speculative_re_embeds_each_drafted_token():
    """Mirrors HF's candidate_generator loop: step i+1's input embedding must
    be embed_fn(token drafted at step i), not the bootstrap token's embedding
    repeated every step."""
    import mlx.core as mx
    from nunspark.assistant import AssistantDrafter

    config = {
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 64,
        "use_ordered_embeddings": False,
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "layer_types": ["sliding_attention", "full_attention"],
        },
    }
    mx.random.seed(0)
    model = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    drafter = AssistantDrafter(model)

    target_embed = mx.random.normal((1, 1, 64))
    target_last_hidden = mx.random.normal((1, 1, 64))
    kv = {
        "sliding_attention": (mx.random.normal((1, 2, 3, 8)),
                              mx.random.normal((1, 2, 3, 8))),
        "full_attention": (mx.random.normal((1, 2, 3, 8)),
                           mx.random.normal((1, 2, 3, 8))),
    }

    embedded_tokens = []

    def embed_fn(tokens):
        embedded_tokens.append(int(tokens[0, 0]))
        return mx.random.normal((1, 1, 64))

    draft_tokens, _, _ = drafter.draft_speculative(
        target_embed, target_last_hidden, kv,
        max_draft_tokens=4, embed_fn=embed_fn,
    )
    mx.eval(draft_tokens)

    # embed_fn is called once per drafted token, with that step's draft.
    assert embedded_tokens == [int(t) for t in draft_tokens[0].tolist()]
