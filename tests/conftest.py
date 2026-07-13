import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.models.qwen3 import Model as Qwen3Model, ModelArgs as Qwen3ModelArgs
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel, ModelArgs as Qwen3MoeModelArgs
from mlx_lm.models.gpt_oss import Model as GptOssModel, ModelArgs as GptOssModelArgs


TINY_CONFIG = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 320,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


@pytest.fixture
def tiny_model_dir(tmp_path) -> Path:
    """Create a seeded random tiny Llama and save it as an mlx-lm model dir."""
    mx.random.seed(0)
    args = ModelArgs.from_dict(TINY_CONFIG)
    model = Model(args)
    mx.eval(model.parameters())  # materialize the random weights

    out = tmp_path / "tiny-llama"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


# mx.quantize requires the quantized axis divisible by its group_size and only
# supports group_size in {32, 64, 128}. tiny_model_dir's head_dim is 16
# (hidden=64 / heads=4), too small for any valid group_size, so KV-quant tests
# need their own model with head_dim=32 (hidden=128 / heads=4).
KVQ_CONFIG = {
    "model_type": "llama",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 320,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


@pytest.fixture
def tiny_kvq_model_dir(tmp_path) -> Path:
    """A tiny Llama with head_dim=32 (group_size=32 divides it for KV-quant)."""
    mx.random.seed(0)
    args = ModelArgs.from_dict(KVQ_CONFIG)
    model = Model(args)
    mx.eval(model.parameters())

    out = tmp_path / "tiny-kvq-llama"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(KVQ_CONFIG, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


_TEST_TOKENIZER_CORPUS = [
    "system: You are a helpful assistant.",
    "user: Hello there, how are you doing today?",
    "assistant: I am doing well, thank you for asking!",
    "user: What is the capital of France?",
    "assistant: The capital of France is Paris.",
    "user: Tell me a short story about a robot.",
    "assistant: Once upon a time, a small robot learned to paint pictures.",
]

_TEST_CHAT_TEMPLATE = (
    '{% for message in messages %}{{ message["role"] }}: {{ message["content"] }}\n'
    "{% endfor %}"
    '{% if add_generation_prompt %}assistant: {% endif %}'
)


def _write_test_tokenizer(out_dir: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
    )
    tok.train_from_iterator(_TEST_TOKENIZER_CORPUS, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>",
    )
    fast.chat_template = _TEST_CHAT_TEMPLATE
    fast.save_pretrained(out_dir)


@pytest.fixture
def tiny_model_dir_with_tokenizer(tiny_model_dir) -> Path:
    _write_test_tokenizer(tiny_model_dir)
    return tiny_model_dir


@pytest.fixture
def tiny_quant_model_dir(tmp_path) -> Path:
    """A 4-bit-quantized tiny Llama saved as an mlx-lm model dir.

    group_size=64 divides every quantized weight's last dim (hidden=64,
    intermediate=128, vocab inner-dim=64), so quantization is exact-round-trippable.
    """
    mx.random.seed(0)
    args = ModelArgs.from_dict(TINY_CONFIG)
    model = Model(args)
    mx.eval(model.parameters())            # materialize random fp32 weights
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())            # materialize the packed uint32 weights

    out = tmp_path / "tiny-llama-q4"
    out.mkdir()
    config = {**TINY_CONFIG, "quantization": {"group_size": 64, "bits": 4}}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


@pytest.fixture
def tiny_quant_untied_model_dir(tmp_path) -> Path:
    """A 4-bit-quantized tiny Llama with an UNTIED lm_head (separate output proj).

    Exercises the engine's QuantizedLinear head branch and the packer's
    not-tied branch, which the tied fixtures never hit.
    """
    mx.random.seed(0)
    config_src = {**TINY_CONFIG, "tie_word_embeddings": False}
    args = ModelArgs.from_dict(config_src)
    model = Model(args)
    mx.eval(model.parameters())
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())

    out = tmp_path / "tiny-llama-q4-untied"
    out.mkdir()
    config = {**config_src, "quantization": {"group_size": 64, "bits": 4}}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


TINY_QWEN3_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "vocab_size": 320,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


@pytest.fixture
def tiny_qwen3_model_dir(tmp_path) -> Path:
    """A seeded random tiny Qwen3 saved as an mlx-lm model dir."""
    mx.random.seed(0)
    model = Qwen3Model(Qwen3ModelArgs.from_dict(TINY_QWEN3_CONFIG))
    mx.eval(model.parameters())

    out = tmp_path / "tiny-qwen3"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(TINY_QWEN3_CONFIG, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


@pytest.fixture
def tiny_qwen3_quant_model_dir(tmp_path) -> Path:
    """A 4-bit-quantized tiny Qwen3 saved as an mlx-lm model dir.

    group_size=64 divides every quantized input dim (hidden=64, intermediate=128),
    so quantization round-trips exactly. q_norm/k_norm are RMSNorm and stay fp.
    """
    mx.random.seed(0)
    model = Qwen3Model(Qwen3ModelArgs.from_dict(TINY_QWEN3_CONFIG))
    mx.eval(model.parameters())
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())

    out = tmp_path / "tiny-qwen3-q4"
    out.mkdir()
    config = {**TINY_QWEN3_CONFIG, "quantization": {"group_size": 64, "bits": 4}}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


TINY_QWEN3_MOE_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "moe_intermediate_size": 64,    # divisible by group_size 64 -> 4-bit round-trips exactly
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-5,
    "vocab_size": 320,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


@pytest.fixture
def tiny_qwen3_moe_model_dir(tmp_path) -> Path:
    """A seeded random tiny Qwen3-MoE saved as an mlx-lm model dir."""
    mx.random.seed(0)
    model = Qwen3MoeModel(Qwen3MoeModelArgs.from_dict(TINY_QWEN3_MOE_CONFIG))
    mx.eval(model.parameters())

    out = tmp_path / "tiny-qwen3-moe"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(TINY_QWEN3_MOE_CONFIG, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


@pytest.fixture
def tiny_qwen3_moe_quant_model_dir(tmp_path) -> Path:
    """A 4-bit-quantized tiny Qwen3-MoE saved as an mlx-lm model dir.

    hidden=64 and moe_intermediate=64 both divide group_size 64, so every
    quantized weight (incl. the stacked-expert switch_mlp) round-trips exactly.
    """
    mx.random.seed(0)
    model = Qwen3MoeModel(Qwen3MoeModelArgs.from_dict(TINY_QWEN3_MOE_CONFIG))
    mx.eval(model.parameters())
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())

    out = tmp_path / "tiny-qwen3-moe-q4"
    out.mkdir()
    config = {**TINY_QWEN3_MOE_CONFIG, "quantization": {"group_size": 64, "bits": 4}}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


TINY_GPT_OSS_CONFIG = {
    "model_type": "gpt_oss",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 64,        # divisible by group_size 64 -> 4-bit round-trips exactly
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "sliding_window": 4,            # < token count, so the windowed mask is exercised for real
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    "rms_norm_eps": 1e-5,
    "vocab_size": 320,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


@pytest.fixture
def tiny_gpt_oss_model_dir(tmp_path) -> Path:
    """A seeded random tiny GPT-OSS saved as an mlx-lm model dir."""
    mx.random.seed(0)
    model = GptOssModel(GptOssModelArgs.from_dict(TINY_GPT_OSS_CONFIG))
    mx.eval(model.parameters())

    out = tmp_path / "tiny-gpt-oss"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(TINY_GPT_OSS_CONFIG, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out


@pytest.fixture
def tiny_gpt_oss_quant_model_dir(tmp_path) -> Path:
    """A 4-bit-quantized tiny GPT-OSS saved as an mlx-lm model dir.

    hidden=64 and intermediate=64 both divide group_size 64, so every quantized
    weight (incl. the stacked-expert SwitchGLU + its per-expert bias) round-trips
    exactly.
    """
    mx.random.seed(0)
    model = GptOssModel(GptOssModelArgs.from_dict(TINY_GPT_OSS_CONFIG))
    mx.eval(model.parameters())
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())

    out = tmp_path / "tiny-gpt-oss-q4"
    out.mkdir()
    config = {**TINY_GPT_OSS_CONFIG, "quantization": {"group_size": 64, "bits": 4}}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    return out
