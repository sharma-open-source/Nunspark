from __future__ import annotations

import mlx.core as mx
from mlx_lm.models import (
    apertus,
    ernie4_5,
    glm,
    glm4,
    gpt_oss,
    helium,
    hunyuan_v1_dense,
    internlm3,
    llama,
    mimo,
    olmo2,
    phi3,
    qwen2,
    qwen3,
    qwen3_moe,
    seed_oss,
    telechat3,
    youtu_llm,
    gemma4,
    gemma4_text,
    gemma3,
    gemma3_text,
)

from . import gemma4_assistant

from .archspec import (
    ArchSpec,
    DEFAULT_RUNNER,
    KV,
    LayerContext,
    LayerRunner,
    PerLayer,
    Rotating,
)

# Why not every mlx_lm architecture?
# ----------------------------------
# StreamingEngine.forward (engine.py) re-implements the model's *outer* loop so
# it can stream one layer's weights at a time. That reimplementation bakes in a
# fixed contract, and only architectures that satisfy ALL of it are registrable:
#
#   * one global full-attention mask, built once from layer 0 (no sliding-window
#     or hybrid attention; no per-layer mask variation),
#   * the stock mlx_lm KVCache (no Mamba/SSM/linear-attention/MLA state caches),
#   * a plain nn.RMSNorm final norm (no LayerNorm, no Gemma (1 + weight) norm),
#   * no embedding scaling and no logit softcapping,
#   * a decoder block callable as block(h, mask=, cache=) whose weight tree
#     matches the checkpoint's per-layer key layout.
#
# Every type below was parity-checked against its stock mlx_lm Model.__call__
# (random weights, fresh nn.Embedding/nn.RMSNorm/block loaded from the same
# tensors): all produced bit-identical logits (max|Δ| = 0). Architectures that
# break the contract are deliberately absent and need engine work, not just a
# registry line:
#   * Gemma family       -> embed scaling + (1 + weight) norm (+ sliding/softcap)
#   * granite / minicpm  -> embedding multipliers
#   * stablelm/starcoder2-> LayerNorm final norm
#   * qwen3_next, mamba,  -> linear-attention / recurrent caches (no KVCache)
#     rwkv7, plamo2, ...
#   * deepseek/dbrx/MLA   -> MLA cache and/or non-qwen3 MoE weight layout
#   * *_vl / multimodal   -> vision towers the streamer doesn't run


def _qwen3_moe_layer_key(args, layer_idx: int) -> str:
    """Return 'moe' or 'dense' for a given Qwen3MoE layer index."""
    if (
        layer_idx not in args.mlp_only_layers
        and args.num_experts > 0
        and (layer_idx + 1) % args.decoder_sparse_step == 0
    ):
        return "moe"
    return "dense"


def _gpt_oss_layer_kinds(args) -> list:
    """Per-layer ('global' | ('sliding', window)) from args.layer_types, mirroring
    GptOssMoeModel's own fallback alternation when layer_types is unset."""
    layer_types = args.layer_types or [
        "sliding_attention", "full_attention",
    ] * (args.num_hidden_layers // 2)
    return [
        "global" if lt == "full_attention" else ("sliding", args.sliding_window)
        for lt in layer_types
    ]


def _gpt_oss_mask_plan(args) -> PerLayer:
    return PerLayer(_gpt_oss_layer_kinds(args))


def _gpt_oss_cache_plan(args) -> list:
    return [
        KV if kind == "global" else Rotating(kind[1])
        for kind in _gpt_oss_layer_kinds(args)
    ]


def _gpt_oss_layer_key(args, layer_idx: int) -> str:
    """Every gpt_oss layer carries an MLPBlock (no dense/moe split, unlike qwen3_moe);
    a constant key keeps one TransformerBlock slot for the whole model."""
    return "moe"


def _gpt_oss_route(args, gate_logits):
    """gpt_oss MLPBlock routing: top-k BEFORE softmax (softmax is over the
    selected logits only), unlike qwen3_moe's softmax-then-top-k. Mirrors
    mlx_topk + mx.softmax(experts, precise=True) from MLPBlock.__call__."""
    k = args.num_experts_per_tok
    inds = mx.argpartition(gate_logits, kth=-k, axis=-1)[..., -k:]
    selected = mx.take_along_axis(gate_logits, inds, axis=-1)
    scores = mx.softmax(selected, axis=-1, precise=True)
    return inds, scores


# --- gemma3 / gemma3_text helpers -------------------------------------------

def _gemma3_embed_scale(args):
    """√hidden scaling, matching gemma3's exact dtype cast."""
    import mlx.core as mx
    scale = args.hidden_size**0.5
    return mx.array(scale, mx.bfloat16).astype(args.dtype if hasattr(args, 'dtype') else mx.float16)


def _gemma3_final_norm(args):
    """Gemma (1+weight) RMSNorm for final norm."""
    import mlx.nn as nn
    return nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


def _gemma3_layer_kinds(args) -> list:
    """Per-layer ('global' | ('sliding', window)) from sliding_window_pattern."""
    kinds = []
    for idx in range(args.num_hidden_layers):
        if (idx + 1) % args.sliding_window_pattern == 0:
            kinds.append("global")
        else:
            kinds.append(("sliding", args.sliding_window))
    return kinds


def _gemma3_mask_plan(args):
    return PerLayer(_gemma3_layer_kinds(args))


def _gemma3_cache_plan(args):
    return [
        KV if kind == "global" else Rotating(kind[1])
        for kind in _gemma3_layer_kinds(args)
    ]


def _gemma3_layer_key(args, layer_idx: int) -> str:
    """Return 'global' or 'sliding' to select RoPE base."""
    if (layer_idx + 1) % args.sliding_window_pattern == 0:
        return "global"
    return "sliding"


# --- gemma4 / gemma4_text helpers -------------------------------------------

def _gemma4_layer_kinds(args) -> list:
    """Per-layer mask kinds from args.layer_types."""
    kinds = []
    for layer_type in args.layer_types:
        if layer_type == "full_attention":
            kinds.append("global")
        elif layer_type == "sliding_attention":
            kinds.append(("sliding", args.sliding_window))
        else:
            kinds.append("global")  # fallback
    return kinds


def _gemma4_mask_plan(args):
    return PerLayer(_gemma4_layer_kinds(args))


def _gemma4_cache_plan(args):
    return [
        KV if kind == "global" else Rotating(kind[1])
        for kind in _gemma4_layer_kinds(args)
    ]


def _gemma4_layer_key(args, layer_idx: int) -> tuple:
    """Return (layer_type, has_kv, enable_moe) for variant slot selection."""
    layer_type = args.layer_types[layer_idx]
    has_kv = layer_idx < (args.num_hidden_layers - args.num_kv_shared_layers)
    enable_moe = args.enable_moe_block and layer_type == "full_attention"
    return (layer_type, has_kv, enable_moe)


def _gemma4_kv_sharing(args):
    """Return list mapping consumer layers to producer layers.

    For num_kv_shared_layers=20 (default), layers 15-34 share from layers 0-14:
    consumer at layer i maps to producer at (i - num_kv_shared_layers).
    """
    sharing = [None] * args.num_hidden_layers
    first_shared_idx = args.num_hidden_layers - args.num_kv_shared_layers
    for consumer_idx in range(first_shared_idx, args.num_hidden_layers):
        producer_idx = consumer_idx - args.num_kv_shared_layers
        sharing[consumer_idx] = producer_idx
    return sharing


def _gemma4_store_full_length_kv(args) -> dict:
    """Return {layer_idx: layer_type} for layers whose K/V the assistant
    drafter cross-attends to.

    Mirrors HF's `store_full_length_kv`: for each distinct layer_type within
    `layer_types[:first_shared_idx]` (the non-KV-shared prefix), the LAST
    layer of that type stores its full-length post-RoPE K/V into
    `shared_kv_states[layer_type]`. For num_kv_shared_layers=0 this prefix is
    every layer, so it's simply the last "full_attention" and last
    "sliding_attention" layer overall.
    """
    first_shared_idx = args.num_hidden_layers - args.num_kv_shared_layers
    prev_layers = args.layer_types[:first_shared_idx]
    result = {}
    for layer_type in set(prev_layers):
        last_idx = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
        result[last_idx] = layer_type
    return result


def _gemma4_per_layer_inputs(args):
    """Return per-layer input spec if the model uses per-layer inputs (2B/4B)."""
    if not args.hidden_size_per_layer_input:
        return None
    return {
        "vocab_size": args.vocab_size_per_layer_input,
        "hidden_size": args.hidden_size_per_layer_input,
    }


def _gemma4_quant_predicate(args):
    """8-bit/group64 for MoE router and MLP projections; default (4-bit) elsewhere.

    Mirrors mlx_lm's stock gemma4_text.quant_predicate (router.proj at 8-bit) and
    extends it to the mlp projections, which NunSpark-packed gemma4 checkpoints
    quantize at 8-bit/group64 across all layers. Returning ``True`` for everything
    else lets nn.quantize fall back to the slot's global group_size/bits, so every
    Linear ends up with the weight/scales/biases parameters the streamed pieces need.
    """
    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj")):
            return {"group_size": 64, "bits": 8}
        return True
    return predicate


def _gemma4_resident_extras(args):
    """Extra hot-set piece IDs for per-layer input tensors."""
    if not args.hidden_size_per_layer_input:
        return None
    return ["embed_tokens_per_layer", "per_layer_model_projection", "per_layer_projection_norm"]


def _gemma4_logit_transform(args):
    """tanh(x/30)*30 softcapping for final logits."""
    import mlx.core as mx
    softcap = args.final_logit_softcapping
    def transform(logits):
        return mx.tanh(logits / softcap) * softcap
    return transform


def _dequantize_kv_out(kv_out, cache):
    """gemma4 layers return cache.update_and_fetch's result; under a
    QuantizedKVCache that's a pair of quantized triples. Dequantize once at
    the stash seam so KV-shared consumers and the MTP drafter see plain
    arrays (same affine dequant params — group_size/bits/scales/biases — as
    the quantized-SDPA path, so the reconstruction matches by construction).

    Output dtype follows scales.dtype, which update_and_fetch sets from the
    original k/v dtype at quantization time — i.e. the model's compute dtype,
    so the dequantized arrays concat cleanly with consumers' fresh k/v."""
    k, v = kv_out
    if isinstance(k, tuple):    # structural check: kv_out shape, not cache type
        k = mx.dequantize(*k, group_size=cache.group_size, bits=cache.bits)
        v = mx.dequantize(*v, group_size=cache.group_size, bits=cache.bits)
    return (k, v)


class Gemma4LayerRunner:
    """Layer runner for gemma4 that handles (h, shared_kv, offset) return values.

    gemma4's DecoderLayer returns (h, shared_kv, offset) instead of just h.
    Per forward pass, each layer's (kv, offset) is stashed in
    `lctx.shared_kv[layer_idx]` (cleared by the engine at the start of every
    `forward()` call). A KV-sharing consumer layer looks up its producer's
    entry via `lctx.kv_sharing[layer_idx]`; a non-consumer layer always gets
    `shared_kv=None, offset=None` and computes its own k/v fresh, mirroring
    mlx_lm's `Gemma4TextModel.__call__` `intermediates`/`previous_kvs` scheme.

    Layers named in `lctx.store_full_length_kv` additionally stash their
    `(k, v)` into `lctx.target_kv_states[layer_type]` for the MTP assistant
    drafter.
    """

    def run(self, lctx: LayerContext, slot, layer_idx: int,
            h: mx.array, mask, cache) -> mx.array:
        # Prepare per-layer input if available
        per_layer_input = None
        if lctx.per_layer_inputs is not None and layer_idx < len(lctx.per_layer_inputs):
            per_layer_input = lctx.per_layer_inputs[layer_idx]

        # Get shared_kv/offset from this layer's producer, if it's a consumer.
        producer_idx = lctx.kv_sharing[layer_idx] if lctx.kv_sharing else None
        shared_kv, offset = lctx.shared_kv.get(producer_idx, (None, None)) \
            if producer_idx is not None else (None, None)

        # Call the layer (returns h, shared_kv, offset)
        result = slot(h, mask=mask, cache=cache, per_layer_input=per_layer_input,
                      shared_kv=shared_kv, offset=offset)

        if isinstance(result, tuple) and len(result) == 3:
            h_out, kv_out, offset_out = result
            # Stash (k, v) only when something downstream reads it: a KV-shared
            # consumer layer (kv_producers) or the MTP drafter
            # (store_full_length_kv). Stashing unconditionally would pin a
            # materialized copy of every layer's full KV until the next forward.
            is_producer = bool(lctx.kv_producers) and layer_idx in lctx.kv_producers
            is_target = (bool(lctx.store_full_length_kv)
                         and layer_idx in lctx.store_full_length_kv)
            if is_producer or is_target:
                # Under a QuantizedKVCache, kv_out is a pair of quantized
                # triples; dequantize ONCE here so both consumer paths (KV-shared
                # layers and the MTP drafter) see plain arrays. fp16 caches return
                # the (k, v) arrays unchanged (same objects, no copy).
                kv_plain = _dequantize_kv_out(kv_out, cache)
                if is_producer:
                    lctx.shared_kv[layer_idx] = (kv_plain, offset_out)
                if is_target:
                    lctx.target_kv_states[lctx.store_full_length_kv[layer_idx]] = kv_plain
            return h_out
        else:
            # Fallback for layers that return just h
            return result


GEMMA4_RUNNER = Gemma4LayerRunner()


def _dense(args_cls: type, block_cls: type) -> ArchSpec:
    """ArchSpec for a llama-style dense arch: one block class, uniform layers,
    all forward hooks defaulted (no embed scale, plain RMSNorm, single mask)."""
    return ArchSpec(args_cls=args_cls, block_factory=lambda args, _=0: block_cls(args))


# model_type -> ArchSpec(args_cls, block_factory(args, layer_idx), layer_key_fn, ...)
# block_factory builds the compute slot for a specific layer index.
# layer_key_fn returns a hashable key for the structural variant of a layer;
# None means all layers share the same structure.
_ARCH: dict[str, ArchSpec] = {
    # --- llama-style dense (verified bit-identical to the stock forward) ---
    "llama": _dense(llama.ModelArgs, llama.TransformerBlock),
    "mistral": _dense(llama.ModelArgs, llama.TransformerBlock),  # HF type 'mistral' uses the llama impl
    "qwen2": _dense(qwen2.ModelArgs, qwen2.TransformerBlock),
    "qwen3": _dense(qwen3.ModelArgs, qwen3.TransformerBlock),
    "apertus": _dense(apertus.ModelArgs, apertus.ApertusDecoderLayer),
    "ernie4_5": _dense(ernie4_5.ModelArgs, ernie4_5.DecoderLayer),
    "glm": _dense(glm.ModelArgs, glm.GLMBlock),
    "glm4": _dense(glm4.ModelArgs, glm4.Glm4DecoderLayer),
    "helium": _dense(helium.ModelArgs, helium.HeliumDecoderLayer),
    "hunyuan_v1_dense": _dense(hunyuan_v1_dense.ModelArgs, hunyuan_v1_dense.TransformerBlock),
    "internlm3": _dense(internlm3.ModelArgs, internlm3.TransformerBlock),
    "mimo": _dense(mimo.ModelArgs, mimo.TransformerBlock),
    "olmo2": _dense(olmo2.ModelArgs, olmo2.TransformerBlock),
    "phi3": _dense(phi3.ModelArgs, phi3.TransformerBlock),
    "seed_oss": _dense(seed_oss.ModelArgs, seed_oss.TransformerBlock),
    "telechat3": _dense(telechat3.ModelArgs, telechat3.Telechat3DecoderLayer),
    "youtu_llm": _dense(youtu_llm.ModelArgs, youtu_llm.YoutuLLMDecoderLayer),
    # --- MoE (per-layer dense/moe variation, selective expert streaming) ---
    "qwen3_moe": ArchSpec(
        args_cls=qwen3_moe.ModelArgs,
        block_factory=qwen3_moe.Qwen3MoeDecoderLayer,  # already a (args, layer_idx) callable
        layer_key_fn=_qwen3_moe_layer_key,
        selective_moe=True,
    ),
    "gpt_oss": ArchSpec(
        args_cls=gpt_oss.ModelArgs,
        block_factory=lambda args, _=0: gpt_oss.TransformerBlock(args),
        layer_key_fn=_gpt_oss_layer_key,
        selective_moe=True,
        mask_plan=_gpt_oss_mask_plan,
        cache_plan=_gpt_oss_cache_plan,
        expert_attr="experts",
        router_attr="router",
        num_experts=lambda args: args.num_local_experts,
        moe_route=_gpt_oss_route,
        supports_quantized_kv=False,   # gpt_oss attention sinks; quantized SDPA raises
    ),
    # --- gemma3 / gemma3_text (sliding window + Gemma norm) ---
    # NOTE: Both gemma3 and gemma3_text use gemma3_text.ModelArgs because the base
    # gemma3 is multimodal with only vocab_size; the real transformer config is in text_config.
    "gemma3": ArchSpec(
        args_cls=gemma3_text.ModelArgs,
        block_factory=lambda args, layer_idx: gemma3.TransformerBlock(args, layer_idx),
        layer_key_fn=_gemma3_layer_key,
        embed_scale=_gemma3_embed_scale,
        final_norm=_gemma3_final_norm,
        mask_plan=_gemma3_mask_plan,
        cache_plan=_gemma3_cache_plan,
    ),
    "gemma3_text": ArchSpec(
        args_cls=gemma3_text.ModelArgs,
        block_factory=lambda args, layer_idx: gemma3_text.TransformerBlock(args, layer_idx),
        layer_key_fn=_gemma3_layer_key,
        embed_scale=_gemma3_embed_scale,
        final_norm=_gemma3_final_norm,
        mask_plan=_gemma3_mask_plan,
        cache_plan=_gemma3_cache_plan,
    ),
    # --- gemma4 / gemma4_text (sliding window + KV sharing + per-layer inputs + MoE) ---
    # NOTE: Both gemma4 and gemma4_text use gemma4_text.ModelArgs because the base
    # gemma4 is multimodal with minimal fields; the real transformer config is in text_config.
    "gemma4": ArchSpec(
        args_cls=gemma4_text.ModelArgs,
        block_factory=lambda args, layer_idx: gemma4.DecoderLayer(args, layer_idx),
        layer_key_fn=_gemma4_layer_key,
        embed_scale=_gemma3_embed_scale,  # Same √hidden scaling
        mask_plan=_gemma4_mask_plan,
        cache_plan=_gemma4_cache_plan,
        layer_runner=GEMMA4_RUNNER,
        kv_sharing=_gemma4_kv_sharing,
        store_full_length_kv=_gemma4_store_full_length_kv,
        per_layer_inputs=_gemma4_per_layer_inputs,
        quant_predicate=_gemma4_quant_predicate,
        resident_extras=_gemma4_resident_extras,
        logit_transform=_gemma4_logit_transform,
    ),
    "gemma4_text": ArchSpec(
        args_cls=gemma4_text.ModelArgs,
        block_factory=lambda args, layer_idx: gemma4_text.DecoderLayer(args, layer_idx),
        layer_key_fn=_gemma4_layer_key,
        embed_scale=_gemma3_embed_scale,  # Same √hidden scaling
        mask_plan=_gemma4_mask_plan,
        cache_plan=_gemma4_cache_plan,
        layer_runner=GEMMA4_RUNNER,
        kv_sharing=_gemma4_kv_sharing,
        store_full_length_kv=_gemma4_store_full_length_kv,
        per_layer_inputs=_gemma4_per_layer_inputs,
        quant_predicate=_gemma4_quant_predicate,
        resident_extras=_gemma4_resident_extras,
        logit_transform=_gemma4_logit_transform,
    ),
    # --- gemma4_assistant (MTP drafter for speculative decoding) ---
    # NOTE: This is a 4-layer Q-only drafter that cross-attends to the target's K/V.
    # It requires special speculative decoding integration and is NOT streamable via
    # the standard StreamingEngine. This registry entry is for packer recognition only.
    "gemma4_assistant": ArchSpec(
        args_cls=gemma4_assistant.ModelArgs,
        # Dummy block_factory - assistant is used as a complete Model, not per-layer
        block_factory=lambda args, _: gemma4_assistant.Model(args),
        layer_key_fn=lambda args, _: "assistant",
    ),
}


def supported_model_types() -> list[str]:
    """The model_type strings StreamingEngine can stream, sorted."""
    return sorted(_ARCH)


def get_architecture(model_type: str) -> ArchSpec:
    """Return the ArchSpec for a model_type.

    Raises ValueError naming the unsupported type and listing the supported ones.
    """
    try:
        return _ARCH[model_type]
    except KeyError:
        raise ValueError(
            f"unsupported model_type {model_type!r}; "
            f"NunSpark streams: {', '.join(supported_model_types())}"
        ) from None
