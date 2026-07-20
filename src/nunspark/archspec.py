from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList, KVCache, QuantizedKVCache, RotatingKVCache


# --- mask planning --------------------------------------------------------

class MaskIndex(Protocol):
    """Per-layer mask lookup produced by a MaskPlan.build()."""

    def __getitem__(self, layer_idx: int) -> Any: ...


class _ConstMask:
    """A MaskIndex that returns one shared mask for every layer."""

    def __init__(self, mask: Any):
        self._mask = mask

    def __getitem__(self, layer_idx: int) -> Any:
        return self._mask


# --- cache planning ---------------------------------------------------------

KV = "kv"   # sentinel CacheKind: a standard growing KVCache


@dataclass(frozen=True)
class KVQuant:
    """Opt-in KV-cache quantization config. Applies to standard KV layers
    only; Rotating (sliding-window) layers stay fp16 — no quantized rotating
    cache exists upstream and they're bounded by the window anyway."""
    bits: int
    group_size: int = 64

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(f"kv-bits must be 4 or 8, got {self.bits}")


@dataclass(frozen=True)
class Rotating:
    """CacheKind for a sliding-window layer -> RotatingKVCache(max_size=window)."""
    window: int


@dataclass(frozen=True)
class Paired:
    """CacheKind for a layer holding N independent KVCaches in a CacheList —
    deepseek_v32/glm_moe_dsa: [0] the MLA latent cache, [1] the DSA indexer's
    key cache. Children are always plain KVCache: the MLA latent path stores a
    compressed latent (not per-head K/V) and the indexer cache abuses `values`
    as a zero-width placeholder, so KV-quantization is meaningless for both
    (the owning ArchSpec sets supports_quantized_kv=False)."""
    n: int = 2


def cache_offset(cache) -> int:
    """The sequence offset of a per-layer cache, CacheList-aware. A CacheList's
    children advance in lockstep (one update_and_fetch each per forward), so the
    first child's offset IS the layer's offset."""
    return cache[0].offset if isinstance(cache, CacheList) else cache.offset


def make_cache(kind, quant: KVQuant | None = None):
    """Build a fresh mlx cache for a CacheKind. None or KV -> KVCache
    (QuantizedKVCache when `quant` is set); Rotating(window) ->
    RotatingKVCache(max_size=window, keep=0), always fp16; Paired(n) ->
    CacheList of n plain KVCaches (never quantized)."""
    if kind is None or kind == KV:
        if quant is not None:
            return QuantizedKVCache(group_size=quant.group_size, bits=quant.bits)
        return KVCache()
    if isinstance(kind, Rotating):
        return RotatingKVCache(max_size=kind.window, keep=0)
    if isinstance(kind, Paired):
        if quant is not None:
            raise ValueError("Paired cache layers do not support KV quantization")
        return CacheList(*(KVCache() for _ in range(kind.n)))
    raise ValueError(f"unknown CacheKind: {kind!r}")


class MaskPlan(Protocol):
    def build(self, h: mx.array, kv) -> MaskIndex: ...


@dataclass(frozen=True)
class UniformCausal:
    """One global causal mask shared by all layers — the historical behavior.

    Mirrors the engine's previous inline call exactly:
        create_attention_mask(h, kv.get(0) if kv is not None else None)
    """

    def build(self, h: mx.array, kv) -> MaskIndex:
        cache0 = kv.get(0) if kv is not None else None
        return _ConstMask(create_attention_mask(h, cache0))


@dataclass(frozen=True)
class ArrayCausal:
    """One global causal mask as an ARRAY, built from layer 0's first
    sub-cache — deepseek_v32/glm_moe_dsa. The DSA indexer applies the mask
    with `mx.where` and the L>1 sparse path intersects it with a boolean
    top-k mask, so the "causal" string sentinel is unusable; mirrors the
    stock model's
        create_attention_mask(h, cache[0][0] if cache[0] else None,
                              return_array=True)
    where each layer's cache is a CacheList and [0] is the MLA latent cache.
    """

    def build(self, h: mx.array, kv) -> MaskIndex:
        cache0 = kv.get(0) if kv is not None else None
        inner = cache0[0] if cache0 is not None else None
        return _ConstMask(create_attention_mask(h, inner, return_array=True))


class _ListMask:
    """A MaskIndex backed by an explicit per-layer list of masks."""

    def __init__(self, masks: list):
        self._masks = masks

    def __getitem__(self, layer_idx: int):
        return self._masks[layer_idx]


@dataclass(frozen=True)
class PerLayer:
    """Per-layer attention masks for heterogeneous attention.

    `kinds[i]` is "global" (full causal) or ("sliding", window). One mask is
    built per *distinct kind value* and shared by all layers of that kind —
    typically two (one global + one sliding window), but a model mixing
    several window sizes builds one array per size.

    Each kind's mask is built from the cache of the FIRST layer of that same
    kind — never from layer 0 unconditionally. create_attention_mask delegates
    to cache.make_mask, and RotatingKVCache.make_mask clamps its offset to the
    window, so building the global mask from a sliding layer's rotating cache
    (as gpt-oss layer 0 is) yields a mask too short for the full-attention
    layers' keys once the sequence exceeds the window (broadcast crash on any
    multi-token pass past that point).
    """
    kinds: list

    def build(self, h, kv) -> MaskIndex:
        built: dict = {}
        masks = []
        for i, kind in enumerate(self.kinds):
            if kind not in built:
                cache_i = kv.get(i) if kv is not None else None
                if kind == "global":
                    built[kind] = create_attention_mask(h, cache_i)
                elif isinstance(kind, tuple) and len(kind) == 2 and kind[0] == "sliding":
                    _, window = kind
                    built[kind] = create_attention_mask(h, cache_i, window_size=window)
                else:
                    raise ValueError(f"unknown mask kind: {kind!r}")
            masks.append(built[kind])
        return _ListMask(masks)


# --- per-layer call strategy ---------------------------------------------

@dataclass(frozen=True)
class LayerContext:
    """Per-forward context handed to a LayerRunner. Default runner ignores it;
    gemma4's runner (M2) reads per_layer_inputs and shares producer KV.

    `per_layer_inputs`, `kv_sharing`, `kv_producers`, and
    `store_full_length_kv` are static — set once at construction and never
    mutated. `shared_kv` and `target_kv_states` are per-forward scratch dicts:
    the engine clears them at the top of every `forward()` call, and the
    runner repopulates them while iterating layers.

    The runner only stashes a layer's (k, v) when some later layer actually
    reads it (`kv_producers`) or the MTP drafter needs it
    (`store_full_length_kv`). Stashing every layer would pin a materialized
    copy of the whole model's KV in memory until the next forward — for a
    model with no KV-shared layers, that's pure waste.
    """

    per_layer_inputs: list | None = None
    kv_sharing: list | None = None
    kv_producers: frozenset | None = None
    store_full_length_kv: dict | None = None
    shared_kv: dict = field(default_factory=dict)
    target_kv_states: dict = field(default_factory=dict)


class LayerRunner(Protocol):
    def run(self, lctx: LayerContext, slot, layer_idx: int,
            h: mx.array, mask, cache) -> mx.array: ...


class DefaultLayerRunner:
    """llama-style invocation: h = slot(h, mask=mask, cache=cache)."""

    def run(self, lctx: LayerContext, slot, layer_idx: int,
            h: mx.array, mask, cache) -> mx.array:
        return slot(h, mask=mask, cache=cache)


DEFAULT_RUNNER = DefaultLayerRunner()


# --- the spec -------------------------------------------------------------

@dataclass(frozen=True)
class ArchSpec:
    """Everything the streaming engine + packer need to know about a model_type.

    Only `args_cls` and `block_factory` are required; every other hook is None/
    default, in which case the engine reproduces the llama-style behavior.
    """

    args_cls: type
    block_factory: Callable                       # (args, layer_idx) -> nn.Module
    layer_key_fn: Callable | None = None          # (args, idx) -> hashable variant key
    selective_moe: bool = False                   # qwen3_moe-style expert splitting (packer + engine)
    # selective-MoE shape (only consulted when selective_moe=True; defaults match
    # qwen3_moe so its registry entry needn't set them):
    expert_attr: str = "switch_mlp"                # name of the SwitchGLU module under .mlp
    router_attr: str = "gate"                      # name of the router/gate nn.Linear under .mlp
    num_experts: Callable | None = None            # (args) -> int; None => args.num_experts
    moe_route: Callable | None = None              # (args, gate_logits) -> (indices, scores);
                                                    # None => qwen3-style softmax -> top-k -> optional renorm
    shared_experts_attr: str | None = None         # always-on expert module under .mlp, added
                                                    # to the routed mix (deepseek-style); None => absent
    moe_mix_cast: bool = False                     # cast the scored expert sum back to the expert
                                                    # output dtype (deepseek: scores are fp32)
    # declarative forward hooks (None => llama default):
    embed_scale: Callable | None = None           # (args) -> float | None
    final_norm: Callable | None = None            # (args) -> nn.Module
    logit_transform: Callable | None = None       # (args) -> (logits->logits) | None
    mask_plan: Callable | None = None             # (args) -> MaskPlan
    cache_plan: Callable | None = None            # (args) -> list[CacheKind]
    layer_runner: LayerRunner = DEFAULT_RUNNER
    # gemma4-only (None for everyone else):
    per_layer_inputs: Callable | None = None      # (args) -> PerLayerInputSpec
    kv_sharing: Callable | None = None            # (args) -> list[int | None]
    store_full_length_kv: Callable | None = None  # (args) -> dict[int, str]
    quant_predicate: Callable | None = None       # (args) -> class_predicate
    resident_extras: Callable | None = None       # (args) -> extra hot-set piece ids
    supports_quantized_kv: bool = True            # False for attention-sink archs
                                                  #   (quantized SDPA rejects sinks)
