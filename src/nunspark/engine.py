from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

from .architectures import get_architecture
from .archspec import LayerContext, UniformCausal
from .manifest import Manifest
from .piece_store import PieceStore
from .piece_cache import PieceCache


def _quant_base_config(quant: dict | None) -> dict | None:
    """The top-level (base) quantization config: the scalar entries that apply
    to every module without an explicit per-path override. None for fp16 models.

    Only the kwargs mx quantization accepts (group_size/bits/mode) are kept — the
    same dict is a config that mixed checkpoints (gpt-oss) carry alongside their
    per-module override dicts — so it can be splatted straight into to_quantized /
    QuantizedLinear / QuantizedEmbedding."""
    if not quant:
        return None
    return {k: quant[k] for k in ("group_size", "bits", "mode")
            if k in quant and not isinstance(quant[k], dict)}


def _resolve_module_quant(quant: dict | None, base: dict | None, full_path: str):
    """Resolve one module's quantization config from the full quantization dict,
    keyed by the module's FULL model path (e.g. ``model.layers.3.self_attn.q_proj``,
    ``model.embed_tokens``, ``lm_head``).

    Mirrors mlx_lm.utils.load_model's per-path class_predicate: an explicit
    per-path entry wins (a config dict -> quantize with it; ``False`` -> leave the
    module unquantized, e.g. the bf16 non-expert weights of an ``mxfp4-bf16``
    checkpoint); every other path falls back to the top-level ``base`` config.
    For a uniformly-quantized checkpoint (no dict-valued overrides) every path
    misses and resolves to ``base`` — i.e. exactly the pre-mixed-quant behavior.

    Returns a splat-ready config dict, or None when the module is unquantized."""
    if not quant:
        return None
    entry = quant.get(full_path)
    if entry is not None:
        return entry if isinstance(entry, dict) else None   # dict -> use it; False -> unquantized
    return base


def _default_moe_route(args, gate_logits):
    """qwen3_moe-style routing: softmax over all logits, then top-k by score,
    then optional re-normalization of the selected scores. Returns (indices, scores)."""
    g = mx.softmax(gate_logits, axis=-1, precise=True)
    k = args.num_experts_per_tok
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(g, inds, axis=-1)
    if getattr(args, "norm_topk_prob", False):
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def _append_quantized_rows(cache: QuantizedKVCache, k_rows, v_rows) -> None:
    """Append already-quantized rows (triples sliced from a same-config
    QuantizedKVCache) to `cache` without a dequant/requant round trip.
    Exact because quantization groups span head_dim, never the sequence
    axis. Mirrors QuantizedKVCache.update_and_fetch's growth logic."""
    num = k_rows[0].shape[-2]
    prev = cache.offset
    step = cache.step
    if cache.keys is None or (prev + num) > cache.keys[0].shape[-2]:
        new_steps = (step + num - 1) // step * step

        def grow(x):
            b, h, _, d = x.shape
            return mx.concatenate(
                [x, mx.zeros((b, h, new_steps, d), dtype=x.dtype)], axis=-2)

        if cache.keys is not None:
            if prev % step != 0:
                cache.keys, cache.values = tree_map(
                    lambda x: x[..., :prev, :], (cache.keys, cache.values))
            cache.keys, cache.values = tree_map(
                grow, (cache.keys, cache.values))
        else:
            def fresh(x):
                b, h, _, d = x.shape
                return mx.zeros((b, h, new_steps, d), dtype=x.dtype)
            cache.keys = tuple(fresh(x) for x in k_rows)
            cache.values = tuple(fresh(x) for x in v_rows)
    cache.offset = prev + num
    for i in range(len(cache.keys)):
        cache.keys[i][..., prev:cache.offset, :] = k_rows[i]
        cache.values[i][..., prev:cache.offset, :] = v_rows[i]


def _clone_kv_cache(pcache):
    """A same-type cache object seeded with `pcache`'s state, for one EPHEMERAL
    verify pass. Buffers are shared (mlx arrays are immutable: an in-place
    cache update rebinds only the clone's own attribute references) but every
    python object is DISTINCT — full-range slices, never the same array object
    — so appending to the clone can never mutate the persistent cache.

    RotatingKVCache is cloned field-by-field (its circular-buffer bookkeeping
    `_idx` and the raw rotated buffer must survive; `state` would reorder or
    truncate). KVCache/QuantizedKVCache clone via `state`, exactly as
    tree_forward seeds its ephemeral batched caches."""
    if isinstance(pcache, RotatingKVCache):
        c = RotatingKVCache(pcache.max_size, keep=pcache.keep)
        if pcache.keys is not None:
            c.keys = pcache.keys[:]
            c.values = pcache.values[:]
        c.offset = pcache.offset
        c._idx = pcache._idx
        return c
    if isinstance(pcache, QuantizedKVCache):
        c = QuantizedKVCache(group_size=pcache.group_size, bits=pcache.bits)
        if pcache.keys is not None:
            c.state = tree_map(lambda x: x[:], pcache.state)
            c.offset = pcache.offset    # state setter doesn't set it
        return c
    c = KVCache()
    if pcache.keys is not None:
        pk, pv = pcache.state           # sliced to offset; setter derives offset
        c.state = (pk[:], pv[:])
    return c


class _RecordingCache:
    """Wraps one ephemeral verify-pass cache and records the raw new-token
    (keys, values) the attention layer appends via update_and_fetch — RoPE'd
    exactly as a persistent pass would produce them — so the accepted prefix
    can later be committed to the persistent cache through its own
    update_and_fetch (rotation-safe and quantization-exact by construction:
    QuantizedKVCache.update_and_fetch takes raw rows and quantizes over
    head_dim groups only). Everything else, including the `bits`/`group_size`
    duck-typing mlx_lm's SDPA helper probes with hasattr, is delegated to the
    wrapped cache."""

    def __init__(self, inner):
        self.inner = inner
        self.recorded = None

    def update_and_fetch(self, keys, values):
        self.recorded = (keys, values)
        return self.inner.update_and_fetch(keys, values)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class StreamingEngine:
    """Runs a decoder-only LLM by streaming one transformer layer at a time.

    Reuses mlx-lm's Llama layer math. The small embed + norm/head pieces stay
    persistently resident (hot-set); transformer layers stream through a single
    reusable compute slot, fed by a byte-budget PieceCache that prefetches the
    next layer while the GPU computes the current one.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: Manifest,
        budget_bytes: int,
        prefetch: bool = True,
        io_threads: int = 1,
        warm_window: int = 1,
        expert_trace: str | Path | None = None,
        expert_cache_frac: float = 0.9,
        expert_prefetch: bool = True,
    ):
        self.manifest = manifest
        spec = get_architecture(manifest.model_type)
        self._spec = spec
        self.args = spec.args_cls.from_dict(manifest.config)
        self.store = PieceStore(root, manifest)

        # Two-region cache wiring (plan4 M2). Selectively-packed MoE models get
        # an LRU expert region sized expert_cache_frac * budget; models without
        # expert pieces pass frac=0.0 so the main (MRU) region keeps the whole
        # budget — dense behavior unchanged. Core pieces are pinned
        # unconditionally when they fit the main region's cap (they are tiny —
        # <1 GB for the 30B — yet re-read every layer of every token otherwise);
        # sizes come from the piece files on disk, nothing hardcoded. embed and
        # norm/head never enter the cache: they are loaded once below and stay
        # permanently resident, i.e. already pinned by construction.
        core_pids = [
            Manifest.layer_core_piece_id(l)
            for l in range(manifest.num_layers)
            if manifest.has_piece(Manifest.layer_core_piece_id(l))
        ]
        frac = float(expert_cache_frac) if core_pids else 0.0
        pinned: list[str] = []
        if core_pids:
            main_cap = int(budget_bytes) - int(int(budget_bytes) * frac)
            core_bytes = sum(
                os.stat(self.store.path_for(p)).st_size for p in core_pids
            )
            if core_bytes <= main_cap:
                pinned = core_pids
        self.cache = PieceCache(
            self.store.load, budget_bytes=budget_bytes, pinned=pinned,
            io_threads=io_threads, pather=self.store.path_for,
            expert_frac=frac,
        )
        # RAM-resident fast path: when every streamed piece fits the cache budget
        # such that the PieceCache never evicts anything, the per-layer mx.eval(h)
        # syncs — which exist ONLY to materialize a layer's output before its
        # weights can be evicted — are pure overhead (16+ full pipeline stalls per
        # token). Computed once here against the SAME byte accounting the cache
        # uses for eviction, so the predicate can only be True when eviction is
        # provably impossible; see _compute_fully_resident.
        self._fully_resident = self._compute_fully_resident(budget_bytes, frac)
        self._prefetch = prefetch
        self._warm_window = max(1, warm_window)

        # Temporal expert prefetch (plan4 M3a). Per MoE layer we remember the
        # fired-expert set from the most recent forward pass; when the window
        # prefetcher warms layer L+1..W's cores it also enqueues those layers'
        # remembered expert sets as SPECULATIVE prefetches, overlapping the
        # 85-120 MB/token of expert misses with attention+router compute instead
        # of stalling on them serially after the router. M1: consecutive
        # verify-pass fired-set overlap is 0.74-0.80 (one spec verify pass = one
        # forward over K+1 tokens), so the previous pass's per-layer union is a
        # strong predictor. State lives here (not the cache) and resets at each
        # generation's prefill (kv offset 0); correctness never depends on it —
        # the real router still decides and _scatter_experts always loads the
        # actually-fired experts, so a wrong guess only wastes bandwidth.
        self._expert_prefetch = expert_prefetch
        # Per MoE layer: (fired_expert_ids, from_multi_token_pass). The bool gates
        # Defect 1 — only multi-token history seeds speculative prefetch.
        self._fired_history: dict[int, tuple[list[int], bool]] = {}
        # Set per forward() from the input token count (B*L); gates speculative
        # issuance to multi-token passes only (Defect 2, consume side).
        self._cur_pass_multi = False
        self._stall_seconds = 0.0    # cumulative router-output -> experts-resident wait
        self._block_factory = spec.block_factory
        self._layer_key_fn = spec.layer_key_fn
        self._embed_scale = spec.embed_scale(self.args) if spec.embed_scale else None
        self._logit_transform = (
            spec.logit_transform(self.args) if spec.logit_transform else None
        )
        self._mask_plan = spec.mask_plan(self.args) if spec.mask_plan else UniformCausal()
        self.cache_kinds = spec.cache_plan(self.args) if spec.cache_plan else None
        self._runner = spec.layer_runner
        # The architecture's own class_predicate (gemma4 routes/mlp at 8-bit),
        # used as the fallback for module paths the checkpoint's quantization dict
        # doesn't override explicitly — so uniformly-packed archs keep their
        # existing per-module quant choices bit-for-bit.
        self._arch_quant_predicate = (
            spec.quant_predicate(self.args) if spec.quant_predicate else None
        )
        kv_sharing = spec.kv_sharing(self.args) if spec.kv_sharing else None
        self._lctx = LayerContext(
            kv_sharing=kv_sharing,
            kv_producers=(
                frozenset(p for p in kv_sharing if p is not None)
                if kv_sharing else None
            ),
            store_full_length_kv=(
                spec.store_full_length_kv(self.args) if spec.store_full_length_kv else None
            ),
        )

        # selective-MoE shape (only meaningful when spec.selective_moe; see _moe_layer_forward)
        self._expert_attr = spec.expert_attr
        self._router_attr = spec.router_attr
        self._num_experts = (
            spec.num_experts(self.args) if spec.num_experts else getattr(self.args, "num_experts", None)
        )
        self._moe_route = spec.moe_route or _default_moe_route

        # opt-in expert-trace instrumentation (M1 locality measurement). Off by
        # default: _trace_fh stays None, so _moe_layer_forward pays one `if` check
        # and nothing else. Buffered JSONL, flushed every _TRACE_FLUSH_EVERY records
        # or on close() so tracing doesn't distort timing.
        self._trace_fh = open(Path(expert_trace), "w") if expert_trace is not None else None
        self._trace_buf: list[str] = []
        self._trace_t = 0

        # quant is None for fp16 models. For quantized checkpoints it is the full
        # quantization dict: top-level scalar base config (group_size/bits/mode)
        # PLUS, for mixed-quant checkpoints (all mlx-community gpt-oss), a
        # per-module-path override dict for each non-base module (e.g. gpt-oss keeps
        # its MoE experts at the mxfp4 base but overrides embed_tokens / attention /
        # router / lm_head to 8-bit affine). Every module below is built from ITS
        # OWN resolved config via _module_quant, so a heterogeneous checkpoint no
        # longer forces one uniform config onto every module.
        quant = manifest.config.get("quantization")
        self._quant = quant
        self._base_quant = _quant_base_config(quant)

        # --- embed (hot-set resident) ---
        embed_piece = self.store.load("embed")  # keys: embed_tokens.{weight[,scales,biases]}
        embed_cfg = self._module_quant("model.embed_tokens")
        if embed_cfg is not None:
            self._embed = nn.QuantizedEmbedding(
                self.args.vocab_size, self.args.hidden_size,
                group_size=embed_cfg["group_size"], bits=embed_cfg["bits"],
                mode=embed_cfg.get("mode", "affine"),
            )
        else:
            self._embed = nn.Embedding(self.args.vocab_size, self.args.hidden_size)
        self._embed.update(
            {k.split("embed_tokens.", 1)[1]: v for k, v in embed_piece.items()}
        )

        # --- final norm (always float32) ---
        norm_head = self.store.load("norm_head")
        if self._spec.final_norm:
            self._norm = self._spec.final_norm(self.args)
        else:
            self._norm = nn.RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self._norm.update({"weight": norm_head["norm.weight"]})

        # --- output head (callable: logits = self._head(h)) ---
        if manifest.tie_word_embeddings:
            # as_linear does h @ weightᵀ for nn.Embedding and the quantized
            # matmul for nn.QuantizedEmbedding, so this covers fp16 and every
            # quantization mode (its config comes from the tied embed above).
            self._head = self._embed.as_linear
        else:
            head = {k[len("lm_head."):]: v for k, v in norm_head.items()
                    if k.startswith("lm_head.")}
            head_cfg = self._module_quant("lm_head")
            if head_cfg is not None:
                lm_head = nn.QuantizedLinear(
                    self.args.hidden_size, self.args.vocab_size, bias=False,
                    group_size=head_cfg["group_size"], bits=head_cfg["bits"],
                    mode=head_cfg.get("mode", "affine"),
                )
            else:
                lm_head = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)
            lm_head.update(head)
            self._head = lm_head

        # --- the single reusable compute slot ---
        # _slot_key tracks the structural variant currently loaded; None means
        # "not yet created".  _make_slot() builds and quantizes a fresh slot.
        self._slot: nn.Module | None = None
        self._slot_key: object = object()  # sentinel — never matches a real key
        self._slot = self._make_slot(0)

    def _module_quant(self, full_path: str):
        """Resolve a single module's quant config (dict or None) from the full
        checkpoint quantization dict, keyed by its FULL model path."""
        return _resolve_module_quant(self._quant, self._base_quant, full_path)

    def _compute_fully_resident(self, budget_bytes: int, expert_frac: float) -> bool:
        """True iff EVERY streamed piece fits the cache budget such that the
        PieceCache never evicts anything — making the per-layer mx.eval(h) syncs
        (whose sole purpose is to materialize a layer before its weights can be
        evicted) safely skippable.

        Byte accounting matches the cache/pinning exactly, deliberately on the
        SAFE side: on-disk file size is a conservative upper bound on the
        materialized nbytes the cache actually counts for eviction (a piece file
        is its tensor data plus a small safetensors header), so if the file-size
        sums fit, the nbytes sums the cache compares against fit too -> eviction
        is provably impossible. Only `layer_*` pieces are counted: they are the
        only pieces that ever enter the cache (fetched via cache.get) — embed,
        norm/head and masked_embed are loaded once and stay permanently resident,
        outside the budget.

        The two-region split is honored rather than approximated. A MoE manifest's
        expert pieces must fit the expert region's cap (expert_frac*budget) AND its
        core/dense pieces the main region's remainder (budget - expert cap): once
        the split is active "fits the total budget" is NOT sufficient for "never
        evicted", so the predicate is TIGHTENED per region. A dense manifest has no
        expert pieces, the split never activates, and the main region owns the whole
        budget — so a single total <= budget check is exact there.
        """
        budget = int(budget_bytes)
        expert_bytes = 0
        main_bytes = 0
        for piece in self.manifest.pieces:
            pid = piece.piece_id
            if not pid.startswith("layer_"):
                continue  # embed / norm_head / masked_embed: never cached
            size = os.stat(self.store.path_for(pid)).st_size
            if "_expert_" in pid:
                expert_bytes += size
            else:
                main_bytes += size
        if expert_bytes == 0:
            # No expert region -> split never activates, main region == whole budget.
            return main_bytes <= budget
        expert_budget = int(budget * float(expert_frac))
        return (expert_bytes <= expert_budget
                and main_bytes <= budget - expert_budget)

    def _sync_layer(self, h: mx.array) -> mx.array:
        """Per-layer materialization barrier. Normally forces the lazy graph now
        so a layer's output exists before the PieceCache may evict that layer's
        weights (see comment at _scatter_experts). On the RAM-resident fast path
        (self._fully_resident) nothing is ever evicted, so this sync is skipped and
        the graph is instead materialized once by the final logits/sampling
        consumption of each pass. mx.eval is scheduling-only — it never changes
        numerics — so skipping is bit-identical. Note first-touch materialization
        on fetch/miss (mx.eval in _materialize / _scatter_experts) is a SEPARATE,
        always-on force-read and is unaffected by this."""
        if not self._fully_resident:
            mx.eval(h)
        return h

    def _slot_class_predicate(self, layer_idx: int):
        """class_predicate for nn.quantize over one layer's compute slot.

        nn.quantize hands us each leaf module's slot-relative path (e.g.
        ``self_attn.q_proj``, ``mlp.experts.gate_proj``); we prefix it with this
        layer's full model path and resolve the module's own config from the
        quantization dict (base config + per-path overrides). An explicit override
        wins (dict -> quantize with it, incl. its ``mode``; ``False`` -> skip);
        otherwise, for paths the checkpoint doesn't name we defer to the arch's own
        predicate (gemma4) if any, else fall back to the base config. Returning a
        splat-ready dict lets a mixed checkpoint build, in one slot, mxfp4 experts
        alongside 8-bit-affine attention/router — each with its own group_size/mode."""
        prefix = f"model.layers.{layer_idx}."
        quant = self._quant
        base = self._base_quant
        arch_pred = self._arch_quant_predicate

        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            entry = quant.get(prefix + path)
            if entry is not None:
                return entry                     # dict -> quantize with it; False -> skip
            if arch_pred is not None:
                return arch_pred(path, module)   # gemma4's per-module choices
            return base
        return predicate

    def _make_slot(self, layer_idx: int) -> nn.Module:
        """Build and (per-module) quantize a fresh compute slot for layer_idx.

        Each Linear/expert-Switch is quantized with its own resolved config, so a
        mixed-quant checkpoint (mxfp4 experts + 8-bit-affine attention/router) is
        built correctly; a uniformly-quantized checkpoint resolves every module to
        the base config, unchanged from before."""
        slot = self._block_factory(self.args, layer_idx)
        if self._quant:
            kwargs = dict(
                group_size=self._base_quant["group_size"],
                bits=self._base_quant["bits"],
                class_predicate=self._slot_class_predicate(layer_idx),
            )
            if "mode" in self._base_quant:      # e.g. gpt-oss base is mxfp4
                kwargs["mode"] = self._base_quant["mode"]
            nn.quantize(slot, **kwargs)
        return slot

    def _get_slot(self, layer_idx: int) -> nn.Module:
        """Return the slot for layer_idx, recreating it if the structural type changed."""
        key = self._layer_key_fn(self.args, layer_idx) if self._layer_key_fn else None
        if key != self._slot_key:
            self._slot = self._make_slot(layer_idx)
            self._slot_key = key
        return self._slot

    def _prefetch_pid(self, layer: int) -> str:
        """Piece to prefetch for `layer`: its core piece if selectively packed,
        else its whole-layer piece."""
        core = Manifest.layer_core_piece_id(layer)
        return core if self.manifest.has_piece(core) else Manifest.layer_piece_id(layer)

    def _spec_fired(self, layer: int) -> list[int]:
        """Fired experts remembered for `layer`, but ONLY when that history came
        from a multi-token pass (Defect 1). Single-token greedy history is a weak
        per-token guess and must not drive speculative prefetch; multi-token
        (K-token verify) history keeps working exactly as before."""
        rec = self._fired_history.get(layer)
        if rec is None:
            return []
        fired, from_multi = rec
        return fired if from_multi else []

    def _scatter_experts(self, slot, layer: int, fired: list[int]) -> None:
        """Load only the `fired` experts and scatter their rows into full-size,
        zero-filled expert-module buffers, then update the slot. Unfired rows stay
        zero and are never gathered by the expert module's forward, so output is
        bit-identical to loading all experts. Each piece's rows are copied in
        immediately (scatter-then-discard), so a budget too small to keep pieces
        resident is still correct."""
        attr = self._expert_attr
        sw = getattr(slot.mlp, attr)
        subkeys = [
            (proj, comp)
            for proj in ("gate_proj", "up_proj", "down_proj")
            for comp in ("weight", "scales", "biases", "bias")
            if comp in getattr(sw, proj)            # fp16: weight[,bias]; 4-bit: + scales/biases
        ]
        bufs = {
            (proj, comp): mx.zeros(getattr(sw, proj)[comp].shape,
                                   dtype=getattr(sw, proj)[comp].dtype)
            for proj, comp in subkeys
        }
        # Stall instrumentation (plan4 M3): wall time the expert loads spend on
        # cache.get misses (router-output -> experts-resident). The scatter
        # assignments below are lazy graph-builds; the real disk wait is inside
        # get(), which returns only once the piece is materialized+eval'd.
        t0 = time.monotonic()
        for e in fired:
            piece = self.cache.get(Manifest.layer_expert_piece_id(layer, e))
            for proj, comp in subkeys:
                bufs[(proj, comp)][e] = piece[f"mlp.{attr}.{proj}.{comp}"]
        self._stall_seconds += time.monotonic() - t0
        mx.eval(list(bufs.values()))    # force reads now; cached pieces may be evicted next
        flat = {f"mlp.{attr}.{proj}.{comp}": buf
                for (proj, comp), buf in bufs.items()}
        slot.update(tree_unflatten(list(flat.items())))

    _TRACE_FLUSH_EVERY = 256  # records buffered before a write, so tracing doesn't distort timing

    def _trace_moe(self, layer: int, fired: list[int], batch_tokens: int) -> None:
        """Append one JSONL record for this MoE layer call. Only reached when
        expert_trace was given (see the `if` in _moe_layer_forward)."""
        self._trace_t += 1
        self._trace_buf.append(json.dumps(
            {"t": self._trace_t, "layer": layer, "fired": fired, "batch_tokens": batch_tokens}
        ))
        if len(self._trace_buf) >= self._TRACE_FLUSH_EVERY:
            self._trace_flush()

    def _trace_flush(self) -> None:
        if self._trace_buf:
            self._trace_fh.write("\n".join(self._trace_buf) + "\n")
            self._trace_fh.flush()
            self._trace_buf.clear()

    def _moe_layer_forward(self, h, mask, kv, layer: int, cache_override=None):
        """Selective MoE layer: load core, run the router, load only the fired
        experts, then run the expert mix. Mirrors the reference decoder layer +
        sparse-MoE block exactly (router math via spec.moe_route) so the output
        is bit-identical — qwen3_moe and gpt_oss share this path.

        `cache_override`, if given, is used as the attention cache instead of
        `kv.get(layer)` — verify_forward passes an ephemeral clone so the
        persistent kv is never mutated by a speculative verify pass."""
        slot = self._get_slot(layer)
        core = self.cache.get(Manifest.layer_core_piece_id(layer))
        slot.update(tree_unflatten(list(core.items())))   # attn, norms, router
        n = self.manifest.num_layers
        if self._prefetch and layer + 1 < n:
            hi = min(layer + 1 + self._warm_window, n)
            self.cache.prefetch([self._prefetch_pid(l) for l in range(layer + 1, hi)])
            if kv is not None:
                kv.prefetch(list(range(layer + 1, hi)))
            # M3a: also enqueue the experts layers L+1..W fired on the PREVIOUS
            # pass as speculative prefetches. _fired_history[l] for l > layer still
            # holds the prior pass's set (this pass overwrites it only when it
            # reaches layer l), so this is a true temporal prediction. They ride
            # the low-priority tier, filling I/O slack behind the demand cores.
            # Consume-side gate (Defect 2): only a multi-token pass issues these —
            # a single-token greedy decode issues nothing even if multi-token
            # history exists (record-side gate in _spec_fired handles the converse).
            if self._expert_prefetch and self._cur_pass_multi:
                spec_pids = [
                    Manifest.layer_expert_piece_id(l, e)
                    for l in range(layer + 1, hi)
                    for e in self._spec_fired(l)
                ]
                if spec_pids:
                    self.cache.prefetch(spec_pids, speculative=True)

        if cache_override is not None:
            cache = cache_override
        else:
            cache = kv.get(layer) if kv is not None else None
        h = self._moe_attn_and_mix(slot, layer, h, mask, cache)
        # Selective-MoE layers already host-sync every layer inside
        # _moe_attn_and_mix (router `inds.reshape(-1).tolist()` and the
        # _scatter_experts force-read), so this trailing barrier gains little even
        # off the fast path; skipping it when fully resident is both safe and
        # keeps the lazy graph from growing (the router sync caps its depth anyway).
        self._sync_layer(h)
        return h

    def _moe_attn_and_mix(self, slot, layer: int, h, mask, cache):
        """Attention sub-block + router + selective expert mix for one selective
        MoE layer, given an already-fetched attention `cache` (persistent, in
        `_moe_layer_forward`; ephemeral batched, in `tree_forward`). Factored out
        so both call sites share the exact same math — the only thing that
        differs between them is which cache the attention reads/writes."""
        r = slot.self_attn(slot.input_layernorm(h), mask, cache)
        h = h + r
        x = slot.post_attention_layernorm(h)

        gate_logits = getattr(slot.mlp, self._router_attr)(x)
        inds, scores = self._moe_route(self.args, gate_logits)

        # fired set (tiny host sync: inds is [..., k])
        fired = sorted({int(e) for e in inds.reshape(-1).tolist()})
        batch_tokens = inds.shape[0] * inds.shape[1]   # B*L positions this pass routed
        if self._trace_fh is not None:
            self._trace_moe(layer, fired, batch_tokens)
        # Multi-token bulk-warm (prefill / spec verify): the router just told us
        # the COMPLETE set of expert pieces this layer needs before we touch any of
        # them. _scatter_experts below fetches them SERIALLY via cache.get, each
        # miss a single-threaded cold mmap fault (~300 MB/s) — essentially the whole
        # time-to-first-token on selectively-packed MoE. Kicking off a parallel
        # page-cache warm of exactly that set first lets the serial get loop read
        # warm pages instead. Gated on _cur_pass_multi ONLY: this is demand-critical
        # (not speculative, so not gated on _prefetch), but single-token greedy
        # decode must stay untouched — its fired sets are small, M3 speculative
        # prefetch already covers that regime, and Phase-1 showed warming is
        # net-negative for decode.
        if self._cur_pass_multi:
            self.cache.warm_bulk(
                Manifest.layer_expert_piece_id(layer, e) for e in fired)
        self._scatter_experts(slot, layer, fired)
        if self._expert_prefetch:
            # Remember this pass's fired set so the NEXT pass can prefetch it.
            # Recorded here (after any L+1..W speculative prefetch already read the
            # prior value) so a pass never reads its own freshly-written set.
            # Tag whether the set came from a MULTI-token pass (B*L > 1): only
            # multi-token history drives speculative prefetch (Defect 1) — the
            # per-token guess of a single-token greedy pass is too weak (consecutive
            # Jaccard ~0.30) and mostly wastes bandwidth, whereas a K-token verify
            # pass overlaps 0.74-0.80 with the next.
            self._fired_history[layer] = (fired, batch_tokens > 1)

        y = getattr(slot.mlp, self._expert_attr)(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return h + y

    def _maybe_reset_prefetch(self, kv) -> None:
        """Drop the remembered fired-expert history at a new generation's start.
        Signal: a persistent KV at offset 0 (fresh sequence prefill). We never
        reset when kv is None (one-off forwards with no decode state, e.g. tests,
        should carry history across calls) — stale history only wastes a pass of
        speculative bandwidth, never correctness, so this is purely a tidy-up."""
        if not self._expert_prefetch or kv is None:
            return
        try:
            if kv.get(0).offset == 0:
                self._fired_history.clear()
        except Exception:
            pass

    def prefetch_stats(self) -> dict:
        """Prefetch/stall instrumentation (plan4 M3). `stall_seconds` is the
        cumulative wall time expert loads waited on cache.get misses (router
        output -> experts resident); the `speculative` block is the cache's
        temporal-prefetch counters (issued / used / wasted_bytes)."""
        return {
            "stall_seconds": self._stall_seconds,
            "speculative": self.cache.stats()["speculative"],
        }

    def forward(self, tokens: mx.array, kv=None) -> mx.array:
        # Advance the cache's speculative-protection epoch: one call per forward
        # pass, so a speculative expert survives its insertion pass plus exactly
        # the next pass (the one its temporal prediction is for).
        self.cache.begin_pass()
        # Consume-side gate (Defect 2): only a MULTI-token pass may ISSUE
        # speculative expert prefetch. Recorded here once from the input shape
        # (B*L) so the per-layer M3a block in _moe_layer_forward can skip issuance
        # on single-token greedy decode — even when multi-token history exists (as
        # right after a prefill), where issuing the prefill's giant per-layer unions
        # would flood the staging buffer and wreck greedy throughput.
        self._cur_pass_multi = tokens.size > 1
        self._maybe_reset_prefetch(kv)
        self._lctx.shared_kv.clear()
        self._lctx.target_kv_states.clear()
        h = self._embed(tokens)
        if self._embed_scale is not None:
            h = h * self._embed_scale
        masks = self._mask_plan.build(h, kv)

        n = self.manifest.num_layers
        for layer in range(n):
            # Selectively-packed MoE layers run the router-then-load path.
            if self.manifest.has_piece(Manifest.layer_core_piece_id(layer)):
                h = self._moe_layer_forward(h, masks[layer], kv, layer)
                continue

            pid = Manifest.layer_piece_id(layer)
            weights = self.cache.get(pid)
            slot = self._get_slot(layer)
            slot.update(tree_unflatten(list(weights.items())))
            if self._prefetch and layer + 1 < n:
                # Warm a window of upcoming layers so the parallel reader pool has
                # several files in flight. _prefetch_pid picks the core piece when a
                # next layer is a selectively-packed MoE layer. (W=1 == old behavior.)
                hi = min(layer + 1 + self._warm_window, n)
                self.cache.prefetch([self._prefetch_pid(l) for l in range(layer + 1, hi)])
                if kv is not None:
                    kv.prefetch(list(range(layer + 1, hi)))

            cache = kv.get(layer) if kv is not None else None
            h = self._runner.run(self._lctx, slot, layer, h, masks[layer], cache)
            self._sync_layer(h)

        # shared_kv is intra-pass producer->consumer state; dropping it here
        # releases the pinned KV copies. target_kv_states must survive the
        # return: the MTP drafter reads it via target_kv_states().
        self._lctx.shared_kv.clear()

        h = self._norm(h)
        self._last_hidden = h
        logits = self._head(h)
        if self._logit_transform is not None:
            logits = self._logit_transform(logits)
        return logits

    @property
    def supports_quantized_kv(self) -> bool:
        """Whether this architecture tolerates a quantized KV cache
        (False for attention-sink archs: quantized SDPA rejects sinks)."""
        return self._spec.supports_quantized_kv

    def last_hidden_state(self) -> mx.array:
        """Post-norm hidden state from the most recent `forward()` call."""
        return self._last_hidden

    def target_kv_states(self) -> dict:
        """{layer_type: (keys, values)} captured by store_full_length_kv
        layers during the most recent `forward()` call (gemma4 only; empty
        dict for other architectures)."""
        return dict(self._lctx.target_kv_states)

    def embed_tokens(self, tokens: mx.array) -> mx.array:
        """Embedding lookup as the model's input-embedding module produces it,
        INCLUDING embed_scale — used by MTP drafters.

        Mirrors HF's `target_model_input_embeddings = target.get_input_embeddings()`:
        for gemma4 that module is Gemma4TextScaledWordEmbedding, whose forward
        multiplies by sqrt(hidden_size) (~73x for the 31B). The drafter was
        trained on these scaled embeddings; feeding raw lookups makes the
        token-identity input ~73x too small and wrecks draft acceptance.
        """
        h = self._embed(tokens)
        if self._embed_scale is not None:
            h = h * self._embed_scale
        return h

    def tree_forward(self, token_paths: mx.array, kv, prefix_len: int):
        """Verify a batch of root-to-leaf paths in ONE streamed sweep.

        `token_paths` is `[B, d]` (B paths, each d tokens, all starting at position
        `prefix_len`). Per layer the committed prefix KV (B=1, length `prefix_len`) is
        tiled to B into an ephemeral cache, the d new tokens are appended, and only the
        new-token K/V (`[B, kv_heads, d, head_dim]`) is stashed. The persistent `kv` is
        NOT mutated. Returns `(logits[B, d, vocab], stash)` where `stash[layer]` is the
        `(keys, values)` for the d new tokens at that layer — each side is an array
        for fp16 caches or a quantized `(q, scales, biases)` triple for quantized caches.

        Requires `kv` to have been prefilled to `prefix_len` tokens (via `forward`);
        each layer's prefix KV is read to seed the ephemeral batched cache.
        """
        self.cache.begin_pass()   # advance the speculative-protection epoch (one per pass)
        self._lctx.shared_kv.clear()
        self._lctx.target_kv_states.clear()
        B = token_paths.shape[0]
        h = self._embed(token_paths)                       # [B, d, D]
        mask = create_attention_mask(h, kv.get(0))         # uses prefix offset == prefix_len

        stash: list[tuple[mx.array, mx.array]] = []
        n = self.manifest.num_layers
        for layer in range(n):
            slot = self._get_slot(layer)
            # Selectively-packed MoE layers have no whole-layer piece: load the
            # core, then mirror _moe_layer_forward's structure — run attention,
            # route on the B*d verify-pass positions, load only the UNION of
            # experts fired across the whole batch, run the expert mix. Measured
            # (docs/plan4-m1-gate-summary.md) the per-K=24-pass fired union is
            # 51-65% of all experts, so this avoids ~2x the necessary expert
            # bytes that loading every expert would cost.
            is_moe = self.manifest.has_piece(Manifest.layer_core_piece_id(layer))
            if is_moe:
                core = self.cache.get(Manifest.layer_core_piece_id(layer))
                slot.update(tree_unflatten(list(core.items())))
            else:
                weights = self.cache.get(Manifest.layer_piece_id(layer))
                slot.update(tree_unflatten(list(weights.items())))
            if self._prefetch and layer + 1 < n:
                # Window-prefetch upcoming layers + their prefix-KV (W=1 == old behavior).
                hi = min(layer + 1 + self._warm_window, n)
                self.cache.prefetch([self._prefetch_pid(l) for l in range(layer + 1, hi)])
                kv.prefetch(list(range(layer + 1, hi)))
                # M3a: same temporal expert prefetch as _moe_layer_forward — enqueue
                # layers L+1..W's PREVIOUS-pass fired sets speculatively. Must read
                # _fired_history before this pass's _moe_attn_and_mix() below
                # overwrites layer `layer`'s entry.
                if is_moe and self._expert_prefetch:
                    spec_pids = [
                        Manifest.layer_expert_piece_id(l, e)
                        for l in range(layer + 1, hi)
                        for e in self._spec_fired(l)
                    ]
                    if spec_pids:
                        self.cache.prefetch(spec_pids, speculative=True)

            pcache = kv.get(layer)
            if isinstance(pcache, QuantizedKVCache):
                ephem = QuantizedKVCache(
                    group_size=pcache.group_size, bits=pcache.bits)
                ephem.state = tree_map(
                    lambda x: mx.repeat(x, B, axis=0), pcache.state)
                ephem.offset = pcache.offset    # state setter doesn't set it
            else:
                pk, pv = pcache.state           # [1, kv_heads, prefix_len, hd]
                ephem = KVCache()
                ephem.state = (mx.repeat(pk, B, axis=0), mx.repeat(pv, B, axis=0))

            if is_moe:
                h = self._moe_attn_and_mix(slot, layer, h, mask, ephem)
            else:
                h = slot(h, mask=mask, cache=ephem)
            self._sync_layer(h)

            # [B, kv_heads, prefix_len+d, ...]: arrays for fp16, triples for
            # quantized — tree_map slices both shapes uniformly.
            stash.append(tree_map(
                lambda x: x[..., prefix_len:, :], ephem.state))

        h = self._norm(h)
        return self._head(h), stash

    def commit_path(self, kv, stash, path_index: int, accepted_len: int) -> None:
        """Append path `path_index`'s first `accepted_len` new-token K/V to the
        persistent B=1 KVStore — no re-feed, no extra weight read. `accepted_len`
        counts from the root (>= 1). Quantized stashes are committed in
        quantized form directly (exact: groups span head_dim only)."""
        def sl(x):
            return x[path_index:path_index + 1, :, :accepted_len, :]
        for layer, (ek, ev) in enumerate(stash):
            cache = kv.get(layer)
            if isinstance(cache, QuantizedKVCache):
                _append_quantized_rows(cache, tree_map(sl, ek), tree_map(sl, ev))
            else:
                cache.update_and_fetch(sl(ek), sl(ev))

    def verify_forward(self, tokens: mx.array, kv):
        """Multi-token verify pass over EPHEMERAL per-layer cache clones —
        the same math as `forward()`, but the persistent `kv` is NEVER mutated.

        Speculative verify passes need rollback of rejected tokens, and
        trim-based rollback is unsound for RotatingKVCache once it has rotated
        (evicted positions cannot be restored; `is_trimmable()` is False past
        the window). Instead each layer runs on a `_clone_kv_cache` of its
        persistent cache wrapped in a `_RecordingCache`; the caller commits
        only the accepted prefix afterwards via `commit_verified` — the
        ephemeral/commit pattern of `tree_forward`/`commit_path`, generalized
        to heterogeneous (rotating + full) cache stacks.

        Masks are built from the persistent caches' offsets, which equal the
        clones' starting offsets, so they are exactly what `forward()` would
        build. Returns `(logits, recs)` where `recs[layer].recorded` holds the
        raw new-token (keys, values) for that layer.
        """
        self.cache.begin_pass()
        self._cur_pass_multi = tokens.size > 1
        self._maybe_reset_prefetch(kv)
        self._lctx.shared_kv.clear()
        self._lctx.target_kv_states.clear()
        h = self._embed(tokens)
        if self._embed_scale is not None:
            h = h * self._embed_scale
        masks = self._mask_plan.build(h, kv)

        recs: list[_RecordingCache] = []
        n = self.manifest.num_layers
        for layer in range(n):
            rec = _RecordingCache(_clone_kv_cache(kv.get(layer)))
            recs.append(rec)
            if self.manifest.has_piece(Manifest.layer_core_piece_id(layer)):
                h = self._moe_layer_forward(h, masks[layer], kv, layer,
                                            cache_override=rec)
                continue

            pid = Manifest.layer_piece_id(layer)
            weights = self.cache.get(pid)
            slot = self._get_slot(layer)
            slot.update(tree_unflatten(list(weights.items())))
            if self._prefetch and layer + 1 < n:
                hi = min(layer + 1 + self._warm_window, n)
                self.cache.prefetch([self._prefetch_pid(l) for l in range(layer + 1, hi)])
                kv.prefetch(list(range(layer + 1, hi)))

            h = self._runner.run(self._lctx, slot, layer, h, masks[layer], rec)
            self._sync_layer(h)

        self._lctx.shared_kv.clear()
        h = self._norm(h)
        self._last_hidden = h
        logits = self._head(h)
        if self._logit_transform is not None:
            logits = self._logit_transform(logits)
        return logits, recs

    def commit_verified(self, kv, recs, accepted_len: int) -> None:
        """Append the first `accepted_len` new-token K/V rows recorded by
        `verify_forward` to the persistent `kv` — no re-feed, no extra weight
        read, no trim. Each persistent cache receives the rows through its OWN
        `update_and_fetch`, so rotation (RotatingKVCache) and quantization
        (QuantizedKVCache quantizes raw rows over head_dim groups) behave
        bit-identically to having fed the accepted tokens directly."""
        if accepted_len <= 0:
            return
        for layer, rec in enumerate(recs):
            k, v = rec.recorded
            kv.get(layer).update_and_fetch(
                k[..., :accepted_len, :], v[..., :accepted_len, :])

    def close(self) -> None:
        """Shut down the cache's background prefetch worker."""
        if self._trace_fh is not None:
            self._trace_flush()
            self._trace_fh.close()
        self.cache.close()
