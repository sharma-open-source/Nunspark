from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from .architectures import get_architecture
from .archspec import LayerContext, UniformCausal
from .manifest import Manifest
from .piece_store import PieceStore
from .piece_cache import PieceCache


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
        self._quant_predicate = (
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

        quant = manifest.config.get("quantization")  # None for fp16 models
        self._quant = quant

        # --- embed (hot-set resident) ---
        embed_piece = self.store.load("embed")  # keys: embed_tokens.{weight[,scales,biases]}
        if quant:
            self._embed = nn.QuantizedEmbedding(
                self.args.vocab_size, self.args.hidden_size,
                group_size=quant["group_size"], bits=quant["bits"],
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
            # matmul for nn.QuantizedEmbedding, so this covers both fp16 and 4-bit.
            self._head = self._embed.as_linear
        else:
            head = {k[len("lm_head."):]: v for k, v in norm_head.items()
                    if k.startswith("lm_head.")}
            if quant:
                lm_head = nn.QuantizedLinear(
                    self.args.hidden_size, self.args.vocab_size, bias=False,
                    group_size=quant["group_size"], bits=quant["bits"],
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

    def _make_slot(self, layer_idx: int) -> nn.Module:
        """Build and optionally quantize a fresh compute slot for layer_idx."""
        slot = self._block_factory(self.args, layer_idx)
        if self._quant:
            nn.quantize(
                slot,
                group_size=self._quant["group_size"],
                bits=self._quant["bits"],
                class_predicate=self._quant_predicate,
            )
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

    def _moe_layer_forward(self, h, mask, kv, layer: int):
        """Selective MoE layer: load core, run the router, load only the fired
        experts, then run the expert mix. Mirrors the reference decoder layer +
        sparse-MoE block exactly (router math via spec.moe_route) so the output
        is bit-identical — qwen3_moe and gpt_oss share this path."""
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

        cache = kv.get(layer) if kv is not None else None
        h = self._moe_attn_and_mix(slot, layer, h, mask, cache)
        mx.eval(h)
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
            mx.eval(h)

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
            mx.eval(h)

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

    def close(self) -> None:
        """Shut down the cache's background prefetch worker."""
        if self._trace_fh is not None:
            self._trace_flush()
            self._trace_fh.close()
        self.cache.close()
