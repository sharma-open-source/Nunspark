from __future__ import annotations

import tempfile
from collections.abc import Generator
from collections import deque
from dataclasses import dataclass

import mlx.core as mx

from .archspec import KVQuant
from .engine import StreamingEngine
from .kv_store import KVStore


def _sample(logits: mx.array, temp: float) -> int:
    if temp <= 0.0:
        return int(mx.argmax(logits, axis=-1).item())
    # categorical takes unnormalized logits, so scaling by 1/temp is sufficient.
    return int(mx.random.categorical(logits * (1.0 / temp)).item())


def check_kv_quant_support(engine: StreamingEngine, kv_quant: KVQuant | None) -> None:
    """Reject kv_quant for attention-sink architectures up front — quantized
    SDPA raises on sinks, and failing here beats failing mid-forward. Runs
    even when the caller passes an external `kv`."""
    if kv_quant is not None and not engine.supports_quantized_kv:
        raise ValueError(
            f"{engine.manifest.model_type} uses attention sinks; "
            "quantized KV cache is unsupported (drop --kv-bits)")


def _open_kv_store(engine: StreamingEngine, tmpdir: str, kv_budget: int, prefetch: bool,
                   kv_quant: KVQuant | None) -> KVStore:
    """Construct the internally-owned KVStore shared by all generation paths."""
    return KVStore(tmpdir, budget_bytes=kv_budget, prefetch=prefetch,
                    cache_kinds=engine.cache_kinds, kv_quant=kv_quant)


@dataclass
class SpecStats:
    """Tally for a speculative run. `target_passes` == streamed weight reads."""
    target_passes: int = 0
    tokens_emitted: int = 0
    draft_tokens_proposed: int = 0
    accepted_offpath: int = 0   # accepted draft tokens that were NOT the target argmax (lossy)
    accepted_total: int = 0     # accepted draft tokens (excludes the bonus correction)

    @property
    def multiplier(self) -> float:
        """M = accepted tokens per target forward pass (the streaming speedup)."""
        return self.tokens_emitted / self.target_passes if self.target_passes else 0.0

    @property
    def deviation_rate(self) -> float:
        """Fraction of accepted draft tokens that diverged from the target's argmax.
        0.0 = lossless (accept_top_k=1); rises with accept_top_k. 0.0 when nothing
        was accepted."""
        return self.accepted_offpath / self.accepted_total if self.accepted_total else 0.0


@dataclass
class AdaptiveSpecStats(SpecStats):
    """Extended stats with adaptive adjustment tracking."""
    acceptance_history: list[float] = None
    adjustment_count: int = 0
    current_draft_tokens: int = 0

    def __post_init__(self):
        if self.acceptance_history is None:
            self.acceptance_history = []

    @property
    def recent_acceptance(self) -> float:
        """Average acceptance rate over last 10 rounds."""
        if not self.acceptance_history:
            return 0.0
        return sum(self.acceptance_history[-10:]) / min(len(self.acceptance_history), 10)


def generate(
    engine: StreamingEngine,
    prompt: list[int],
    max_tokens: int = 64,
    temp: float = 0.0,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
) -> list[int]:
    """Greedy/temperature decode using the streaming engine.

    The per-layer KV cache is held by a KVStore: layers within `kv_budget` stay
    resident (no disk cost), overflow layers stream to/from SSD. The default
    budget is large enough that everything stays resident (RAM-only behavior).
    Pass a `kv` to borrow an externally-owned store (caller closes it and reads
    its stats); otherwise generate creates and disposes one internally. The
    caller always owns the engine and is responsible for engine.close().
    `kv_quant`, if given, quantizes the KV cache entries (unsupported on
    attention-sink architectures; raises ValueError).
    """
    check_kv_quant_support(engine, kv_quant)
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        logits = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]
        out: list[int] = []
        for _ in range(max_tokens):
            nxt = _sample(logits, temp)
            out.append(nxt)
            logits = engine.forward(mx.array([nxt])[None], kv=kv)[:, -1, :]
        return out
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()


def stream_generate(
    engine: StreamingEngine,
    prompt: list[int],
    max_tokens: int = 64,
    temp: float = 0.0,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
    processed_tokens: int = 0,
) -> Generator[int, None, None]:
    """Like generate(), but yields one token id at a time for live streaming.

    The caller owns the engine; KVStore lifecycle follows the same rules as
    generate() — pass kv= to borrow an external store, or let it manage one.
    `kv_quant`, if given, quantizes the KV cache entries (unsupported on
    attention-sink architectures; raises ValueError).

    `processed_tokens` is the number of leading `prompt` tokens already held
    by a borrowed `kv` (prompt-prefix reuse): only `prompt[processed_tokens:]`
    is prefilled, the cache offset supplies the correct positions. Must be 0
    for an internally-owned (fresh) store.
    """
    check_kv_quant_support(engine, kv_quant)
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        prefill = prompt[processed_tokens:]
        logits = engine.forward(mx.array(prefill)[None], kv=kv)[:, -1, :]
        for _ in range(max_tokens):
            nxt = _sample(logits, temp)
            yield nxt
            logits = engine.forward(mx.array([nxt])[None], kv=kv)[:, -1, :]
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()


def speculative_generate(
    engine: StreamingEngine,
    draft_model,
    prompt: list[int],
    max_tokens: int = 64,
    num_draft_tokens: int = 16,
    accept_top_k: int = 1,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
    eos_id: int | None = None,
    stats: SpecStats | None = None,
    processed_tokens: int = 0,
) -> Generator[int, None, None]:
    """Speculative decoding over the streaming target.

    A resident `draft_model` (a normal mlx_lm model, callable as
    `draft_model(tokens, cache=cache_list)`) proposes `num_draft_tokens` tokens;
    the streamed target verifies them in ONE forward pass.

    `accept_top_k` controls the accept test:
      * 1 (default) -> LOSSLESS: a draft token is accepted only if it equals the
        target's argmax, so output is bit-identical to greedy `generate()`
        regardless of draft quality.
      * >1 -> opt-in FAST MODE: a draft token is accepted if it lies within the
        target's top-k logits (still a token the target conditioned on during
        verification), trading exactness for a higher acceptance multiplier.
        `stats.deviation_rate` reports the measured lossiness.

    KVStore lifecycle follows the same rules as `generate()`. `kv_quant`, if
    given, quantizes the KV cache entries (unsupported on attention-sink
    architectures; raises ValueError).

    `processed_tokens` is the number of leading `prompt` tokens the borrowed
    target `kv` already holds (prompt-prefix reuse): the streamed target
    prefills only `prompt[processed_tokens:]`, while the resident draft model
    (its cache is fresh per call) is always prefilled over the FULL prompt so
    its proposals stay well-conditioned. Must be 0 for a fresh store.
    """
    from mlx_lm.models.cache import make_prompt_cache

    check_kv_quant_support(engine, kv_quant)
    K = num_draft_tokens
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        # Prefill the target over the un-cached suffix (the borrowed kv already
        # holds the first processed_tokens); the draft sees the full prompt.
        # b is the first confirmed token. The target's cache offset supplies the
        # correct positions for the suffix.
        logits = engine.forward(
            mx.array(prompt[processed_tokens:])[None], kv=kv)[:, -1, :]
        draft_cache = make_prompt_cache(draft_model)
        draft_model(mx.array(prompt)[None], cache=draft_cache)
        b = int(mx.argmax(logits, axis=-1).item())

        emitted = 0
        yield b
        emitted += 1
        if stats:
            stats.tokens_emitted += 1
        if b == eos_id:
            return

        while emitted < max_tokens:
            # 1) Draft K tokens from the current bootstrap b.
            q: list[int] = []
            di = mx.array([b])[None]
            for _ in range(K):
                dl = draft_model(di, cache=draft_cache)[:, -1, :]
                nx = int(mx.argmax(dl, axis=-1).item())
                q.append(nx)
                di = mx.array([nx])[None]
            if stats:
                stats.draft_tokens_proposed += K

            # 2) Verify [b, q0..q_{K-1}] in one target sweep -> K+1 logit rows.
            vlog = engine.forward(mx.array([b] + q)[None], kv=kv)
            targ = mx.argmax(vlog[0], axis=-1).tolist()   # len K+1; targ[i] = argmax after position i
            if stats:
                stats.target_passes += 1

            # 3) Accept the longest prefix of plausible draft tokens.
            #    accept_top_k <= 1 -> lossless greedy (draft must equal the target argmax).
            #    accept_top_k  > 1 -> accept if the draft token is within the target's top-k.
            #    accept_top_k >= V -> accept any token (avoids an out-of-range argpartition).
            V = vlog.shape[-1]
            m = 0
            for i in range(K):
                if accept_top_k <= 1:
                    ok = (q[i] == targ[i])
                elif accept_top_k >= V:
                    ok = True
                else:
                    row = vlog[0, i]                                          # [V]
                    topk = mx.argpartition(row, -accept_top_k)[-accept_top_k:]
                    ok = q[i] in set(topk.tolist())
                if not ok:
                    break
                m += 1
                if stats and q[i] != targ[i]:
                    stats.accepted_offpath += 1
            if stats:
                stats.accepted_total += m
            bonus = int(targ[m])   # correct token after last accepted (m <= K, len(targ)=K+1)

            # 4) Roll back the rejected tail in both caches.
            #    Target cached b+q0..q_{K-1} (K+1 beyond prompt); keep b+q0..q_{m-1}.
            kv.truncate(K - m)
            if m < K:
                #    Draft cached b+q0..q_{K-2} (K beyond prompt); keep b+q0..q_{m-1}.
                drop = (K - 1) - m
                for c in draft_cache:
                    c.trim(drop)
            else:
                #    All K accepted, but the draft never cached q_{K-1}; feed it once
                #    to keep draft/target offsets aligned for the next round.
                draft_model(mx.array([q[K - 1]])[None], cache=draft_cache)

            # 5) Emit accepted tokens, then the bonus correction.
            for t in q[:m]:
                if emitted >= max_tokens:
                    return
                yield t
                emitted += 1
                if stats:
                    stats.tokens_emitted += 1
                if t == eos_id:
                    return
            if emitted >= max_tokens:
                return
            yield bonus
            emitted += 1
            if stats:
                stats.tokens_emitted += 1
            if bonus == eos_id:
                return
            b = bonus
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()


def ngram_speculative_generate(
    engine: StreamingEngine,
    drafter,
    prompt: list[int],
    max_tokens: int = 64,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
    eos_id: int | None = None,
    stats: SpecStats | None = None,
    processed_tokens: int = 0,
) -> Generator[int, None, None]:
    """Prompt-lookup (n-gram) speculative decoding over the streaming target.

    A model-free `drafter` (an `NGramDrafter`) proposes draft tokens by finding
    the most recent prior occurrence of the current context suffix and returning
    the tokens that followed; the streamed target verifies them in ONE forward
    pass. This is LOSSLESS greedy: a draft token is accepted only if it equals
    the target's argmax, so the output is bit-identical to plain `generate()`
    regardless of draft quality. When the drafter finds no match it returns [],
    and this round degrades to a single greedy target step.

    Even modest acceptance amortizes the streaming cost of a multi-token verify
    pass (core re-reads + overlapping expert unions) over several emitted tokens,
    and the multi-token verify pass automatically enables M3 speculative expert
    prefetch (`engine._cur_pass_multi`). SpecStats semantics match
    `speculative_generate` so bench reporting is unchanged: `draft_tokens_proposed`
    counts proposed n-gram tokens, `accepted_total` counts accepted ones (the
    bonus correction is not a draft token), `target_passes` counts every target
    forward (verify sweeps AND single greedy fallbacks).

    KVStore lifecycle and `processed_tokens` semantics follow `generate()`;
    `kv_quant`, if given, quantizes the KV cache (unsupported on attention-sink
    architectures; raises ValueError).
    """
    check_kv_quant_support(engine, kv_quant)
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        # Prefill the target over the un-cached suffix; b is the first confirmed
        # token. `context` tracks prompt + everything emitted so far and is what
        # the drafter looks tokens up in.
        logits = engine.forward(
            mx.array(prompt[processed_tokens:])[None], kv=kv)[:, -1, :]
        b = int(mx.argmax(logits, axis=-1).item())
        context = list(prompt)
        context.append(b)

        emitted = 0
        yield b
        emitted += 1
        if stats:
            stats.tokens_emitted += 1
        if b == eos_id:
            return

        while emitted < max_tokens:
            # 1) Draft tokens by prompt-lookup on the current context.
            q = drafter.propose(context)

            if not q:
                # No n-gram match -> single greedy target step. b (not yet in the
                # cache) is fed once, producing the next token.
                nl = engine.forward(mx.array([b])[None], kv=kv)[:, -1, :]
                if stats:
                    stats.target_passes += 1
                nxt = int(mx.argmax(nl, axis=-1).item())
                yield nxt
                emitted += 1
                if stats:
                    stats.tokens_emitted += 1
                context.append(nxt)
                if nxt == eos_id:
                    return
                b = nxt
                continue

            Kq = len(q)
            if stats:
                stats.draft_tokens_proposed += Kq

            # 2) Verify [b, q0..q_{Kq-1}] in one EPHEMERAL target sweep -> Kq+1
            #    logit rows. verify_forward never mutates the persistent kv:
            #    trim-based rollback is unsound for RotatingKVCache (sliding-
            #    window archs like gpt-oss) once it has rotated, so rejected
            #    tokens are simply never committed instead of trimmed away.
            vlog, recs = engine.verify_forward(mx.array([b] + q)[None], kv)
            targ = mx.argmax(vlog[0], axis=-1).tolist()   # targ[i] = argmax after position i
            if stats:
                stats.target_passes += 1

            # 3) Accept the longest prefix where draft == target argmax (lossless).
            m = 0
            for i in range(Kq):
                if q[i] != targ[i]:
                    break
                m += 1
            if stats:
                stats.accepted_total += m
            bonus = int(targ[m])   # correct token after last accepted (m <= Kq, len(targ)=Kq+1)

            # 4) Commit the accepted prefix [b, q0..q_{m-1}] (m+1 tokens) to the
            #    persistent kv. The bonus token is NOT committed — it becomes the
            #    next round's bootstrap b and enters the cache on that pass,
            #    exactly like the round-1 bootstrap.
            engine.commit_verified(kv, recs, m + 1)

            # 5) Emit accepted tokens, then the bonus correction.
            for t in q[:m]:
                if emitted >= max_tokens:
                    return
                yield t
                emitted += 1
                if stats:
                    stats.tokens_emitted += 1
                context.append(t)
                if t == eos_id:
                    return
            if emitted >= max_tokens:
                return
            yield bonus
            emitted += 1
            if stats:
                stats.tokens_emitted += 1
            context.append(bonus)
            if bonus == eos_id:
                return
            b = bonus
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()


def gemma4_mtp_speculative_generate(
    engine: StreamingEngine,
    drafter,
    prompt: list[int],
    max_tokens: int = 64,
    num_draft_tokens: int = 16,
    accept_top_k: int = 1,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
    eos_id: int | None = None,
    stats: SpecStats | None = None,
) -> Generator[int, None, None]:
    """Speculative decoding using a Gemma 4 MTP "assistant" drafter.

    Unlike `speculative_generate` (which drives a normal mlx_lm drafter with
    its own KV cache), the assistant has no KV cache of its own — it
    cross-attends to the target's per-layer-type K/V via `shared_kv_states`
    (`drafter.draft_speculative`). Each verify sweep over the streamed target
    re-derives the next round's `last_hidden`/`shared_kv_states` by slicing
    this pass's outputs at the accepted position `m`, so no extra target
    forward pass is needed for re-bootstrapping.

    KVStore lifecycle and `accept_top_k` semantics follow `speculative_generate`.
    `kv_quant`, if given, quantizes the KV cache entries (unsupported on
    attention-sink architectures; raises ValueError).
    """
    check_kv_quant_support(engine, kv_quant)
    K = num_draft_tokens
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        # Prefill the target; capture the state the drafter cross-attends to.
        logits = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]
        last_hidden = engine.last_hidden_state()[:, -1:, :]
        kv_states = engine.target_kv_states()
        prefix_len = len(prompt)
        b = int(mx.argmax(logits, axis=-1).item())

        emitted = 0
        yield b
        emitted += 1
        if stats:
            stats.tokens_emitted += 1
        if b == eos_id:
            return

        while emitted < max_tokens:
            # 1) Draft K tokens from the current bootstrap b via cross-attention
            #    to the target's shared_kv_states (q_len==1 -> mask=None is exact).
            target_embed = engine.embed_tokens(mx.array([b])[None])
            position_ids = mx.array([[prefix_len]])
            q_arr, _, _ = drafter.draft_speculative(
                target_embed=target_embed,
                target_last_hidden=last_hidden,
                shared_kv_states=kv_states,
                position_ids=position_ids,
                max_draft_tokens=K,
                embed_fn=engine.embed_tokens,
            )
            q = [int(t) for t in q_arr[0].tolist()]
            if stats:
                stats.draft_tokens_proposed += K

            # 2) Verify [b, q0..q_{K-1}] in one target sweep -> K+1 logit rows.
            vlog = engine.forward(mx.array([b] + q)[None], kv=kv)
            targ = mx.argmax(vlog[0], axis=-1).tolist()   # len K+1; targ[i] = argmax after position i
            if stats:
                stats.target_passes += 1

            # 3) Accept the longest prefix of plausible draft tokens.
            V = vlog.shape[-1]
            m = 0
            for i in range(K):
                if accept_top_k <= 1:
                    ok = (q[i] == targ[i])
                elif accept_top_k >= V:
                    ok = True
                else:
                    row = vlog[0, i]                                          # [V]
                    topk = mx.argpartition(row, -accept_top_k)[-accept_top_k:]
                    ok = q[i] in set(topk.tolist())
                if not ok:
                    break
                m += 1
                if stats and q[i] != targ[i]:
                    stats.accepted_offpath += 1
            if stats:
                stats.accepted_total += m
            bonus = int(targ[m])   # correct token after last accepted (m <= K, len(targ)=K+1)

            # 4) Roll back the rejected tail in the target KV.
            #    Target cached b+q0..q_{K-1} (K+1 beyond prefix); keep b+q0..q_{m-1}.
            kv.truncate(K - m)

            # 5) Re-derive next round's drafter inputs by slicing this pass's
            #    outputs at position m (no extra target forward needed).
            last_hidden = engine.last_hidden_state()[:, m:m + 1, :]
            kv_states = {
                t: (k[:, :, :prefix_len + m + 1, :], v[:, :, :prefix_len + m + 1, :])
                for t, (k, v) in engine.target_kv_states().items()
            }
            prefix_len += m + 1

            # 6) Emit accepted tokens, then the bonus correction.
            for t in q[:m]:
                if emitted >= max_tokens:
                    return
                yield t
                emitted += 1
                if stats:
                    stats.tokens_emitted += 1
                if t == eos_id:
                    return
            if emitted >= max_tokens:
                return
            yield bonus
            emitted += 1
            if stats:
                stats.tokens_emitted += 1
            if bonus == eos_id:
                return
            b = bonus
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()


def get_optimal_config(prompt_tokens: int, max_tokens: int) -> dict:
    """Return optimal speculative decoding configuration based on workload.

    Implements workload-aware configuration selection to maximize throughput
    across different task types.

    Args:
        prompt_tokens: Number of tokens in the input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with optimal configuration parameters
    """
    # Short generation - overhead dominates
    if max_tokens < 20:
        return {
            "use_speculative": False,
            "reason": "short_generation_overhead_dominates"
        }

    # Medium task - optimal for speculation
    elif max_tokens <= 50 and prompt_tokens < 50:
        return {
            "use_speculative": True,
            "draft_tokens": 40,
            "budget_mb": 512,
            "reason": "optimal_speculation_range"
        }

    # Long generation - conservative speculation
    elif max_tokens <= 100:
        return {
            "use_speculative": True,
            "draft_tokens": 24,
            "budget_mb": 512,
            "reason": "degraded_acceptance_conservative_draft"
        }

    # Very long generation - minimal speculation
    else:
        return {
            "use_speculative": True,
            "draft_tokens": 16,
            "budget_mb": 512,
            "reason": "long_generation_minimal_draft"
        }


def eagle_speculative_generate(
    engine: StreamingEngine,
    eagle_drafter,
    prompt: list[int],
    max_tokens: int = 64,
    num_draft_tokens: int = 16,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
    eos_id: int | None = None,
    stats: SpecStats | None = None,
) -> Generator[int, None, None]:
    """EAGLE feature-level speculative decoding over the streaming target.

    Instead of a full draft model (0.5B-1.7B params), uses a tiny 2-layer
    feature-level drafter (~0.3 GB) that predicts the target's next hidden
    state from the current hidden state + token embedding.

    Benefits over token-level speculative decoding:
      - No draft model download (~4 GB saved)
      - No draft KV cache management (draft is stateless)
      - Draft step is ~100× cheaper (2 layers vs 24+)
      - Higher acceptance from feature-level prediction

    Args:
        engine: Streaming target engine
        eagle_drafter: EagleDrafter instance
        prompt: Input token IDs
        max_tokens: Maximum tokens to generate
        num_draft_tokens: Draft tokens per round
        kv_budget: KV cache budget
        prefetch: Enable prefetching
        kv: KVStore instance (optional)
        kv_quant: if given, quantizes the KV cache
        eos_id: End-of-sequence token ID
        stats: Statistics tracking object
    """
    check_kv_quant_support(engine, kv_quant)
    K = num_draft_tokens
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        logits = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]
        last_hidden = engine.last_hidden_state()
        b = int(mx.argmax(logits, axis=-1).item())

        emitted = 0
        yield b
        emitted += 1
        if stats:
            stats.tokens_emitted += 1
        if b == eos_id:
            return

        while emitted < max_tokens:
            draft_arr = eagle_drafter.draft_speculative(
                last_hidden=last_hidden[:, -1:, :],
                bootstrap_token=mx.array([[b]]),
                max_draft_tokens=K,
                temp=0.0,
            )[0]
            q = draft_arr[0].tolist()
            if stats:
                stats.draft_tokens_proposed += K

            vlog = engine.forward(mx.array([b] + q)[None], kv=kv)
            targ = mx.argmax(vlog[0], axis=-1).tolist()
            if stats:
                stats.target_passes += 1

            m = 0
            for i in range(K):
                if q[i] == targ[i]:
                    m += 1
                else:
                    break
            if stats:
                stats.accepted_total += m
            bonus = int(targ[m])

            kv.truncate(K - m)

            last_hidden = engine.last_hidden_state()[:, m:m + 1, :]

            for t in q[:m]:
                if emitted >= max_tokens:
                    return
                yield t
                emitted += 1
                if stats:
                    stats.tokens_emitted += 1
                if t == eos_id:
                    return
            if emitted >= max_tokens:
                return
            yield bonus
            emitted += 1
            if stats:
                stats.tokens_emitted += 1
            if bonus == eos_id:
                return
            b = bonus
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()


def adaptive_speculative_generate(
    engine: StreamingEngine,
    draft_model,
    prompt: list[int],
    max_tokens: int = 64,
    initial_draft_tokens: int = 32,
    min_draft_tokens: int = 16,
    max_draft_tokens: int = 48,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    kv_quant: KVQuant | None = None,
    eos_id: int | None = None,
    stats: AdaptiveSpecStats | None = None,
    adjustment_interval: int = 3,
) -> Generator[int, None, None]:
    """Greedy speculative decoding with adaptive draft token adjustment.

    Dynamically adjusts the number of draft tokens based on recent acceptance
    rates to optimize throughput across varying workload characteristics.

    Adaptation strategy:
    - Acceptance > 0.8: Increase draft tokens (be aggressive)
    - Acceptance 0.6-0.8: Maintain current level
    - Acceptance < 0.6: Decrease draft tokens (be conservative)

    Args:
        engine: Streaming target engine
        draft_model: Resident draft model
        prompt: Input token IDs
        max_tokens: Maximum tokens to generate
        initial_draft_tokens: Starting draft token count
        min_draft_tokens: Minimum draft tokens (floor)
        max_draft_tokens: Maximum draft tokens (ceiling)
        kv_budget: KV cache budget
        prefetch: Enable prefetching
        kv: KVStore instance
        kv_quant: if given, quantizes the KV cache entries (unsupported on
            attention-sink architectures; raises ValueError)
        eos_id: End-of-sequence token ID
        stats: Statistics tracking object
        adjustment_interval: Adjust draft tokens every N passes
    """
    from mlx_lm.models.cache import make_prompt_cache

    check_kv_quant_support(engine, kv_quant)
    if stats is None:
        stats = AdaptiveSpecStats()
    stats.current_draft_tokens = initial_draft_tokens

    K = initial_draft_tokens
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = _open_kv_store(engine, tmp.name, kv_budget, prefetch, kv_quant)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        # Prefill both models over the prompt
        logits = engine.forward(mx.array(prompt)[None], kv=kv)[:, -1, :]
        draft_cache = make_prompt_cache(draft_model)
        draft_model(mx.array(prompt)[None], cache=draft_cache)
        b = int(mx.argmax(logits, axis=-1).item())

        emitted = 0
        yield b
        emitted += 1
        stats.tokens_emitted += 1
        if b == eos_id:
            return

        passes_since_adjustment = 0

        while emitted < max_tokens:
            # 1) Draft K tokens from the current bootstrap b
            q: list[int] = []
            di = mx.array([b])[None]
            for _ in range(K):
                dl = draft_model(di, cache=draft_cache)[:, -1, :]
                nx = int(mx.argmax(dl, axis=-1).item())
                q.append(nx)
                di = mx.array([nx])[None]
            stats.draft_tokens_proposed += K

            # 2) Verify [b, q0..q_{K-1}] in one target sweep
            vlog = engine.forward(mx.array([b] + q)[None], kv=kv)
            targ = mx.argmax(vlog[0], axis=-1).tolist()
            stats.target_passes += 1
            passes_since_adjustment += 1

            # 3) Accept the longest prefix where draft == target
            m = 0
            for i in range(K):
                if q[i] == targ[i]:
                    m += 1
                else:
                    break
            bonus = int(targ[m])

            # 4) Track acceptance rate
            acceptance_rate = m / K if K > 0 else 0
            stats.acceptance_history.append(acceptance_rate)

            # 5) Adaptive draft token adjustment
            if passes_since_adjustment >= adjustment_interval and len(stats.acceptance_history) >= 3:
                recent_acceptance = stats.recent_acceptance

                if recent_acceptance > 0.8:
                    # High acceptance - be aggressive
                    new_K = min(max_draft_tokens, K + 4)
                elif recent_acceptance > 0.6:
                    # Medium acceptance - maintain
                    new_K = K
                else:
                    # Low acceptance - be conservative
                    new_K = max(min_draft_tokens, K - 8)

                if new_K != K:
                    K = new_K
                    stats.adjustment_count += 1
                    stats.current_draft_tokens = K

                passes_since_adjustment = 0

            # 6) Roll back the rejected tail in both caches
            kv.truncate(K - m)
            if m < K:
                drop = (K - 1) - m
                for c in draft_cache:
                    c.trim(drop)
            else:
                draft_model(mx.array([q[K - 1]])[None], cache=draft_cache)

            # 7) Emit accepted tokens, then the bonus correction
            for t in q[:m]:
                if emitted >= max_tokens:
                    return
                yield t
                emitted += 1
                stats.tokens_emitted += 1
                if t == eos_id:
                    return
            if emitted >= max_tokens:
                return
            yield bonus
            emitted += 1
            stats.tokens_emitted += 1
            if bonus == eos_id:
                return
            b = bonus
    finally:
        if own_kv:
            kv.close()
            tmp.cleanup()
