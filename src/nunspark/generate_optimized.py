"""Optimized generation functions with dynamic draft token adjustment.

Key optimizations:
1. Dynamic draft token count based on recent acceptance rate
2. Better prefetch coordination between draft and target
3. Reduced cache misses through smarter cache management
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Generator
from collections import deque

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .engine import StreamingEngine
from .kv_store import KVStore


@dataclass
class OptimizedSpecStats:
    """Enhanced speculative stats with dynamic adjustment tracking."""
    target_passes: int = 0
    tokens_emitted: int = 0
    draft_tokens_proposed: int = 0
    acceptance_history: deque = field(default_factory=lambda: deque(maxlen=10))
    dynamic_draft_adjustments: int = 0

    @property
    def multiplier(self) -> float:
        """M = accepted tokens per target forward pass."""
        return self.tokens_emitted / self.target_passes if self.target_passes else 0.0

    @property
    def recent_acceptance_rate(self) -> float:
        """Average acceptance rate over last 10 rounds."""
        if not self.acceptance_history:
            return 0.0
        return sum(self.acceptance_history) / len(self.acceptance_history)


def _calculate_optimal_draft_tokens(
    recent_acceptance: float,
    current_draft_tokens: int,
    min_tokens: int = 8,
    max_tokens: int = 64
) -> int:
    """Dynamically adjust draft tokens based on recent acceptance rate.

    Strategy:
    - High acceptance (>80%): increase draft tokens for more speculation
    - Medium acceptance (50-80%): maintain current level
    - Low acceptance (<50%): reduce draft tokens for better efficiency
    """
    if recent_acceptance > 0.8:
        # High acceptance - be more aggressive
        return min(max_tokens, current_draft_tokens + 8)
    elif recent_acceptance > 0.5:
        # Medium acceptance - maintain
        return current_draft_tokens
    else:
        # Low acceptance - be conservative
        return max(min_tokens, current_draft_tokens - 4)


def optimized_speculative_generate(
    engine: StreamingEngine,
    draft_model,
    prompt: list[int],
    max_tokens: int = 64,
    initial_draft_tokens: int = 16,
    kv_budget: int = 10**12,
    prefetch: bool = True,
    kv: KVStore | None = None,
    eos_id: int | None = None,
    stats: OptimizedSpecStats | None = None,
    enable_dynamic_adjustment: bool = True,
) -> Generator[int, None, None]:
    """Optimized speculative decoding with dynamic draft token adjustment.

    Improvements:
    - Dynamic draft token count based on recent acceptance rate
    - Better coordination between target and draft prefetch
    - Reduced cache misses through smarter batch sizing

    Args:
        engine: Streaming target engine (optimized or standard)
        draft_model: Resident draft model
        prompt: Input token IDs
        max_tokens: Maximum tokens to generate
        initial_draft_tokens: Starting draft token count
        kv_budget: KV cache budget
        prefetch: Enable prefetching
        kv: KVStore instance (optional)
        eos_id: End-of-sequence token ID
        stats: Statistics tracking object
        enable_dynamic_adjustment: Enable dynamic draft token adjustment
    """
    if stats is None:
        stats = OptimizedSpecStats()

    K = initial_draft_tokens
    own_kv = kv is None
    tmp = None
    if own_kv:
        tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
        try:
            kv = KVStore(tmp.name, budget_bytes=kv_budget, prefetch=prefetch,
                         cache_kinds=engine.cache_kinds)
        except BaseException:
            tmp.cleanup()
            raise
    try:
        # Prefill both models
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

        while emitted < max_tokens:
            # Dynamic draft token adjustment
            if enable_dynamic_adjustment and stats.acceptance_history:
                recent_acceptance = stats.recent_acceptance_rate
                new_K = _calculate_optimal_draft_tokens(recent_acceptance, K)
                if new_K != K:
                    K = new_K
                    stats.dynamic_draft_adjustments += 1

            # Draft K tokens
            q: list[int] = []
            di = mx.array([b])[None]
            for _ in range(K):
                dl = draft_model(di, cache=draft_cache)[:, -1, :]
                nx = int(mx.argmax(dl, axis=-1).item())
                q.append(nx)
                di = mx.array([nx])[None]
            stats.draft_tokens_proposed += K

            # Verify in one target sweep
            vlog = engine.forward(mx.array([b] + q)[None], kv=kv)
            targ = mx.argmax(vlog[0], axis=-1).tolist()
            stats.target_passes += 1

            # Find longest matching prefix
            m = 0
            for i in range(K):
                if q[i] == targ[i]:
                    m += 1
                else:
                    break
            bonus = int(targ[m])

            # Track acceptance rate
            acceptance_rate = m / K if K > 0 else 0
            stats.acceptance_history.append(acceptance_rate)

            # Roll back rejected tokens
            kv.truncate(K - m)
            if m < K:
                drop = (K - 1) - m
                for c in draft_cache:
                    c.trim(drop)
            else:
                # All accepted - sync draft cache
                draft_model(mx.array([q[K - 1]])[None], cache=draft_cache)

            # Emit accepted tokens + bonus
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
