from __future__ import annotations

from pathlib import Path

from .archspec import KVQuant, Rotating, cache_offset
from .kv_store import KVStore


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    """Single-slot prompt-prefix reuse for the (single-stream) server.

    Holds the last request's live KVStore plus the token sequence its KV
    represents. `begin()` trims the store to the longest common prefix with
    the incoming prompt and returns only the un-cached suffix to prefill —
    skipping entire streamed layer sweeps, the most expensive operation in
    the disk-bound regime. The cache is an optimization only: any state that
    can't be reused safely is discarded and the request prefills cold.
    """

    def __init__(self, scratch_dir, kv_budget: int, prefetch: bool,
                 cache_kinds: list | None = None,
                 kv_quant: KVQuant | None = None):
        self._dir = Path(scratch_dir)
        self._kv_budget = kv_budget
        self._prefetch = prefetch
        self._cache_kinds = cache_kinds
        self._kv_quant = kv_quant
        self.kv: KVStore | None = None
        self.tokens: list[int] = []
        # The true sequence length is read from a layer whose cache offset
        # always equals it. Rotating layers stop growing at their window, so
        # probe the first non-rotating layer; if every layer rotates the
        # length is unknowable from the caches and reuse is disabled.
        self._probe_layer: int | None = 0
        if cache_kinds is not None:
            for i, kind in enumerate(cache_kinds):
                if not isinstance(kind, Rotating):
                    self._probe_layer = i
                    break
            else:
                self._probe_layer = None

    # ---- public API ----
    def begin(self, prompt_ids: list[int]) -> tuple[KVStore, list[int]]:
        """Trim the slot to the longest reusable prefix of `prompt_ids` and
        return (kv, suffix_ids). The suffix is never empty: a fully-cached
        prompt is trimmed by one token and that token re-fed, so the forward
        pass has logits to sample from."""
        common = _common_prefix_len(self.tokens, prompt_ids)
        common = min(common, len(prompt_ids) - 1)   # always feed >= 1 token

        if (self.kv is None or common <= 0 or self._probe_layer is None
                or self._rotated(len(self.tokens))):
            return self._reset(), list(prompt_ids)

        drop = len(self.tokens) - common
        if drop > 0:
            self.kv.truncate(drop)
        self.tokens = self.tokens[:common]
        return self.kv, list(prompt_ids[common:])

    def commit(self, tokens: list[int]) -> None:
        """Record the sequence generation says the KV now holds. The store's
        actual length can differ (the last sampled token is never fed back;
        an aborted speculative round can leave extra rows) — reconcile by
        trimming the store or the claim, never trusting either alone."""
        if self.kv is None or self._probe_layer is None:
            return
        actual = cache_offset(self.kv.get(self._probe_layer))
        if actual > len(tokens):
            if self._rotated(actual):
                # Trimming a rotated sliding-window cache decrements its
                # offset without restoring evicted rows -> corrupt. Drop the
                # slot; the next request prefills cold.
                self._drop()
                return
            self.kv.truncate(actual - len(tokens))
            actual = len(tokens)
        self.tokens = list(tokens[:actual])

    def close(self) -> None:
        self._drop()

    # ---- internals ----
    def _rotated(self, length: int) -> bool:
        """True if some sliding-window layer has rotated past its window at
        `length` tokens — its evicted rows are gone, so trimming back is
        unsafe. Computed from cache_kinds alone (no cache reloads)."""
        if self._cache_kinds is None:
            return False
        return any(isinstance(k, Rotating) and length >= k.window
                   for k in self._cache_kinds)

    def _drop(self) -> None:
        if self.kv is not None:
            self.kv.close()
            self.kv = None
        self.tokens = []

    def _reset(self) -> KVStore:
        self._drop()
        self.kv = KVStore(self._dir, budget_bytes=self._kv_budget,
                          prefetch=self._prefetch,
                          cache_kinds=self._cache_kinds,
                          kv_quant=self._kv_quant)
        return self.kv
