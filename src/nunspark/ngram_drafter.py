from __future__ import annotations


class NGramDrafter:
    """Prompt-lookup (n-gram) drafter for speculative decoding.

    Apoorv Saxena's prompt-lookup technique (as in transformers'
    ``prompt_lookup_num_tokens``): to draft the next tokens, find the MOST
    RECENT prior occurrence of the current suffix n-gram inside the context
    (prompt + everything generated so far) and propose the tokens that
    followed it. There is no model and no weight cost -- the proposal is a
    pure lookup into the context itself, so it is ideal for the weight-bound
    streaming regime where any draft model would compete for the same SSD.

    ``max_ngram`` bounds the longest suffix tried: the drafter matches the
    longest suffix n-gram first (most specific -> highest-quality proposal)
    and falls back to shorter n-grams down to 1 when no match is found.
    ``num_draft_tokens`` (K) caps how many following tokens are proposed.
    """

    def __init__(self, max_ngram: int = 3, num_draft_tokens: int = 16):
        if max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        self.max_ngram = max_ngram
        self.num_draft_tokens = num_draft_tokens

    def propose(self, context: list[int]) -> list[int]:
        """Propose up to K draft tokens by prompt-lookup.

        Tries the longest suffix n-gram first (length min(max_ngram, len-1))
        down to 1. For each n, scans ``context`` from the end backwards for
        the most recent EARLIER occurrence of the suffix n-gram and returns
        up to K tokens that followed that occurrence. Returns ``[]`` when no
        suffix n-gram has a prior match -- the caller then falls back to a
        single greedy step for that round. O(len(context)) per call.
        """
        n_ctx = len(context)
        K = self.num_draft_tokens
        # An n-gram match needs at least one earlier token to follow, so the
        # longest usable suffix is n_ctx - 1.
        max_n = min(self.max_ngram, n_ctx - 1)
        for n in range(max_n, 0, -1):
            suffix = context[n_ctx - n:]
            # Scan for the most recent PRIOR occurrence: candidate start
            # positions run from the latest possible (just before the suffix)
            # down to 0. The occurrence at start == n_ctx - n is the suffix
            # itself, so we start one earlier.
            for start in range(n_ctx - n - 1, -1, -1):
                if context[start:start + n] == suffix:
                    follow = context[start + n:start + n + K]
                    if follow:
                        return follow
        return []
