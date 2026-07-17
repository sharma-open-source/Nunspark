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
    ``num_draft_tokens`` (K) is the CAP on how many following tokens may be
    proposed.

    When ``adaptive`` (default on), the drafter tracks a live proposal
    length ``k_cur`` (starting at the K cap) and adjusts it based on how well
    recent rounds verified, via ``observe()``. This is purely a COST
    optimization: because acceptance is lossless-by-construction
    (``ngram_speculative_generate`` only ever accepts a proposed token that
    equals the target's own argmax), a shorter, longer, or disabled proposal
    can never change the emitted tokens -- only how much verify work each
    round costs. When ``adaptive`` is False, ``propose()`` always caps at the
    fixed ``num_draft_tokens`` and ``k_cur`` never changes, matching the
    original (pre-M2) behavior exactly.

    M2 v2 policy (multiplicative decrease, disable-to-zero, periodic
    re-probe -- see docs/plan5-tensorfold-adoptions.md):

    - full/near-full acceptance (``accepted >= proposed - 1`` AND
      ``accepted >= 2``): double ``k_cur``, capped at ``num_draft_tokens``.
      Requiring ``accepted >= 2`` means a lucky 1-of-1 or 1-of-2 round can't
      trigger growth on its own.
    - poor round (``accepted * 4 < proposed``, i.e. under a quarter accepted;
      the multiplication form keeps the branch reachable at small ``proposed``
      -- with integer division ``accepted < proposed // 4`` could never fire
      at ``proposed <= 3``, which left the drafter unable to disable from the
      default floor of 2 and oscillating 2..16 forever) while ``k_cur`` is at
      the floor (``min_draft_tokens``): DISABLE the drafter by setting
      ``k_cur = 0``. While disabled, ``propose()`` returns ``[]`` immediately
      without scanning the context, so the generate loop degrades to plain
      greedy steps -- zero verify-pass tax for a drafter that isn't earning
      its keep at the smallest useful proposal length.
    - poor round while ``k_cur`` is above the floor: halve ``k_cur``, floored
      at ``min_draft_tokens``.
    - otherwise: hold ``k_cur`` steady.

    Re-probing: while disabled, the drafter counts ``propose()`` calls. Every
    ``reprobe_every`` such calls, instead of the cheap no-scan short-circuit,
    it runs one real proposal capped at ``min_draft_tokens`` (a probe). The
    caller's subsequent ``observe()`` call for that round decides the
    outcome: full/near-full acceptance re-enables (and can grow further, same
    rule as above, starting from ``min_draft_tokens``); anything else leaves
    it disabled for another ``reprobe_every``-call cycle. A probe round with
    no match at all (``proposed == 0``) is a no-op, same as elsewhere, and
    the drafter stays disabled.
    """

    def __init__(self, max_ngram: int = 3, num_draft_tokens: int = 16,
                 adaptive: bool = True, min_draft_tokens: int = 2,
                 reprobe_every: int = 50):
        if max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        if min_draft_tokens < 1:
            raise ValueError("min_draft_tokens must be >= 1")
        if reprobe_every < 1:
            raise ValueError("reprobe_every must be >= 1")
        self.max_ngram = max_ngram
        self.num_draft_tokens = num_draft_tokens
        self.adaptive = adaptive
        self.min_draft_tokens = min_draft_tokens
        self.reprobe_every = reprobe_every
        # k_cur is the live proposal-length cap; starts at the full K cap.
        # k_cur == 0 means "disabled" (adaptive only).
        self.k_cur = num_draft_tokens
        self.k_min_seen = num_draft_tokens
        self.k_max_seen = num_draft_tokens
        # Reporting counters.
        self.disabled_rounds = 0  # propose() calls short-circuited while disabled
        self.probes = 0  # re-probe attempts fired while disabled
        # Internal bookkeeping for the disable/re-probe cycle.
        self._disabled_calls = 0
        self._probing = False

    def propose(self, context: list[int]) -> list[int]:
        """Propose up to K draft tokens by prompt-lookup.

        Tries the longest suffix n-gram first (length min(max_ngram, len-1))
        down to 1. For each n, scans ``context`` from the end backwards for
        the most recent EARLIER occurrence of the suffix n-gram and returns
        up to K tokens that followed that occurrence, where K is ``k_cur``
        when ``adaptive`` (else the fixed ``num_draft_tokens``). Returns
        ``[]`` when no suffix n-gram has a prior match -- the caller then
        falls back to a single greedy step for that round. O(len(context))
        per call.

        When ``adaptive`` and the drafter is disabled (``k_cur == 0``), this
        returns ``[]`` immediately WITHOUT scanning the context, unless this
        call lands on a re-probe boundary (every ``reprobe_every`` disabled
        calls), in which case it runs one real proposal capped at
        ``min_draft_tokens``.
        """
        n_ctx = len(context)
        if self.adaptive and self.k_cur == 0:
            self._disabled_calls += 1
            if self._disabled_calls % self.reprobe_every == 0:
                self.probes += 1
                self._probing = True
                K = self.min_draft_tokens
            else:
                self.disabled_rounds += 1
                return []
        else:
            K = self.k_cur if self.adaptive else self.num_draft_tokens
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

    def observe(self, proposed: int, accepted: int) -> None:
        """Update ``k_cur`` after a verify round proposed ``proposed`` tokens
        and had ``accepted`` of them accepted. See the class docstring for
        the full multiplicative decrease / disable-to-zero / re-probe policy.

        A no-op when not adaptive, or when ``proposed == 0`` (an empty
        proposal -- including a re-probe that found no match -- carries no
        acceptance signal; the drafter's disabled/enabled state and ``k_cur``
        are left unchanged).
        """
        if not self.adaptive or proposed == 0:
            self._probing = False
            return
        was_probing = self._probing
        self._probing = False
        # The value k_cur is effectively acting at for this round: k_cur
        # itself normally, or min_draft_tokens when this was a re-probe
        # (k_cur is 0 -- disabled -- during a probe).
        base = self.min_draft_tokens if was_probing else self.k_cur
        if accepted >= proposed - 1 and accepted >= 2:
            self.k_cur = min(self.num_draft_tokens, base * 2)
        elif accepted * 4 < proposed:
            if base <= self.min_draft_tokens:
                self.k_cur = 0  # disable
            else:
                self.k_cur = max(self.min_draft_tokens, base // 2)
        else:
            self.k_cur = base  # hold
        if self.k_cur < self.k_min_seen:
            self.k_min_seen = self.k_cur
        if self.k_cur > self.k_max_seen:
            self.k_max_seen = self.k_cur
