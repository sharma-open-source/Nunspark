"""M2 (Plan 5): adaptive proposal shrink for the n-gram drafter.

Adaptivity only changes proposal LENGTH, never which tokens are proposed --
and acceptance in `ngram_speculative_generate` is lossless-by-construction
(a proposed token is only ever accepted if it equals the target's own
argmax). So adaptive vs fixed-K vs plain greedy must all produce bit-identical
output; the only thing adaptivity can change is how much verify work each
round costs. These tests cover: the `observe()` policy in isolation,
`propose()` respecting `k_cur`, and the bit-identical invariant on the same
tiny-model fixture used by tests/test_garbage_drafter_invariant.py.
"""
import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate, ngram_speculative_generate, SpecStats
from nunspark.ngram_drafter import NGramDrafter


VOCAB = 320  # matches TINY_CONFIG vocab_size in tests/conftest.py


def _build_engine(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


# --- observe() policy --------------------------------------------------

def test_observe_shrinks_on_poor_round():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True)
    assert d.k_cur == 16
    d.observe(proposed=16, accepted=1)  # 1 < 16 // 4 == 4 -> poor round
    assert d.k_cur == 8


def test_observe_shrink_floors_at_min_draft_tokens():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True, min_draft_tokens=2)
    d.k_cur = 3
    d.observe(proposed=16, accepted=0)  # poor round: 3 // 2 == 1, floored to 2
    assert d.k_cur == 2
    d.observe(proposed=16, accepted=0)  # poor round AT the floor -> disable
    assert d.k_cur == 0


def test_observe_grow_requires_at_least_two_accepted():
    # A 1-of-2 round must not trigger growth even though accepted >= proposed - 1.
    d = NGramDrafter(num_draft_tokens=16, adaptive=True)
    d.k_cur = 2
    d.observe(proposed=2, accepted=1)  # accepted >= proposed-1 (1>=1) but accepted < 2
    assert d.k_cur == 2  # holds, not poor (1 < 2//4==0 is False) either


def test_observe_grows_on_near_full_acceptance():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True)
    d.k_cur = 4
    d.observe(proposed=4, accepted=4)  # full acceptance
    assert d.k_cur == 8
    d.observe(proposed=8, accepted=7)  # accepted >= proposed - 1 -> near-full
    assert d.k_cur == 16  # capped at num_draft_tokens


def test_observe_grow_caps_at_num_draft_tokens():
    d = NGramDrafter(num_draft_tokens=6, adaptive=True)
    d.k_cur = 6
    d.observe(proposed=6, accepted=6)
    assert d.k_cur == 6  # already at cap, doubling would overshoot


def test_observe_holds_in_between():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True)
    d.k_cur = 8
    # accepted=3 of proposed=8: not near-full (>= 7), not poor (< 2) -> hold
    d.observe(proposed=8, accepted=3)
    assert d.k_cur == 8


def test_observe_noop_when_not_adaptive():
    d = NGramDrafter(num_draft_tokens=16, adaptive=False)
    assert d.k_cur == 16
    d.observe(proposed=16, accepted=0)
    assert d.k_cur == 16


def test_observe_noop_when_proposed_zero():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True)
    d.k_cur = 8
    d.observe(proposed=0, accepted=0)
    assert d.k_cur == 8


def test_observe_tracks_k_min_max_seen():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True)
    assert d.k_min_seen == 16
    assert d.k_max_seen == 16
    d.observe(proposed=16, accepted=0)  # poor -> shrink to 8
    d.observe(proposed=8, accepted=0)   # poor -> shrink to 4
    assert d.k_min_seen == 4
    assert d.k_max_seen == 16
    d.observe(proposed=4, accepted=4)   # full -> grow to 8
    d.observe(proposed=8, accepted=8)   # full -> grow to 16
    assert d.k_max_seen == 16
    assert d.k_min_seen == 4


def test_observe_disables_on_poor_round_at_floor():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True, min_draft_tokens=2)
    d.k_cur = d.min_draft_tokens  # already at the floor
    d.observe(proposed=8, accepted=0)  # poor round at floor -> disable
    assert d.k_cur == 0
    assert d.k_min_seen == 0


def test_observe_disables_at_floor_with_small_proposal():
    # Regression (G-M2 v3 root cause): at the floor the proposal IS
    # min_draft_tokens=2, and with the old integer-division condition
    # (accepted < proposed // 4 == 0) a 0-of-2 round could never register as
    # poor -- the drafter oscillated k=2..16 forever instead of disabling.
    # The multiplication form (accepted * 4 < proposed) fires: 0*4 < 2.
    d = NGramDrafter(num_draft_tokens=16, adaptive=True, min_draft_tokens=2)
    d.k_cur = d.min_draft_tokens
    d.observe(proposed=2, accepted=0)  # 0-of-2 at the floor -> disable
    assert d.k_cur == 0


def test_observe_disable_is_noop_when_not_at_floor():
    d = NGramDrafter(num_draft_tokens=16, adaptive=True, min_draft_tokens=2)
    d.k_cur = 4  # above floor
    d.observe(proposed=8, accepted=0)  # poor round -> halve, not disable
    assert d.k_cur == 2


# --- disable-to-zero + re-probe -----------------------------------------

def test_propose_returns_empty_without_scanning_while_disabled():
    pattern = list(range(100, 120))
    context = pattern + pattern
    d = NGramDrafter(max_ngram=3, num_draft_tokens=16, adaptive=True,
                      min_draft_tokens=2, reprobe_every=50)
    d.k_cur = 0  # disabled
    got = d.propose(context)
    assert got == []
    assert d.disabled_rounds == 1
    assert d.probes == 0


def test_reprobe_fires_after_reprobe_every_disabled_calls():
    pattern = list(range(100, 120))
    context = pattern + pattern
    d = NGramDrafter(max_ngram=3, num_draft_tokens=16, adaptive=True,
                      min_draft_tokens=2, reprobe_every=5)
    d.k_cur = 0  # disabled
    for _ in range(4):
        got = d.propose(context)
        assert got == []
    assert d.disabled_rounds == 4
    assert d.probes == 0
    # 5th disabled call is the re-probe boundary: runs a real proposal
    # capped at min_draft_tokens.
    probed = d.propose(context)
    assert probed != []
    assert len(probed) == d.min_draft_tokens
    assert d.probes == 1
    assert d.disabled_rounds == 4  # unchanged: the probe call isn't a short-circuit


def test_reenable_on_successful_probe():
    pattern = list(range(100, 120))
    context = pattern + pattern
    d = NGramDrafter(max_ngram=3, num_draft_tokens=16, adaptive=True,
                      min_draft_tokens=2, reprobe_every=1)
    d.k_cur = 0  # disabled
    probed = d.propose(context)  # immediate re-probe (reprobe_every=1)
    assert len(probed) == d.min_draft_tokens
    d.observe(proposed=len(probed), accepted=len(probed))  # full acceptance
    assert d.k_cur == d.min_draft_tokens * 2  # re-enabled and grown
    assert d.k_cur > 0


def test_disable_again_on_failed_probe():
    pattern = list(range(100, 120))
    context = pattern + pattern
    # min_draft_tokens=4 so proposed // 4 == 1 -> accepted=0 is unambiguously
    # "poor" (with min_draft_tokens=2, proposed // 4 == 0 and no accepted
    # count can be < 0, so a probe there can never register as poor).
    d = NGramDrafter(max_ngram=3, num_draft_tokens=16, adaptive=True,
                      min_draft_tokens=4, reprobe_every=1)
    d.k_cur = 0  # disabled
    probed = d.propose(context)
    assert len(probed) == d.min_draft_tokens
    d.observe(proposed=len(probed), accepted=0)  # poor probe
    assert d.k_cur == 0  # stays disabled
    # Next call is disabled again (not an immediate re-probe -- the probe
    # itself consumed this cycle's boundary; the counter keeps advancing).
    got = d.propose(context)
    assert got != []  # reprobe_every=1 means every call is a probe boundary


# --- propose() respects k_cur -------------------------------------------

def test_propose_caps_at_k_cur_after_shrink():
    # A long repeated pattern so a long earlier match exists (K=16 worth of
    # tokens follow the first occurrence of the suffix).
    pattern = list(range(100, 120))  # 20 distinct tokens
    context = pattern + pattern  # second copy's suffix matches the first
    d = NGramDrafter(max_ngram=3, num_draft_tokens=16, adaptive=True)
    # Un-shrunk: proposal length is capped by num_draft_tokens (16), and the
    # match has plenty of follow-on tokens available.
    full = d.propose(context)
    assert len(full) == 16

    d.k_cur = 3
    shrunk = d.propose(context)
    assert len(shrunk) == 3
    assert shrunk == full[:3]


def test_propose_uses_fixed_num_draft_tokens_when_not_adaptive():
    pattern = list(range(100, 120))
    context = pattern + pattern
    d = NGramDrafter(max_ngram=3, num_draft_tokens=16, adaptive=False)
    d.k_cur = 3  # would matter if adaptive; ignored here
    got = d.propose(context)
    assert len(got) == 16


# --- bit-identical: adaptive vs fixed vs greedy -------------------------

def test_ngram_adaptive_bit_identical_to_fixed_and_greedy(tiny_model_dir, tmp_path):
    # Prompt has repetition so the n-gram drafter actually fires proposals
    # (and so acceptance/rejection mix triggers both shrink and hold/grow).
    prompt = [3, 7, 42, 1, 3, 7, 42, 1, 9, 9, 3, 7, 42, 1]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=32, temp=0.0)

        stats_fixed = SpecStats()
        drafter_fixed = NGramDrafter(max_ngram=3, num_draft_tokens=6, adaptive=False)
        got_fixed = list(ngram_speculative_generate(
            engine, drafter_fixed, prompt, max_tokens=32, stats=stats_fixed))

        stats_adaptive = SpecStats()
        drafter_adaptive = NGramDrafter(max_ngram=3, num_draft_tokens=6, adaptive=True)
        got_adaptive = list(ngram_speculative_generate(
            engine, drafter_adaptive, prompt, max_tokens=32, stats=stats_adaptive))
    finally:
        engine.close()

    assert got_fixed == ref
    assert got_adaptive == ref
    assert len(got_fixed) == 32
    assert len(got_adaptive) == 32
    # sanity: proposals actually fired on both arms (repetition triggers matches)
    assert stats_fixed.draft_tokens_proposed > 0
    assert stats_adaptive.draft_tokens_proposed > 0


def test_ngram_adaptive_bit_identical_gpt_oss(tiny_gpt_oss_model_dir, tmp_path):
    # Same invariant on the sliding-window (RotatingKVCache) fixture, matching
    # the coverage tests/test_garbage_drafter_invariant.py gives the plain
    # (non-adaptive) ngram path.
    prompt = [3, 7, 42, 1, 9] * 3
    out = tmp_path / "tiny_gpt_oss.nunspark"
    pack(tiny_gpt_oss_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        drafter = NGramDrafter(max_ngram=3, num_draft_tokens=5, adaptive=True)
        stats = SpecStats()
        got = list(ngram_speculative_generate(
            engine, drafter, prompt, max_tokens=24, stats=stats))
    finally:
        engine.close()
    assert got == ref
    assert len(got) == 24
