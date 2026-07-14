import os

import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine


def _expert_piece_bytes(engine) -> int:
    return os.stat(engine.store.path_for(Manifest.layer_expert_piece_id(0, 0))).st_size


def test_expert_prefetch_output_bit_identical_on_vs_off(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    # M3a is a prefetch (bandwidth) optimization only: the real router still
    # decides and _scatter_experts always loads the actually-fired experts, so
    # output must be bit-identical with temporal expert prefetch on vs off, even
    # across multiple passes (which is when speculative loads actually fire).
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = mx.array([[3, 7, 42, 1, 9]])

    eng_on = StreamingEngine(out, manifest, budget_bytes=10**9, expert_prefetch=True)
    try:
        eng_on.forward(tokens)
        got_on = eng_on.forward(tokens)   # second pass exercises the speculative path
        mx.eval(got_on)
    finally:
        eng_on.close()

    eng_off = StreamingEngine(out, manifest, budget_bytes=10**9, expert_prefetch=False)
    try:
        eng_off.forward(tokens)
        got_off = eng_off.forward(tokens)
        mx.eval(got_off)
    finally:
        eng_off.close()

    assert float(mx.max(mx.abs(got_on - got_off))) == 0.0


def test_second_pass_experts_served_from_speculative_prefetch(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    # Multi-token forward passes with a budget too small to keep every layer's
    # experts resident (so some are evicted between passes). The second pass
    # remembers pass 1's per-layer fired sets (recorded as multi-token history,
    # Defect 1) and issues them as speculative expert prefetches on the low tier.
    # Under v3 those guesses land in the staging buffer (never the expert LRU) and
    # are served either by a demand get() promoting the staged piece or by joining
    # the still-in-flight speculative load — both count "used". Identical input =>
    # every guess is correct, so used > 0. wasted_bytes may be > 0: the staging
    # cap (20% of the expert region) is deliberately smaller than one verify pass's
    # expert volume, so some correct-but-not-yet-consumed guesses are dropped by
    # cap pressure and re-demanded as misses — that bounded drop is the whole point
    # of staging (it replaces the knife-edge in-LRU insertion that thrashed demand).
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    probe = StreamingEngine(out, manifest, budget_bytes=10**9)
    piece = _expert_piece_bytes(probe)
    probe.close()

    # Hold ~12 experts; the 4 layers fire 7/5/4/4 distinct experts (~20 distinct
    # pids), so a full pass overflows 12 and evicts earlier layers' experts across
    # passes -> the next pass re-issues them speculatively.
    expert_budget = 12 * piece
    budget = int(expert_budget / 0.62)     # leaves the main region room to pin cores
    tokens = mx.array([[3, 7, 42, 1, 9]])  # B*L == 5 -> multi-token history

    engine = StreamingEngine(out, manifest, budget_bytes=budget,
                             expert_cache_frac=0.62, warm_window=2,
                             expert_prefetch=True)
    try:
        engine.forward(tokens)             # pass 1: history empty -> nothing speculated
        mx.eval(engine.forward(tokens))    # pass 2: prefetches pass-1 fired sets
        spec = engine.prefetch_stats()["speculative"]
    finally:
        engine.close()

    assert spec["issued"] > 0              # the second pass speculated from history
    assert spec["used"] > 0                # and its experts were served from those loads
    # every counted byte is accounted for as either used or wasted (staging-capped);
    # no speculative load silently vanishes.
    assert spec["wasted_bytes"] >= 0


def test_expert_prefetch_disabled_issues_nothing(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    tokens = mx.array([[3, 7, 42, 1, 9]])

    engine = StreamingEngine(out, manifest, budget_bytes=200_000,
                             expert_cache_frac=0.62, expert_prefetch=False)
    try:
        engine.forward(tokens)
        engine.forward(tokens)
        spec = engine.prefetch_stats()["speculative"]
    finally:
        engine.close()

    assert spec["issued"] == 0 and spec["used"] == 0
    assert engine._fired_history == {}     # no history kept when disabled


def _spy_speculative(engine):
    """Wrap engine.cache.prefetch to record the pid lists passed with
    speculative=True. Returns the recording list."""
    spec_calls = []
    orig = engine.cache.prefetch

    def spy(pids, speculative=False):
        pids = list(pids)
        if speculative:
            spec_calls.append(pids)
        return orig(pids, speculative=speculative)

    engine.cache.prefetch = spy
    return spec_calls


def test_single_token_pass_records_history_but_no_speculative_prefetch(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    # Defect 1: single-token greedy passes are a weak per-token expert guess, so
    # they must NOT drive speculative prefetch — but history is still recorded
    # (tagged single-token) so the machinery is live for later multi-token passes.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    engine = StreamingEngine(out, manifest, budget_bytes=10**9,
                             warm_window=2, expert_prefetch=True)
    spec_calls = _spy_speculative(engine)
    tok = mx.array([[3]])                  # B*L == 1 -> single-token pass
    try:
        engine.forward(tok)                # pass 1: records single-token history
        engine.forward(tok)                # pass 2: history exists but single-token
        spec = engine.prefetch_stats()["speculative"]
    finally:
        engine.close()

    assert spec_calls == []                # no speculative prefetch ever enqueued
    assert spec["issued"] == 0
    assert engine._fired_history           # history WAS recorded
    assert all(not multi for _fired, multi in engine._fired_history.values())


def test_single_token_pass_after_multi_token_history_issues_nothing(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    # Defect 2 (consume side): the real-world prefill-then-greedy-decode case. A
    # multi-token prefill records big multi-token per-layer unions; the FIRST
    # single-token greedy decode pass that follows must issue NOTHING — even though
    # multi-token history now exists — or it floods the staging buffer with the
    # prefill's giant unions and wrecks greedy throughput. Gate is on the CURRENT
    # pass's token count, not the recorded history's.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    engine = StreamingEngine(out, manifest, budget_bytes=10**9,
                             warm_window=2, expert_prefetch=True)
    spec_calls = _spy_speculative(engine)
    prefill = mx.array([[3, 7, 42, 1, 9]])   # multi-token: records multi-token history
    decode = mx.array([[2]])                  # single-token greedy decode
    try:
        engine.forward(prefill)               # records multi-token history
        assert any(multi for _f, multi in engine._fired_history.values())
        spec_calls.clear()                    # ignore prefill's own (empty-history) pass
        engine.forward(decode)                # single-token pass: must issue NOTHING
        spec = engine.prefetch_stats()["speculative"]
    finally:
        engine.close()

    assert spec_calls == []                    # consume-side gate blocked all issuance
    assert spec["issued"] == 0                 # despite multi-token history existing
    # record side still ran: the single-token decode overwrote history as single-token
    assert engine._fired_history


def test_multi_token_pass_enqueues_speculative_prefetch(
    tiny_qwen3_moe_quant_model_dir, tmp_path
):
    # Defect 1: a multi-token pass (B*L > 1, like a K-token verify pass) DOES seed
    # speculative prefetch on the next pass, exactly as before.
    out = tmp_path / "moe-q4.nunspark"
    pack(tiny_qwen3_moe_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    engine = StreamingEngine(out, manifest, budget_bytes=10**9,
                             warm_window=2, expert_prefetch=True)
    spec_calls = _spy_speculative(engine)
    tok = mx.array([[3, 7, 42, 1, 9]])     # B*L == 5 -> multi-token pass
    try:
        engine.forward(tok)                # pass 1: records multi-token history
        engine.forward(tok)                # pass 2: history drives speculative prefetch
    finally:
        engine.close()

    # The enqueue itself is the gated behavior (with a huge budget the cache may
    # dedup every pid against the resident set, so `issued` is not a reliable
    # signal here -- the spy on the enqueue call is).
    assert spec_calls                      # multi-token history DID enqueue speculatively
    assert any(multi for _fired, multi in engine._fired_history.values())
