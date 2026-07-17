"""Plan 5 M3b-1 — batched-decode engine core.

Gate intent (docs/plan5-m3-design.md §11 + §14 orchestrator amendment): each row
of `batched_generate` must reproduce what `generate()` produces for that prompt
run sequentially — validating the key-padding mask, the left-pad positions, and
the shared-offset KV.

IMPORTANT empirical finding (documented, reproduced in *stock* mlx_lm with no
NunSpark engine): mlx's batched decode kernel is lane-nondeterministic past a
handful of decode steps — two *identical* rows in one batch diverge from each
other deep in decode (inter-lane logit diffs ~0.2), which flips argmaxes. This is
NOT a NunSpark bug and NOT a mask/position error; the §14 amendment predicted
exactly this. It means "row == sequential exactly for the whole run" does NOT
hold at fixture scale for B>1 beyond the near-tie horizon.

What DOES hold exactly, and is what these tests assert, isolates mask/position
correctness cleanly:
  * B=1 (no batch-lane effect) is bit-identical to generate() for the full run.
  * A padded row's PREFILL last-position logits match the unpadded single-row
    prefill to fp round-off (~1e-6) with identical argmax -> the pad mask +
    left-pad positions are correct (this is the crisp mask gate).
  * Every row's FIRST generated token equals its sequential run.
  * Batched tracks sequential token-for-token over a common prefix; the first
    divergence (if any within the horizon) is never at token 0 -> no structural
    corruption a mask/offset bug would cause (that corrupts from token 0).
"""
import mlx.core as mx
import pytest

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import (
    generate, batched_generate, _open_kv_store, _prefill_batched,
)


# Conservatively below the earliest observed batched-kernel near-tie divergence
# on these seeded fixtures (qwen3_moe: step ~4). A structural mask/offset bug
# corrupts from token 0, so this margin firmly catches it.
MIN_EXACT = 3


def _build_engine(model_dir, tmp_path):
    out = tmp_path / "packed.nunspark"
    pack(model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    return StreamingEngine(out, manifest, budget_bytes=10**9)


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _assert_tracks_sequential(got, refs):
    """Each batched row tracks its sequential run over a common prefix; any
    divergence is a deep batched-kernel near-tie (index >= MIN_EXACT), never a
    token-0 structural corruption."""
    for i, (g, r) in enumerate(zip(got, refs)):
        assert len(g) == len(r), f"row {i} length {len(g)} != {len(r)}"
        cpl = _common_prefix_len(g, r)
        assert g[0] == r[0], f"row {i} FIRST token {g[0]} != sequential {r[0]}"
        assert cpl >= MIN_EXACT, (
            f"row {i} diverges from sequential at token {cpl} (< {MIN_EXACT}); "
            f"a divergence this early indicates a mask/offset bug, not a "
            f"deep-decode near-tie")


# --- B=1: batched == generate() bit-identical, full run (pad-free path) ------

def test_b1_batched_matches_generate_llama(tiny_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 9]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        got = batched_generate(engine, [prompt], max_tokens=24, temp=0.0)
    finally:
        engine.close()
    assert got == [ref]


def test_b1_batched_matches_generate_qwen3_moe(tiny_qwen3_moe_model_dir, tmp_path):
    prompt = [3, 7, 42, 1, 9, 2]
    engine = _build_engine(tiny_qwen3_moe_model_dir, tmp_path)
    try:
        ref = generate(engine, prompt, max_tokens=24, temp=0.0)
        got = batched_generate(engine, [prompt], max_tokens=24, temp=0.0)
    finally:
        engine.close()
    assert got == [ref]


# --- CRISP MASK GATE: padded-row prefill == unpadded single-row prefill ------
# A wrong key-padding mask or wrong left-pad positions corrupts the padded
# prompt's final-position logits, flipping the argmax; this asserts they match.

def _check_prefill_mask(engine, prompts):
    import tempfile
    L = max(len(p) for p in prompts)
    pad = [L - len(p) for p in prompts]
    ids = mx.array([[0] * (L - len(p)) + list(p) for p in prompts])
    kv = _open_kv_store(engine, tempfile.mkdtemp(), 10**12, True, None)
    try:
        plog = _prefill_batched(engine, ids, kv, mx.array(pad))   # [B, V]
        for i, p in enumerate(prompts):
            kv1 = _open_kv_store(engine, tempfile.mkdtemp(), 10**12, True, None)
            try:
                slog = _prefill_batched(engine, mx.array([p]), kv1, None)[0]
            finally:
                kv1.close()
            diff = mx.max(mx.abs(plog[i] - slog)).item()
            a_batched = int(mx.argmax(plog[i]).item())
            a_seq = int(mx.argmax(slog).item())
            assert a_batched == a_seq, (
                f"row {i} (pad={pad[i]}) prefill argmax {a_batched} != "
                f"single-row {a_seq}: key-pad mask / left-pad positions wrong")
            assert diff < 1e-3, (
                f"row {i} (pad={pad[i]}) prefill logit diff {diff} too large "
                f"(expected ~fp round-off from RoPE position shift)")
    finally:
        kv.close()


def test_prefill_mask_matches_single_row_llama(tiny_model_dir, tmp_path):
    prompts = [[3, 7, 42, 1, 9, 2, 6], [5, 2, 11], [1, 1, 2, 3], [9, 8, 7, 6, 5, 4]]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        _check_prefill_mask(engine, prompts)
    finally:
        engine.close()


def test_prefill_mask_matches_single_row_qwen3_moe(tiny_qwen3_moe_model_dir, tmp_path):
    prompts = [[3, 7, 42, 1, 9, 2, 6], [5, 2, 11], [1, 1, 2, 3], [9, 8, 7, 6, 5, 4]]
    engine = _build_engine(tiny_qwen3_moe_model_dir, tmp_path)
    try:
        _check_prefill_mask(engine, prompts)
    finally:
        engine.close()


# --- equal-length batch B in {2,4}: pad-free path tracks sequential ----------

@pytest.mark.parametrize("B", [2, 4])
def test_equal_length_batch_tracks_sequential_llama(B, tiny_model_dir, tmp_path):
    base = [[3, 7, 42, 1, 9], [5, 2, 11, 8, 4], [1, 1, 2, 3, 5], [9, 8, 7, 6, 5]]
    prompts = base[:B]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        refs = [generate(engine, p, max_tokens=16, temp=0.0) for p in prompts]
        got = batched_generate(engine, prompts, max_tokens=16, temp=0.0)
    finally:
        engine.close()
    _assert_tracks_sequential(got, refs)


@pytest.mark.parametrize("B", [2, 4])
def test_equal_length_batch_tracks_sequential_qwen3_moe(B, tiny_qwen3_moe_model_dir, tmp_path):
    base = [[3, 7, 42, 1, 9], [5, 2, 11, 8, 4], [1, 1, 2, 3, 5], [9, 8, 7, 6, 5]]
    prompts = base[:B]
    engine = _build_engine(tiny_qwen3_moe_model_dir, tmp_path)
    try:
        refs = [generate(engine, p, max_tokens=16, temp=0.0) for p in prompts]
        got = batched_generate(engine, prompts, max_tokens=16, temp=0.0)
    finally:
        engine.close()
    _assert_tracks_sequential(got, refs)


# --- mixed-length batch: exercises key-padding mask + left-pad positions ------

@pytest.mark.parametrize("B", [2, 4])
def test_mixed_length_batch_tracks_sequential_llama(B, tiny_model_dir, tmp_path):
    base = [[3, 7, 42, 1, 9, 2, 6], [5, 2, 11], [1, 1, 2, 3], [9, 8, 7, 6, 5, 4]]
    prompts = base[:B]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        refs = [generate(engine, p, max_tokens=16, temp=0.0) for p in prompts]
        got = batched_generate(engine, prompts, max_tokens=16, temp=0.0)
    finally:
        engine.close()
    _assert_tracks_sequential(got, refs)


@pytest.mark.parametrize("B", [2, 4])
def test_mixed_length_batch_tracks_sequential_qwen3_moe(B, tiny_qwen3_moe_model_dir, tmp_path):
    base = [[3, 7, 42, 1, 9, 2, 6], [5, 2, 11], [1, 1, 2, 3], [9, 8, 7, 6, 5, 4]]
    prompts = base[:B]
    engine = _build_engine(tiny_qwen3_moe_model_dir, tmp_path)
    try:
        refs = [generate(engine, p, max_tokens=16, temp=0.0) for p in prompts]
        got = batched_generate(engine, prompts, max_tokens=16, temp=0.0)
    finally:
        engine.close()
    _assert_tracks_sequential(got, refs)


# --- ragged completion: a row hits eos early and freezes; others continue -----

def _trunc_at_eos(seq, eos_id, limit):
    for i, t in enumerate(seq[:limit]):
        if t == eos_id:
            return seq[: i + 1], True
    return seq[:limit], False


def test_ragged_completion_freezes_finished_rows(tiny_model_dir, tmp_path):
    prompts = [[3, 7, 42, 1, 9, 2, 6], [5, 2, 11], [1, 1, 2, 3], [9, 8, 7, 6, 5, 4]]
    N = 16
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        refs = [generate(engine, p, max_tokens=N, temp=0.0) for p in prompts]
        # eos = row0's first token: row0 stops immediately (len 1) while rows
        # that don't emit it early keep decoding -> a genuinely ragged batch.
        eos_id = refs[0][0]
        got = batched_generate(engine, prompts, max_tokens=N, temp=0.0, eos_id=eos_id)
    finally:
        engine.close()

    # Ragged: rows finished at different lengths (early finishers vs full-budget).
    assert len({len(g) for g in got}) >= 2
    finished_early = [i for i in range(len(got)) if len(got[i]) < N]
    assert finished_early, "expected at least one row to hit eos early"

    for i, (g, r) in enumerate(zip(got, refs)):
        expected, hit = _trunc_at_eos(r, eos_id, N)
        if len(expected) <= MIN_EXACT:
            # stop is inside the exact horizon -> batched must match sequential
            # exactly (a late finisher is unaffected by earlier finishers).
            assert g == expected, f"row {i} ragged output {g} != expected {expected}"
            if hit:
                assert g[-1] == eos_id
        else:
            # beyond the horizon a deep near-tie may flip; only require the
            # exact-horizon prefix to track sequential.
            assert _common_prefix_len(g, r) >= MIN_EXACT, (
                f"row {i} diverges inside the exact horizon (mask/offset bug)")
    # A finished row's output is frozen at its eos and never resumes.
    for i in finished_early:
        assert got[i][-1] == eos_id


# --- temp>0 smoke: runs and returns correct shapes ---------------------------

def test_temperature_smoke_shapes(tiny_model_dir, tmp_path):
    prompts = [[3, 7, 42, 1, 9], [5, 2, 11], [1, 1, 2, 3, 5, 6]]
    engine = _build_engine(tiny_model_dir, tmp_path)
    try:
        got = batched_generate(engine, prompts, max_tokens=12, temp=0.8)
    finally:
        engine.close()
    assert len(got) == 3
    assert all(len(g) == 12 for g in got)
    assert all(all(isinstance(t, int) for t in g) for g in got)


# --- sliding-window (RotatingKVCache) rejection: raises before any forward ----

def test_rotating_arch_rejected_before_forward(tiny_gpt_oss_model_dir, tmp_path, monkeypatch):
    engine = _build_engine(tiny_gpt_oss_model_dir, tmp_path)

    def _boom(*a, **k):
        raise AssertionError("forward ran despite rotating-arch rejection")

    monkeypatch.setattr(engine, "forward", _boom)
    try:
        with pytest.raises(ValueError, match="sliding-window"):
            batched_generate(engine, [[3, 7, 42, 1, 9]], max_tokens=8)
    finally:
        engine.close()
