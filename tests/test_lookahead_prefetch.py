"""Plan 7 M1: online router-lookahead expert prefetch.

Lookahead is a read-only prediction that drives speculative prefetch only, so
the token stream MUST be identical with the flag on vs off (constraint 1), the
prediction must never block the compute path (a non-resident target core is
skipped via a non-blocking peek, not a load — constraint 3), and archs whose
router cannot be replicated from the core dict (glm_moe_dsa's MoEGate correction
bias) must silently no-op (constraint 6). These tests use the tiny seeded MoE
fixtures — never a real model.
"""
import mlx.core as mx

from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import generate


def _pack(model_dir, tmp_path):
    out = tmp_path / "model.nunspark"
    pack(model_dir, out)
    return out, Manifest.load(out / "manifest.json")


# --- (a) bit-identity: lookahead ON == OFF ---------------------------------

def _run_identity(model_dir, tmp_path, prompt=(3, 7, 42, 1, 9), max_tokens=10):
    out, manifest = _pack(model_dir, tmp_path)
    prompt = list(prompt)

    e_off = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        off = generate(e_off, prompt, max_tokens=max_tokens, temp=0.0)
    finally:
        e_off.close()

    e_on = StreamingEngine(out, manifest, budget_bytes=10**9,
                           lookahead_prefetch=True, lookahead_topn=4)
    try:
        on = generate(e_on, prompt, max_tokens=max_tokens, temp=0.0)
        issued = e_on.lookahead_issued
        supported = e_on._lookahead_supported
    finally:
        e_on.close()

    assert on == off, "lookahead must not change the greedy token stream"
    return issued, supported


def test_lookahead_bit_identical_qwen3_moe(tiny_qwen3_moe_model_dir, tmp_path):
    issued, supported = _run_identity(tiny_qwen3_moe_model_dir, tmp_path)
    assert supported is True          # qwen3_moe router IS replicable
    assert issued > 0                 # and the decode passes actually issued prefetch


def test_lookahead_bit_identical_gpt_oss(tiny_gpt_oss_model_dir, tmp_path):
    issued, supported = _run_identity(tiny_gpt_oss_model_dir, tmp_path)
    assert supported is True          # gpt_oss (biased-linear router) IS replicable
    assert issued > 0


def test_lookahead_bit_identical_glm_moe_dsa(tiny_glm_moe_dsa_model_dir, tmp_path):
    # (d) glm_moe_dsa: chosen outcome is the GRACEFUL SKIP path (MoEGate carries
    # e_score_correction_bias + noaux_tc group selection, not a plain logits
    # top-k router). Assert the verdict is False and nothing was ever issued.
    issued, supported = _run_identity(tiny_glm_moe_dsa_model_dir, tmp_path)
    assert supported is False
    assert issued == 0


# --- (b) non-blocking skip: a missing target core issues nothing -----------

def test_missing_core_skips_and_counts(tiny_qwen3_moe_model_dir, tmp_path):
    out, manifest = _pack(tiny_qwen3_moe_model_dir, tmp_path)
    prompt = [3, 7, 42, 1, 9]

    e_off = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        off = generate(e_off, prompt, max_tokens=8, temp=0.0)
    finally:
        e_off.close()

    engine = StreamingEngine(out, manifest, budget_bytes=10**9,
                             lookahead_prefetch=True, lookahead_topn=4)
    # Force every target core to look non-resident: the prediction must skip it
    # (counting the miss) rather than block on a load, and never touch output.
    engine.cache.peek = lambda pid: None
    try:
        on = generate(engine, prompt, max_tokens=8, temp=0.0)
    finally:
        skipped = engine.lookahead_skipped_core_missing
        issued = engine.lookahead_issued
        engine.close()

    assert on == off                        # no stall / no output change
    assert skipped > 0                      # skip counter fired
    assert issued == 0                       # nothing issued when the core is absent


def test_peek_is_inert(tiny_qwen3_moe_model_dir, tmp_path):
    # peek must not count hits/misses, reorder LRU/MRU, or trigger a load.
    out, manifest = _pack(tiny_qwen3_moe_model_dir, tmp_path)
    engine = StreamingEngine(out, manifest, budget_bytes=10**9)
    try:
        pid = Manifest.layer_core_piece_id(1)
        engine.cache.get(pid)               # make it genuinely resident
        hits0, misses0 = engine.cache.hits, engine.cache.misses
        order0 = engine.cache.resident_ids

        assert engine.cache.peek(pid) is not None
        assert engine.cache.peek("layer_9999_core") is None   # absent -> None, no load

        assert (engine.cache.hits, engine.cache.misses) == (hits0, misses0)
        assert engine.cache.resident_ids == order0            # no LRU/MRU reorder
    finally:
        engine.close()


# --- (c) math guard: engine prediction == the probe's reference math -------

def _reference_predict_order(engine, target_layer, h):
    """The probe's `_predict_topn` reference math (router_lookahead_probe.py),
    replicated here to guard the engine's prediction against silent drift.
    Experts ordered best-first (full ordering; slice for any widen)."""
    core = engine.cache.get(Manifest.layer_core_piece_id(target_layer))
    r = engine._router_attr
    x = mx.fast.rms_norm(h, core["post_attention_layernorm.weight"],
                         engine.args.rms_norm_eps)
    wkey = f"mlp.{r}.weight"
    if f"mlp.{r}.scales" in core:
        cfg = engine._module_quant(f"model.layers.{target_layer}.mlp.{r}")
        logits = mx.quantized_matmul(
            x, core[wkey], scales=core[f"mlp.{r}.scales"],
            biases=core.get(f"mlp.{r}.biases"), transpose=True,
            group_size=cfg["group_size"], bits=cfg["bits"],
            mode=cfg.get("mode", "affine"),
        )
    else:
        logits = x @ core[wkey].T
    order = mx.argsort(-logits.reshape(-1).astype(mx.float32))
    return [int(i) for i in order.tolist()]


def test_prediction_matches_reference_math(tiny_qwen3_moe_model_dir, tmp_path):
    out, manifest = _pack(tiny_qwen3_moe_model_dir, tmp_path)
    topn = 3
    engine = StreamingEngine(out, manifest, budget_bytes=10**9,
                             lookahead_prefetch=True, lookahead_topn=topn)
    try:
        mx.random.seed(1)
        h = mx.random.normal((1, 1, engine.args.hidden_size))
        tgt = 2
        core = engine.cache.get(Manifest.layer_core_piece_id(tgt))
        pred = set(engine._predict_experts(tgt, h, core))
        ref = set(_reference_predict_order(engine, tgt, h)[:topn])
        assert len(pred) == topn
        assert pred == ref              # engine top-N == probe reference top-N
    finally:
        engine.close()
