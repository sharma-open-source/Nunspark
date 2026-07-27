"""Plan 8 / M0 — compact fired-only expert scatter: numerics go/no-go.

Backlog #16 measured decode as barrier-bound: `_scatter_experts` scattering ~8
fired experts into full-size [num_experts, ...] persistent buffers is the #1
decode bucket (36%). Candidate (a) is to scatter into compact [k, ...] buffers
with the router's `inds` remapped to 0..k-1 (~16x fewer bytes).

Per (token, selected expert) the gathered weight rows are numerically identical
whether the buffer holds k experts or num_experts, so the switch matmul output
SHOULD be bit-identical. The RISK (the whole reason for this gate before any
engine change): MLX's SwitchGLU / quantized_matmul may choose different tiling
or reduction order at expert-dim k vs num_experts, perturbing fp16 rounding.

This test prototypes compact scatter as an instance monkeypatch (fresh compact
buffers per call — M0 is numerics-only; the persistent-slice perf form is M1)
and asserts its logits are BIT-IDENTICAL to the stock streamed engine across a
multi-token prefill AND several single-token decode steps, on qwen3_moe and
gpt_oss, fp16 and 4-bit. Bit-identity (`==`, not allclose) is the bar: this is
the same streamed math with reindexed gathers, so anything less than exact means
the tiling risk is real and candidate (a) needs a different formulation (or is
dead) — recorded either way in backlog #16.

Run: uv run pytest tests/test_compact_scatter_numerics.py
"""
import tempfile
import types

import mlx.core as mx
from mlx.utils import tree_unflatten

from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest
from nunspark.packer import pack

PROMPT_TOKENS = [3, 7, 42, 1, 9, 2, 5, 11]
N_DECODE = 8

_SUBKEY_PROJS = ("gate_proj", "up_proj", "down_proj")
_SUBKEY_COMPS = ("weight", "scales", "biases", "bias")


def _subkeys(sw):
    return [(proj, comp) for proj in _SUBKEY_PROJS for comp in _SUBKEY_COMPS
            if hasattr(sw, proj) and comp in getattr(sw, proj)]


def _compact_moe_attn_and_mix(self, slot, layer, h, mask, cache):
    """Prototype of the compact-scatter forward. Mirrors the real
    StreamingEngine._moe_attn_and_mix (engine.py:662-722) exactly EXCEPT the
    scatter + expert call, which use compact [k, ...] buffers and 0..k-1 remapped
    inds. Prefetch/lookahead/trace side effects are dropped (they don't affect
    output; M0 tests numerics only)."""
    r = slot.self_attn(slot.input_layernorm(h), mask, cache)
    h = h + r
    x = slot.post_attention_layernorm(h)

    gate_logits = getattr(slot.mlp, self._router_attr)(x)
    inds, scores = self._moe_route(self.args, gate_logits)

    fired = sorted({int(e) for e in inds.reshape(-1).tolist()})
    attr = self._expert_attr
    sw = getattr(slot.mlp, attr)
    subkeys = _subkeys(sw)

    # num_experts from the ROUTER output, NOT the expert weight: the engine reuses
    # ONE slot object across all layers, and slot.update() below overwrites its
    # expert weight with a compact [k, ...] buffer — so a later layer that read
    # weight.shape[0] would see the previous layer's k, not the true count. The
    # router always projects to exactly num_experts logits, and it is read before
    # any compaction, so it is the stable source.
    num_experts = gate_logits.shape[-1]

    # global expert id -> compact row 0..k-1. Non-fired ids map to 0; they never
    # appear in `inds` (fired is exactly the distinct set of inds), so the 0 is
    # unreachable, never gathered.
    lut_host = [0] * num_experts
    for j, e in enumerate(fired):
        lut_host[e] = j
    lut = mx.array(lut_host, dtype=inds.dtype)
    inds_local = lut[inds]

    # Compact scatter: fresh [k, ...] buffers, fired rows written at 0..k-1.
    k = len(fired)
    compact = {}
    for proj, comp in subkeys:
        full = getattr(sw, proj)[comp]
        compact[(proj, comp)] = mx.zeros((k,) + tuple(full.shape[1:]), dtype=full.dtype)
    for j, e in enumerate(fired):
        piece = self.cache.get(Manifest.layer_expert_piece_id(layer, e))
        for proj, comp in subkeys:
            compact[(proj, comp)][j] = piece[f"mlp.{attr}.{proj}.{comp}"]
    mx.eval(list(compact.values()))
    flat = {f"mlp.{attr}.{proj}.{comp}": buf for (proj, comp), buf in compact.items()}
    slot.update(tree_unflatten(list(flat.items())))

    y = getattr(slot.mlp, attr)(x, inds_local)
    mix = (y * scores[..., None]).sum(axis=-2)
    if self._moe_mix_cast:
        mix = mix.astype(y.dtype)
    if self._shared_experts_attr is not None:
        mix = mix + getattr(slot.mlp, self._shared_experts_attr)(x)
    return h + mix


def _decode_logits(engine, tokens, n_decode):
    """Prefill `tokens`, then greedily decode n_decode steps; return the list of
    per-step logit vectors (prefill's last-position logits first, then each
    decode step's)."""
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_compact_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    out = []
    try:
        logits = _prefill(engine, tokens, kv, 1024)
        out.append(mx.array(logits))
        tok = int(mx.argmax(logits, axis=-1).item())
        for _ in range(n_decode):
            logits = engine.forward(mx.array([[tok]]), kv=kv)[:, -1, :]
            mx.eval(logits)
            out.append(mx.array(logits))
            tok = int(mx.argmax(logits, axis=-1).item())
        return out
    finally:
        kv.close()
        tmp.cleanup()


def _assert_compact_bit_identical(model_dir, tmp_path):
    packed = tmp_path / "compact.nunspark"
    pack(model_dir, packed)
    manifest = Manifest.load(packed / "manifest.json")
    assert manifest.has_piece(Manifest.layer_core_piece_id(0)), "expected selective MoE pack"

    stock = StreamingEngine(packed, manifest, budget_bytes=10**9, wire_limit=False)
    try:
        ref = _decode_logits(stock, PROMPT_TOKENS, N_DECODE)
    finally:
        stock.close()

    compact = StreamingEngine(packed, manifest, budget_bytes=10**9, wire_limit=False)
    compact._moe_attn_and_mix = types.MethodType(_compact_moe_attn_and_mix, compact)
    try:
        got = _decode_logits(compact, PROMPT_TOKENS, N_DECODE)
    finally:
        compact.close()

    assert len(ref) == len(got)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape, f"step {i}: shape {a.shape} != {b.shape}"
        # Bit-identity: same streamed math, only the gather is reindexed.
        assert bool(mx.all(a == b).item()), (
            f"step {i}: compact scatter diverged from stock — MLX likely tiles "
            f"the switch matmul differently at expert-dim k vs num_experts "
            f"(max abs diff {float(mx.max(mx.abs(a - b)).item()):.2e}). "
            f"Plan 8 candidate (a) needs a different formulation; record in #16.")


def test_compact_scatter_qwen3_moe_fp16(tiny_qwen3_moe_model_dir, tmp_path):
    _assert_compact_bit_identical(tiny_qwen3_moe_model_dir, tmp_path)


def test_compact_scatter_qwen3_moe_q4(tiny_qwen3_moe_quant_model_dir, tmp_path):
    _assert_compact_bit_identical(tiny_qwen3_moe_quant_model_dir, tmp_path)


def test_compact_scatter_gpt_oss_fp16(tiny_gpt_oss_model_dir, tmp_path):
    _assert_compact_bit_identical(tiny_gpt_oss_model_dir, tmp_path)


def test_compact_scatter_gpt_oss_q4(tiny_gpt_oss_quant_model_dir, tmp_path):
    _assert_compact_bit_identical(tiny_gpt_oss_quant_model_dir, tmp_path)
