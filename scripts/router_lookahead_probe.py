"""Backlog #9 go/no-go: cross-layer router lookahead accuracy (offline).

Hypothesis (community suggestion): the residual stream changes slowly between
layers, so layer L+k's router gate evaluated on layer L's OUTPUT hidden state
predicts layer L+k's true fired experts well enough to drive prefetch — i.e.
true top-8 at L+k lands inside the L-state-predicted widened top-N.

Method: greedy-decode T tokens normally; a recording wrapper around
_moe_attn_and_mix captures, for every single-token decode pass, each layer's
output hidden state and its true fired expert ids (the math is replicated
verbatim from engine._moe_attn_and_mix — single execution, KV mutated once,
output tokens unchanged). Afterwards, for each layer L and lookahead k, the
probe applies layer L+k's own post_attention_layernorm + router gate (both
core-resident, read straight from the pinned core piece) to h_out(L) and
scores the prediction. Page-cache state is irrelevant — accuracy only.

Usage: python router_lookahead_probe.py <packed_root> <budget_gb> <decode_tokens>
"""
import json
import sys
import tempfile
import types
from pathlib import Path

import mlx.core as mx

from nunspark.bench import _load_tokenizer, _encode
from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest

PROMPT = ("Explain, step by step, how a modern operating system schedules "
          "threads across performance and efficiency cores, and what a "
          "userspace developer can do to cooperate with the scheduler.")

LOOKAHEADS = (1, 2, 3)
WIDENS = (8, 10, 12, 16)

# per decode pass: {layer: (h_out [1,1,H], true_fired list[int])}
RECORDS: list[dict[int, tuple[mx.array, list[int]]]] = []


def _recording_moe_attn_and_mix(self, slot, layer, h, mask, cache):
    # Verbatim replica of StreamingEngine._moe_attn_and_mix with recording
    # added for single-token passes (probe-only; drift here only affects the
    # probe, never the engine).
    r = slot.self_attn(slot.input_layernorm(h), mask, cache)
    h = h + r
    x = slot.post_attention_layernorm(h)
    gate_logits = getattr(slot.mlp, self._router_attr)(x)
    inds, scores = self._moe_route(self.args, gate_logits)
    fired = sorted({int(e) for e in inds.reshape(-1).tolist()})
    batch_tokens = inds.shape[0] * inds.shape[1]
    if self._trace_fh is not None:
        self._trace_moe(layer, fired, batch_tokens)
    if self._cur_pass_multi or self._decode_bulk_warm:
        self.cache.warm_bulk(
            Manifest.layer_expert_piece_id(layer, e) for e in fired)
    self._scatter_experts(slot, layer, fired)
    if self._expert_prefetch:
        self._fired_history[layer] = (fired, batch_tokens > 1)
    y = getattr(slot.mlp, self._expert_attr)(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    out = h + y
    if batch_tokens == 1:  # decode pass only
        RECORDS[-1][layer] = (out, fired)
    return out


def _predict_topn(engine, target_layer: int, h: mx.array) -> list[int]:
    """Router logits of `target_layer` evaluated on hidden state `h`, experts
    ordered best-first (full ordering; slice for any widen)."""
    core = engine.cache.get(Manifest.layer_core_piece_id(target_layer))
    r = engine._router_attr
    norm_w = core["post_attention_layernorm.weight"]
    x = mx.fast.rms_norm(h, norm_w, engine.args.rms_norm_eps)
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


def main() -> None:
    root = Path(sys.argv[1])
    budget = int(float(sys.argv[2]) * (1 << 30))
    n_decode = int(sys.argv[3])

    manifest = Manifest.load(root / "manifest.json")
    tokenizer = _load_tokenizer(root)
    ids = _encode(tokenizer, PROMPT)

    engine = StreamingEngine(root, manifest, budget_bytes=budget)
    engine._moe_attn_and_mix = types.MethodType(_recording_moe_attn_and_mix, engine)
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_probe_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        RECORDS.append({})  # prefill records nothing (batch_tokens > 1)
        logits = _prefill(engine, ids, kv, 1024)
        tok = int(mx.argmax(logits, axis=-1).item())
        for _ in range(n_decode - 1):
            RECORDS.append({})
            logits = engine.forward(mx.array([[tok]]), kv=kv)
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        passes = [r for r in RECORDS if r]
        n_layers = manifest.num_layers
        # stats[k][widen] -> [sum_recall, n, n_fully_contained]
        stats = {k: {w: [0.0, 0, 0] for w in WIDENS} for k in LOOKAHEADS}
        # depth buckets for k=1, widen=12: early/mid/late thirds
        depth = {b: [0.0, 0] for b in ("early", "mid", "late")}
        for rec in passes:
            for layer, (h_out, _fired) in rec.items():
                for k in LOOKAHEADS:
                    tgt = layer + k
                    if tgt >= n_layers or tgt not in rec:
                        continue
                    true8 = set(rec[tgt][1])
                    order = _predict_topn(engine, tgt, h_out)
                    for w in WIDENS:
                        pred = set(order[:w])
                        hit = len(true8 & pred) / len(true8)
                        s = stats[k][w]
                        s[0] += hit
                        s[1] += 1
                        s[2] += true8 <= pred
                    if k == 1:
                        b = ("early", "mid", "late")[min(2, 3 * layer // n_layers)]
                        pred12 = set(order[:12])
                        depth[b][0] += len(true8 & pred12) / len(true8)
                        depth[b][1] += 1

        out = {
            "probe": "cross-layer router lookahead accuracy (backlog #9)",
            "model": str(root),
            "budget_gb": budget / (1 << 30),
            "prompt_tokens": len(ids),
            "decode_passes_recorded": len(passes),
            "num_experts_per_tok": int(engine.args.num_experts_per_tok),
            "recall": {
                f"k={k}": {
                    f"top{w}": {
                        "mean_recall": round(s[0] / s[1], 4),
                        "fully_contained_rate": round(s[2] / s[1], 4),
                        "n": s[1],
                    }
                    for w, s in stats[k].items() if s[1]
                }
                for k in LOOKAHEADS
            },
            "k1_top12_recall_by_depth": {
                b: round(v[0] / v[1], 4) for b, v in depth.items() if v[1]
            },
        }
        print(json.dumps(out, indent=2))
        with open("scripts/results/router_lookahead_probe.json", "w") as f:
            json.dump(out, f, indent=1)
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()


if __name__ == "__main__":
    main()
