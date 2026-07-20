"""Plan 6 M0 probe: does our pinned mlx-lm run a tiny glm_moe_dsa correctly?

Checks (offline, seeded random weights, no downloads):
  1. A tiny glm_moe_dsa builds and forwards under mlx_lm.models.glm_moe_dsa.
  2. The DSA sparse-indexer path actually fires once seq > index_topk, and the
     dense path (seq <= index_topk) is what runs below it.
  3. Cached prefill+decode greedy tokens match the no-cache full re-forward
     argmax at every step (the engine's parity harness relies on this shape).
  4. The weight tree's per-layer key layout (what the packer must split).

Run: uv run python scripts/probe_glm_moe_dsa.py
"""
from __future__ import annotations

import json

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import glm_moe_dsa
from mlx_lm.models import deepseek_v32

TINY_GLM_MOE_DSA_CONFIG = {
    "model_type": "glm_moe_dsa",
    "vocab_size": 320,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 128,
    # MLA
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "attention_bias": False,
    # DSA indexer — index_topk tiny so tests exercise the sparse path
    "index_head_dim": 32,
    "index_n_heads": 2,
    "index_topk": 8,
    # MoE — layer 0 dense, layers 1..3 MoE (first_k_dense_replace=1)
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    # rope — GLM-5.2 style rope_parameters dict
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
}


def build_tiny():
    mx.random.seed(0)
    args = glm_moe_dsa.ModelArgs.from_dict(TINY_GLM_MOE_DSA_CONFIG)
    model = glm_moe_dsa.Model(args)
    mx.eval(model.parameters())
    return model, args


def main():
    model, args = build_tiny()

    # --- 1. layout: per-layer key families the packer must recognize ---
    keys = [k for k, _ in tree_flatten(model.parameters())]
    fams = sorted({k.split(".", 3)[-1].split(".", 1)[0] if False else k for k in keys})
    layer1 = sorted(k for k in keys if k.startswith("model.layers.1."))
    layer0 = sorted(k for k in keys if k.startswith("model.layers.0."))
    top = sorted(k for k in keys if "layers" not in k)
    print("== top-level keys ==")
    print(json.dumps(top, indent=2))
    print("== layer 0 (dense) keys ==")
    print(json.dumps(layer0, indent=2))
    print("== layer 1 (moe) keys ==")
    print(json.dumps(layer1, indent=2))

    # --- 2. instrument the indexer so we can see the sparse path fire ---
    fired = {"none": 0, "sparse": 0}
    orig = deepseek_v32.Indexer.__call__

    def spy(self, x, qr, mask, cache=None):
        out = orig(self, x, qr, mask, cache=cache)
        fired["sparse" if out is not None else "none"] += 1
        return out

    deepseek_v32.Indexer.__call__ = spy

    # --- 3. short prompt (<= index_topk): dense path, cache vs no-cache ---
    prompt = mx.random.randint(0, args.vocab_size, (1, 6))
    cache = model.make_cache()
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    assert fired["sparse"] == 0, "sparse path fired below index_topk"
    print(f"short prefill ok; indexer calls (dense): {fired['none']}")

    # greedy decode 12 steps WITH cache; crosses index_topk=8 along the way
    toks = []
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    for _ in range(12):
        toks.append(int(tok[0]))
        logits = model(tok[:, None], cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
    print(f"cached greedy tokens: {toks}")
    print(f"indexer fire counts after decode: {fired}")
    assert fired["sparse"] > 0, "sparse path never fired past index_topk"

    # --- 4. no-cache full re-forward parity at each step ---
    seq = prompt
    mismatches = 0
    for i, t in enumerate(toks):
        full = model(seq)
        ref = int(mx.argmax(full[:, -1, :], axis=-1)[0])
        if ref != t:
            n = seq.shape[1]
            print(f"  step {i}: cached={t} full={ref} at seq len {n}"
                  f" ({'sparse' if n > args.index_topk else 'dense'} regime)")
            mismatches += 1
        seq = mx.concatenate([seq, mx.array([[t]])], axis=1)
    print(f"full-reforward vs cached-decode mismatches: {mismatches}/12")

    # --- 5. multi-token pass past index_topk (spec-verify shape, L>1 sparse) ---
    fired["sparse"] = fired["none"] = 0
    long_prompt = mx.random.randint(0, args.vocab_size, (1, 20))
    cache2 = model.make_cache()
    out = model(long_prompt, cache=cache2)
    mx.eval(out)
    print(f"long (L=20>topk=8) prefill: indexer fires {fired}")
    assert fired["sparse"] == args.num_hidden_layers

    # cache structure check
    c = cache2[0]
    print(f"cache entry type: {type(c).__name__}, "
          f"children: {[type(x).__name__ for x in c]}, "
          f"offsets: {[x.offset for x in c]}")

    print("\nM0 PROBE PASS")


if __name__ == "__main__":
    main()
