"""Backlog #10 / Plan 8 #16 M2 micro-bench: size the compact-scatter win.

Loads ONE real layer's core+expert pieces from a pack (so shapes/dtypes/quant
are exactly the engine's), then times 48 simulated "layers" of each strategy,
at EACH k in a set of fired-expert counts (decode k and a verify-union k), so
the scatter bucket is isolated from disk I/O and the FFN matmul:

  A  current        fresh mx.zeros full-size ([num_experts, ...]) bufs + k row
                    scatters + eval  (the pre-#10 behavior; kept for reference)
  B  persistent     reuse one full-size buf set, k row scatters, NO re-zero, eval
                    -> THIS is the shipped DEFAULT path (_scatter_experts full
                    branch): setitem k rows into [num_experts, ...] and eval the
                    whole buffer every layer.
  D  compact_zeros  the shipped M1 COMPACT path (_compact_scatter): fresh
                    [k, ...] mx.zeros + k row scatters at local rows 0..k-1 +
                    eval only the k-row buffers. ~num_experts/k fewer bytes
                    moved+eval'd. Faithful to engine.py's compact branch.
  C  compact_stack  compact via mx.stack (an alternative build of the [k, ...]
                    buffer; kept to check it isn't cheaper than the setitem form).

The head-to-head that sizes M3 is B (default) vs D (compact) at decode k.

Usage: python scatter_strategy_microbench.py <packed_root> [n_layers_sim]
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.manifest import Manifest


def main() -> None:
    root = Path(sys.argv[1])
    n_sim = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    manifest = Manifest.load(root / "manifest.json")
    # tiny budget: we only touch layer 0's pieces
    engine = StreamingEngine(root, manifest, budget_bytes=2 << 30, prefetch=False)
    attr = engine._expert_attr
    sw = getattr(engine._slot.mlp, attr)
    subkeys = [
        (proj, comp)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for comp in ("weight", "scales", "biases", "bias")
        if comp in getattr(sw, proj)
    ]
    # Read num_experts from the FRESH slot's expert weight (row 0 dim) before any
    # compaction mutates it (plan8 #6: never read it from a mutated slot; here the
    # slot is untouched, so shape[0] is the true count).
    num_experts = getattr(sw, "gate_proj")["weight"].shape[0]

    # decode k = qwen3 top-8 (capped for tiny fixtures); verify-union k ~= the
    # 51-65% distinct-expert union a K-token verify pass hits (plan8 #16 / #4).
    decode_k = min(8, num_experts)
    verify_k = max(decode_k + 1, round(0.55 * num_experts))
    verify_k = min(verify_k, num_experts)
    k_values = sorted({decode_k, verify_k})

    # preload every expert of layer 0 so any k-subset reads resident arrays
    pieces = {}
    for e in range(num_experts):
        pieces[e] = engine.cache.get(Manifest.layer_expert_piece_id(0, e))
    full_shapes = {(p, c): (getattr(sw, p)[c].shape, getattr(sw, p)[c].dtype)
                   for p, c in subkeys}
    per_layer_mb = sum(
        getattr(sw, p)[c].nbytes for p, c in subkeys) / (1 << 20)

    def fired_for(i, k):  # rotate so consecutive "layers" differ like real routing
        return [(i * 3 + j) % num_experts for j in range(k)]

    def run(name, k, fn, n=n_sim):
        # one warmup iteration outside the timer (allocator settles)
        fn(0)
        mx.synchronize()
        t0 = time.monotonic()
        for i in range(1, n + 1):
            fn(i)
        mx.synchronize()
        dt = time.monotonic() - t0
        return {"strategy": name, "k": k,
                "ms_per_layer": round(1000 * dt / n, 3),
                "ms_per_token_48_layers": round(1000 * dt / n * 48, 1)}

    results = []

    for k in k_values:
        # A: current engine behavior -- fresh full-size zeros every layer
        def strat_a(i, k=k):
            bufs = {key: mx.zeros(s, dtype=d) for key, (s, d) in full_shapes.items()}
            for e in fired_for(i, k):
                piece = pieces[e]
                for proj, comp in subkeys:
                    bufs[(proj, comp)][e] = piece[f"mlp.{attr}.{proj}.{comp}"]
            mx.eval(list(bufs.values()))
        results.append(run("A_current_zeros_scatter", k, strat_a))

        # B: persistent full-size bufs, no re-zero -- the shipped DEFAULT path
        persist = {key: mx.zeros(s, dtype=d) for key, (s, d) in full_shapes.items()}
        mx.eval(list(persist.values()))

        def strat_b(i, k=k):
            for e in fired_for(i, k):
                piece = pieces[e]
                for proj, comp in subkeys:
                    persist[(proj, comp)][e] = piece[f"mlp.{attr}.{proj}.{comp}"]
            mx.eval(list(persist.values()))
        results.append(run("B_persistent_default_path", k, strat_b))

        # D: compact [k, ...] zeros + local-row scatter -- the shipped M1 COMPACT
        #    path (engine.py _compact_scatter). Faithful: same mx.zeros((k,)+...)
        #    build + setitem[j] + eval only the k rows.
        compact_shapes = {key: ((k,) + tuple(s[1:]), d)
                          for key, (s, d) in full_shapes.items()}

        def strat_d(i, k=k, cshapes=compact_shapes):
            bufs = {key: mx.zeros(s, dtype=d) for key, (s, d) in cshapes.items()}
            for j, e in enumerate(fired_for(i, k)):
                piece = pieces[e]
                for proj, comp in subkeys:
                    bufs[(proj, comp)][j] = piece[f"mlp.{attr}.{proj}.{comp}"]
            mx.eval(list(bufs.values()))
        results.append(run("D_compact_zeros_scatter", k, strat_d))

        # C: compact via mx.stack (alternative build; sanity check vs D)
        def strat_c(i, k=k):
            bufs = {}
            for proj, comp in subkeys:
                bufs[(proj, comp)] = mx.stack(
                    [pieces[e][f"mlp.{attr}.{proj}.{comp}"] for e in fired_for(i, k)])
            mx.eval(list(bufs.values()))
        results.append(run("C_compact_stack", k, strat_c))

    out = {
        "probe": "scatter strategy micro-bench (backlog #10 / plan8 #16 M2)",
        "model": str(root),
        "num_experts": num_experts,
        "full_buffer_mb_per_layer": round(per_layer_mb, 1),
        "k_values": {"decode": decode_k, "verify_union": verify_k},
        "n_layers_simulated": n_sim,
        "note": "head-to-head that sizes M3: B_persistent_default_path vs "
                "D_compact_zeros_scatter at k=decode. Lower ms_per_token is better.",
        "results": results,
    }
    print(json.dumps(out, indent=2))
    with open("scripts/results/scatter_strategy_microbench.json", "w") as f:
        json.dump(out, f, indent=1)
    engine.close()


if __name__ == "__main__":
    main()
