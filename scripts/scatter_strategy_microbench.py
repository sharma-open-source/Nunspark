"""Backlog #10 micro-bench: which buffer strategy kills the scatter tax?

Loads ONE real layer's core+expert pieces from a pack (so shapes/dtypes/quant
are exactly the engine's), then times 48 simulated "layers" of each strategy:

  A  current        fresh mx.zeros full-size (128-expert) bufs + 8 row scatters + eval
  B  persistent     reuse one full-size buf set, 8 row scatters, NO re-zero, eval
                    (only our dict references the bufs -> scatter may donate)
  B2 persistent+ref same as B but a second reference is held across the update
                    (mimics the slot keeping last layer's arrays alive -> forced copy)
  C  compact        mx.stack the 8 fired pieces' rows into a (8, ...) buf + eval
                    (16x fewer bytes; needs inds remap 0..7 in the real engine)

Usage: python scatter_strategy_microbench.py <packed_root> [n_layers_sim]
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.manifest import Manifest

N_FIRED = 8


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
    # preload 16 experts of layer 0 so every strategy reads resident arrays
    pieces = {}
    for e in range(16):
        pieces[e] = engine.cache.get(Manifest.layer_expert_piece_id(0, e))
    full_shapes = {(p, c): (getattr(sw, p)[c].shape, getattr(sw, p)[c].dtype)
                   for p, c in subkeys}
    per_layer_mb = sum(
        getattr(sw, p)[c].nbytes for p, c in subkeys) / (1 << 20)

    def fired_for(i):  # rotate so consecutive "layers" differ like real routing
        return [(i * 3 + j) % 16 for j in range(N_FIRED)]

    def run(name, fn, n=n_sim):
        # one warmup iteration outside the timer (allocator settles)
        fn(0)
        mx.synchronize()
        t0 = time.monotonic()
        for i in range(1, n + 1):
            fn(i)
        mx.synchronize()
        dt = time.monotonic() - t0
        return {"strategy": name, "ms_per_layer": round(1000 * dt / n, 2),
                "ms_per_token_48_layers": round(1000 * dt / n * 48, 0)}

    results = []

    # A: current engine behavior
    def strat_a(i):
        bufs = {k: mx.zeros(s, dtype=d) for k, (s, d) in full_shapes.items()}
        for row, e in enumerate(fired_for(i)):
            piece = pieces[e]
            for proj, comp in subkeys:
                bufs[(proj, comp)][e] = piece[f"mlp.{attr}.{proj}.{comp}"]
        mx.eval(list(bufs.values()))
    results.append(run("A_current_zeros_scatter", strat_a))

    # B: persistent bufs, no re-zero, unique reference
    persist = {k: mx.zeros(s, dtype=d) for k, (s, d) in full_shapes.items()}
    mx.eval(list(persist.values()))

    def strat_b(i):
        for e in fired_for(i):
            piece = pieces[e]
            for proj, comp in subkeys:
                persist[(proj, comp)][e] = piece[f"mlp.{attr}.{proj}.{comp}"]
        mx.eval(list(persist.values()))
    results.append(run("B_persistent_unique_ref", strat_b))

    # B2: persistent bufs but an extra ref held across layers (slot-like)
    persist2 = {k: mx.zeros(s, dtype=d) for k, (s, d) in full_shapes.items()}
    mx.eval(list(persist2.values()))
    extra_ref = {}

    def strat_b2(i):
        for k in persist2:
            extra_ref[k] = persist2[k]        # what slot.update effectively does
        for e in fired_for(i):
            piece = pieces[e]
            for proj, comp in subkeys:
                persist2[(proj, comp)][e] = piece[f"mlp.{attr}.{proj}.{comp}"]
        mx.eval(list(persist2.values()))
    results.append(run("B2_persistent_slot_ref_held", strat_b2))

    # C: compact (k, ...) buffers via stack
    def strat_c(i):
        bufs = {}
        for proj, comp in subkeys:
            bufs[(proj, comp)] = mx.stack(
                [pieces[e][f"mlp.{attr}.{proj}.{comp}"] for e in fired_for(i)])
        mx.eval(list(bufs.values()))
    results.append(run("C_compact_stack", strat_c))

    out = {
        "probe": "scatter strategy micro-bench (backlog #10)",
        "model": str(root),
        "full_buffer_mb_per_layer": round(per_layer_mb, 1),
        "n_fired": N_FIRED,
        "n_layers_simulated": n_sim,
        "results": results,
    }
    print(json.dumps(out, indent=2))
    with open("scripts/results/scatter_strategy_microbench.json", "w") as f:
        json.dump(out, f, indent=1)
    engine.close()


if __name__ == "__main__":
    main()
