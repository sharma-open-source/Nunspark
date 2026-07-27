"""Probe 2 (Plan-8 follow-on / #16 verification): how much decode time is the
per-layer barrier actually WORTH? -- an exact, bit-identical A/B.

The per-layer materialization barrier `mx.eval(h)` (engine.py:_sync_layer, the
`layer_sync` bucket = 19% in #16) is ALREADY conditionally skipped when the model
is fully RAM-resident (`_fully_resident`) -- and the engine certifies the skip is
numerically identical ("mx.eval is scheduling-only -- it never changes numerics").
That makes the recoverable size of this barrier directly measurable: run decode
with the barrier ON vs OFF, experts pre-resident so DISK is out of the loop, and
the token streams MUST match. The gap is the recoverable `layer_sync` tax -- i.e.
the ceiling for M4 (barrier pin/merge). No simulation, no engine surgery.

Requires a packed model that FITS the budget (so residency is real and turning
the barrier off is safe/bit-identical). On a 16 GB Mac use a model that fits
(e.g. an 8B pack); the per-layer tax generalizes to the 30B by layer count. If
the given model does NOT fit, the probe refuses (turning the barrier off under
eviction would be lossy).

ABBA-interleaved to cancel drift; asserts byte-identical greedy streams across
all four runs; warms once before timing so no disk read lands in the timed loop.

Usage: python barrier_headroom_probe.py <packed_root> [n_decode]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest
from nunspark.sysmem import unified_ram_bytes

PROMPT = [3, 7, 42, 1, 9, 2, 5, 11]
# The model must PHYSICALLY fit RAM with headroom (KV + framework + OS), else the
# OS compressor/paging (#15) confounds the timing and "disk out" is a lie. 0.60 of
# unified RAM leaves room below the ~0.75 wire limit. `_fully_resident` alone is a
# BYTE-BUDGET check, not a physical-RAM check -- it is NOT sufficient (this bit the
# first 30B-on-16GB run: 64 GB nominal budget => _fully_resident True while the
# model swapped, on1=278 vs on2=108 ms churn, gap > the whole attributed bucket).
_RAM_SAFE_FRACTION = 0.60


def _decode(engine, n_decode: int, barrier_on: bool):
    """One greedy decode run. barrier_on=True forces the per-layer mx.eval(h)
    (streaming semantics); False skips it (resident semantics). Returns
    (ms_per_token, token_ids). Experts are resident, so this isolates the barrier
    from disk. Timing excludes prefill."""
    engine._fully_resident = not barrier_on   # True skips the layer barrier
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_headroom_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    try:
        logits = _prefill(engine, PROMPT, kv, 1024)
        tok = int(mx.argmax(logits, axis=-1).item())
        toks = []
        mx.synchronize()
        t0 = time.monotonic()
        for _ in range(n_decode):
            logits = engine.forward(mx.array([[tok]]), kv=kv)[:, -1, :]
            tok = int(mx.argmax(logits, axis=-1).item())
            toks.append(tok)
        mx.synchronize()
        dt = time.monotonic() - t0
        return 1e3 * dt / n_decode, toks
    finally:
        kv.close()
        tmp.cleanup()


def main() -> None:
    root = Path(sys.argv[1])
    n_decode = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    manifest = Manifest.load(root / "manifest.json")

    # On-disk footprint ~= the resident nbytes the cache holds (safe upper bound).
    footprint = sum(p.stat().st_size for p in root.rglob("*.safetensors"))
    ram = unified_ram_bytes()
    if ram is not None and footprint > _RAM_SAFE_FRACTION * ram:
        raise SystemExit(
            f"REFUSING: model footprint {footprint/2**30:.1f} GB exceeds "
            f"{_RAM_SAFE_FRACTION:.0%} of {ram/2**30:.1f} GB unified RAM -> it will "
            "NOT physically fit; the OS compressor/paging (#15) would confound the "
            "barrier A/B (that is what invalidated the first 30B-on-16GB run). Use "
            "a packed model that fits (the per-layer tax generalizes by layer count).")

    # Budget just above footprint so _fully_resident is TRUE and honest (never
    # evicts) while staying within the physical-RAM envelope checked above.
    budget = footprint + (2 << 30)
    engine = StreamingEngine(root, manifest, budget_bytes=budget)
    if not engine._fully_resident:
        engine.close()
        raise SystemExit(
            "REFUSING: _fully_resident is False at the sized budget -> turning the "
            "layer barrier off would run under eviction and be LOSSY.")

    # warm: one full decode so every piece is resident before any timed run
    _decode(engine, 8, barrier_on=True)

    # ABBA: on, off, off, on  -> cancels linear drift
    on1, t_on1 = _decode(engine, n_decode, True)
    off1, t_off1 = _decode(engine, n_decode, False)
    off2, t_off2 = _decode(engine, n_decode, False)
    on2, t_on2 = _decode(engine, n_decode, True)
    engine.close()

    # Losslessness: barrier on/off must emit identical greedy tokens.
    assert t_on1 == t_off1 == t_off2 == t_on2, (
        "BIT-IDENTITY VIOLATED: barrier on/off produced different tokens -- the "
        "skip is supposed to be scheduling-only; do not trust the timing.")

    # Validity: the two same-arm runs must agree. A wide spread means #15
    # compressor churn / paging polluted the run (e.g. on1=278 vs on2=108 on the
    # busted 30B-on-16GB run), not a clean barrier isolation -> the gap is noise.
    def _spread(a, b):
        return abs(a - b) / min(a, b)
    on_spread, off_spread = _spread(on1, on2), _spread(off1, off2)
    churn = on_spread > 0.25 or off_spread > 0.25

    on = (on1 + on2) / 2
    off = (off1 + off2) / 2
    gap = on - off
    out = {
        "probe": "layer-barrier headroom A/B (Plan-8 follow-on, #16 verify)",
        "model": str(root),
        "n_layers": manifest.num_layers,
        "n_decode": n_decode,
        "ms_per_token_barrier_ON": round(on, 3),
        "ms_per_token_barrier_OFF": round(off, 3),
        "gap_ms_per_token": round(gap, 3),
        "gap_pct": round(100 * gap / on, 2) if on else 0.0,
        "raw": {"on1": round(on1, 3), "off1": round(off1, 3),
                "off2": round(off2, 3), "on2": round(on2, 3)},
        "same_arm_spread": {"on": round(on_spread, 3), "off": round(off_spread, 3)},
        "churn_suspected": churn,
        "streams_identical": True,
        "verdict": (
            "INVALID: same-arm spread > 25% -> #15 churn/paging polluted the run, "
            "gap is noise; re-run when idle / on a model with more headroom"
            if churn else
            "layer barrier is real recoverable time -> M4 (pin/merge) worth gating"
            if gap > 0.05 * on else
            "layer barrier is ~free -> M4 dead; time was drained compute"),
        "note": "isolates ONLY the per-layer mx.eval(h) (layer_sync bucket); disk "
                "out (resident), scatter eval unchanged in both arms so it cancels. "
                "router_sync (tolist) is structural and NOT toggled here.",
    }
    print(json.dumps(out, indent=2))
    with open("scripts/results/barrier_headroom_probe.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
