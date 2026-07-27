"""Plan 9 M0 — windowed per-layer eval realizability (go/no-go).

The per-layer `mx.eval(h)` barrier (layer_sync, #16) is worth ~0.27 ms/layer
(Probe 2), but it is MEMORY-LOAD-BEARING under streaming: MLX refcounts a lazy
graph's inputs, so deferring the eval keeps every layer's weights alive and the
PieceCache cannot free them (piece_cache._evict_locked + mx.clear_cache). Dropping
it wholesale on the streaming 30B accumulates the whole token's weight set ->
exceeds budget -> macOS-compressor churn (#15) -> net slower. The realizable lever
is WINDOWED: eval every W-th layer, keeping ~W layers' weights live at once. This
recovers (W-1)/W of the barrier but peak memory grows with W, so W is bounded by
the slack under the wire. This probe finds whether ANY W beats the gate WITHOUT
churn.

Refcounting pins the graph-referenced weights, so windowed eval is bit-identical
to W=1 (mx.eval is scheduling-only) — no explicit cache-pin API needed to measure.

Method: real streaming 30B @ 10 GB wired. Sweep W in {1,2,3,4}, ABBA-symmetric
order [1,2,3,4,4,3,2,1] to cancel drift; page cache flushed and engine rebuilt per
run; 200 greedy decode tokens timed after a short prefill; per-run compressor
delta as the #15 validity check; ALL token streams asserted byte-identical to the
first W=1 run.

GATE: some W gives >=10% median tok/s over W=1 AND stays in the clean compression
band (no churn) AND streams identical. Else Plan 9 stops (recorded dead in #16).

Usage: python windowed_eval_probe.py <packed_root> [n_decode]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

from nunspark.engine import StreamingEngine
from nunspark.generate import _open_kv_store, _prefill
from nunspark.manifest import Manifest

PROMPT = [3, 7, 42, 1, 9, 2, 5, 11]
BUDGET = 10 * (1 << 30)         # the measured WIRED 30B optimum on 16 GiB
CLEAN_COMPRESS_GB = 30.0        # #15: clean wired runs compress ~15-20 GB; >30 = churn
REPS = 3                        # timed runs per W; median over them (robust to a stray)


# --- house-style inlined probe helpers (each probe keeps its own copy) ---
def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def _page_size() -> int:
    out = _sh(["sysctl", "-n", "hw.pagesize"]).strip()
    return int(out) if out.isdigit() else 4096


def _vm_compressed_pages() -> int:
    """Pages currently held in the macOS compressor (churn signal)."""
    for line in _sh(["vm_stat"]).splitlines():
        if "occupied by compressor" in line.lower():
            digits = "".join(c for c in line if c.isdigit())
            return int(digits) if digits else 0
    return 0


def _flush_page_cache() -> None:
    _sh(["sync"])
    _sh(["purge"])          # best-effort; no-op / needs privileges on some setups


# --- windowed _sync_layer monkeypatch ---------------------------------------
def _install_windowed_sync(engine, W: int):
    """Replace _sync_layer so it evals only every W-th layer call. W=1 reproduces
    today's per-layer eval exactly. The trailing partial window at token end is
    drained by the logits/sample eval."""
    state = {"i": 0}

    def windowed(h):
        state["i"] += 1
        if state["i"] % W == 0:
            mx.eval(h)
        return h

    engine._sync_layer = windowed
    engine._fully_resident = False   # force the streaming barrier path
    return state


def _run_once(root, manifest, W: int, n_decode: int):
    _flush_page_cache()
    engine = StreamingEngine(root, manifest, budget_bytes=BUDGET)
    _install_windowed_sync(engine, W)
    import tempfile
    tmp = tempfile.TemporaryDirectory(prefix="nunspark_window_kv_")
    kv = _open_kv_store(engine, tmp.name, 10**12, True, None)
    page = _page_size()
    try:
        logits = _prefill(engine, PROMPT, kv, 1024)
        tok = int(mx.argmax(logits, axis=-1).item())
        toks = []
        mx.synchronize()
        c0 = _vm_compressed_pages()
        t0 = time.monotonic()
        for _ in range(n_decode):
            logits = engine.forward(mx.array([[tok]]), kv=kv)[:, -1, :]
            tok = int(mx.argmax(logits, axis=-1).item())
            toks.append(tok)
        mx.synchronize()
        dt = time.monotonic() - t0
        c1 = _vm_compressed_pages()
    finally:
        kv.close()
        tmp.cleanup()
        engine.close()
    return {
        "W": W,
        "tok_per_s": round(n_decode / dt, 3),
        "ms_per_token": round(1e3 * dt / n_decode, 3),
        "compressed_gb_during_run": round(max(0, c1 - c0) * page / (1 << 30), 2),
        "tokens": toks,
    }


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> None:
    root = Path(sys.argv[1])
    n_decode = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    manifest = Manifest.load(root / "manifest.json")

    Ws = [1, 2, 3, 4]
    # Discard one warmup run FIRST so no TIMED run is the global cold start (the
    # OS page cache / allocator ramp otherwise lands entirely on run 1 and, under
    # a symmetric order, biases whichever W sits at the ends — it dragged W=1's
    # mean down in the first pass). purge() is best-effort and may no-op without
    # privileges, so steady-state warm-disk is the fair, representative regime.
    _run_once(root, manifest, 1, max(32, n_decode // 4))   # warmup, discarded

    # REPS runs per W in a drift-canceling order; median per W is robust to a stray.
    order = (Ws + Ws[::-1]) * ((REPS + 1) // 2)
    order = order[:REPS * len(Ws)]
    runs = [_run_once(root, manifest, W, n_decode) for W in order]

    ref = runs[0]["tokens"]
    streams_identical = all(r["tokens"] == ref for r in runs)

    # aggregate per W (MEDIAN over its reps); flag churned runs out of the band
    per_w = {}
    for W in Ws:
        rs = [r for r in runs if r["W"] == W]
        tps = _median([r["tok_per_s"] for r in rs])
        comp = max(r["compressed_gb_during_run"] for r in rs)
        per_w[W] = {"tok_per_s": round(tps, 3),
                    "reps": len(rs),
                    "max_compressed_gb": comp,
                    "in_clean_band": comp <= CLEAN_COMPRESS_GB}
    base = per_w[1]["tok_per_s"]
    for W in Ws:
        per_w[W]["pct_vs_W1"] = round(100 * (per_w[W]["tok_per_s"] - base) / base, 2)

    in_band = {W: d for W, d in per_w.items() if d["in_clean_band"] and W != 1}
    best_W = max(in_band, key=lambda W: in_band[W]["tok_per_s"]) if in_band else None
    best_pct = per_w[best_W]["pct_vs_W1"] if best_W else 0.0
    passed = best_W is not None and best_pct >= 10.0 and streams_identical

    out = {
        "probe": "windowed per-layer eval realizability (Plan 9 M0)",
        "model": str(root),
        "budget_gb": BUDGET / (1 << 30),
        "n_layers": manifest.num_layers,
        "n_decode": n_decode,
        "streams_identical": streams_identical,
        "per_W": {str(W): d for W, d in per_w.items()},
        "best_in_band_W": best_W,
        "best_pct_vs_W1": best_pct,
        "GATE_passed": passed,
        "verdict": (
            f"PASS: W={best_W} gives {best_pct:.1f}% over W=1 in the clean band "
            "-> proceed to Plan 9 M1"
            if passed else
            "FAIL: no in-band W clears +10% (or streams diverged) -> Plan 9 stops; "
            "the layer barrier is real but not streaming-realizable to the bar. "
            "Record dead in #16."),
        "raw_runs": [{k: v for k, v in r.items() if k != "tokens"} for r in runs],
    }
    print(json.dumps(out, indent=2))
    with open("scripts/results/windowed_eval_probe.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
