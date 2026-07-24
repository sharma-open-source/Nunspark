"""Unified-memory sizing helpers shared by the CLI and the web UI.

`auto_budget_bytes` derives a default resident-weight budget from total
unified RAM so a user who never passes `--budget` still gets a cache sized
for their machine instead of a fixed, tiny default. The formula --
`0.75 * RAM - 2 GB`, floored at 2 GB -- is the WIRED-regime formula
(backlog #14, 2026-07-24): the engine now wires the MLX buffer pool to the
device's max_recommended_working_set_size (~= 0.75 * RAM on Apple Silicon),
and the budget must keep the peak working set (budget + ~1 GB engine
overhead) UNDER that wire -- past it, the tail gets compressed again
(measured: an 11 GB budget's 11.99 GB peak crossed the 11.84 GB wire and
dropped 7.8 -> 6.4 tok/s with 300k swapouts). Calibration:

- 16 GB  -> 10 GB  (measured wired optimum for Qwen3-30B-A3B: wired sweep
                    6/8/9/10/11 GB = 4.59 / 6.40 / 6.69 / 7.79 / 6.36 tok/s
                    flushed probes, scripts/results/wired_limit_ab*.json;
                    live-verified 6.92 tok/s over 800 tokens)
- 64 GB  -> 46 GB  (community M1 Max ran comfortably at 44-58 GB, unwired)
- 128 GB -> 94 GB  (community M5 Max ran fine at 90 GB resident, unwired;
                    wired re-verification welcome)

The PRE-WIRING formula `0.75 * (RAM - 8 GB)` (16 -> 6 GB) is obsolete: its
16 GB "budget cliff" (6 GB optimum, 10 GB collapsing to 1.33 tok/s) was the
macOS compressor stealing the unwired cache, not a real cache-size optimum
-- wired, throughput rises monotonically with budget up to the wire.

Deliberately a pure function of TOTAL RAM: a dynamic clamp against
currently-AVAILABLE memory was tried (2026-07-17) and reverted the same day
-- macOS's free-percentage estimate is too volatile mid-session and starved
the budget to the 2 GB floor on a machine that ran fine at 6 GB moments
later. Total RAM is the stable, predictable signal; users on loaded machines
can always pass an explicit smaller `--budget`.
"""

from __future__ import annotations

import os

_FRACTION = 0.75          # tracks max_recommended_working_set_size ~= 0.75 * RAM
_WIRED_HEADROOM_BYTES = 2 * 2**30  # keeps budget + ~1 GB engine overhead under the wire
_FLOOR_BYTES = 2 * 2**30


def unified_ram_bytes() -> int | None:
    """Total unified/physical RAM in bytes, or None when it can't be
    determined. Never raises."""
    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        size = info.get("memory_size")
        if isinstance(size, int) and size > 0:
            return size
    except Exception:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total = page_size * phys_pages
        if isinstance(total, int) and total > 0:
            return total
    except Exception:
        pass

    return None


def wire_memory_limit() -> int | None:
    """Wire the MLX buffer pool: raise the Metal wired-memory limit to the
    device's ``max_recommended_working_set_size``, mirroring what mlx-lm's own
    generate loop does before decoding large models.

    Without this, every cached weight lives in pageable anonymous memory --
    exactly what the macOS compressor takes under pressure. Measured 2026-07-24
    (scripts/results/wired_limit_ab.json, backlog #14): the 16 GB "budget
    cliff" was entirely a wiring artifact -- an unwired 10 GB budget kept only
    ~5 GB actually resident (2.7 tok/s, ~9 GB of swap traffic per 150 tokens)
    vs 8.0 tok/s wired, byte-identical tokens, at every budget wired >= unwired.

    Returns the applied limit in bytes, or None when the running mlx has no
    wired-limit API or no device info (non-Metal builds, CI). Never raises;
    calling repeatedly is harmless (it sets the same cap).
    """
    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = info.get("max_recommended_working_set_size")
        if not isinstance(limit, int) or limit <= 0:
            return None
        setter = getattr(mx, "set_wired_limit", None) or getattr(
            getattr(mx, "metal", None), "set_wired_limit", None)
        if setter is None:
            return None
        setter(limit)
        return limit
    except Exception:
        return None


def auto_budget_bytes(ram_bytes: int | None = None) -> int | None:
    """Derive a resident-weight budget from total unified RAM, or None when
    RAM is unknowable (non-macOS without sysconf support, sandboxed CI, etc.).
    `0.75 * RAM - 2 GB`, floored at 2 GB -- sized to keep the peak working
    set just under the wired-memory limit; see the module docstring for the
    calibration data."""
    if ram_bytes is None:
        ram_bytes = unified_ram_bytes()
    if ram_bytes is None:
        return None

    return max(int(_FRACTION * ram_bytes) - _WIRED_HEADROOM_BYTES, _FLOOR_BYTES)


def resolve_budget(spec: str, fallback: str = "4GB") -> int:
    """Resolve a `--budget`-style string to a byte count.

    `"auto"` (case-insensitive) derives the budget from unified RAM via
    `auto_budget_bytes()`; when RAM can't be determined, falls back to
    `fallback` parsed the same way a concrete value would be. Any other
    string is parsed as a plain size (e.g. "512MB", "8GB").
    """
    from .cli import _parse_size

    if spec.strip().lower() == "auto":
        budget = auto_budget_bytes()
        if budget is None:
            return _parse_size(fallback)
        return budget
    return _parse_size(spec)
