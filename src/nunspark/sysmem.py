"""Unified-memory sizing helpers shared by the CLI and the web UI.

`auto_budget_bytes` derives a default resident-weight budget from total
unified RAM so a user who never passes `--budget` still gets a cache sized
for their machine instead of a fixed, tiny default. The formula --
`0.75 * (RAM - 8 GB)`, floored at 2 GB -- is calibrated against real runs:

- 16 GB  -> 6 GB   (measured optimum for Qwen3-30B-A3B on this repo:
                    greedy 4/5/6/8/10 GB sweep = 1.79 / 2.52 / 3.23 /
                    2.84 / 1.33 tok/s, post scatter fix)
- 64 GB  -> 42 GB  (community report: an M1 Max ran comfortably at 44-58 GB)
- 128 GB -> 90 GB  (community report: an M5 Max ran fine at 90 GB resident,
                     peak unified memory around 99 GB)

Subtracting a FIXED 8 GB before taking the fraction is what fits all three
calibration machines at once: the OS-plus-apps baseline is an absolute cost,
not proportional to RAM, so a 16 GB machine's optimum is ~37% of total while
a 128 GB machine's is ~70%. (The previous `0.75 * RAM - 4 GB` picked 8 GB on
16 GB machines -- measurably past the optimum once the scatter fix made
decode fast enough to feel the compressor. The cliff is asymmetric: on that
sweep, 2 GB under the optimum cost ~22% while 4 GB over it cost ~59%.)

A dynamic clamp against currently-AVAILABLE memory was tried (2026-07-17)
and reverted the same day: macOS's free-percentage estimate is too volatile
mid-session and starved the budget to the 2 GB floor on a machine that ran
perfectly well at 6 GB moments later. Total RAM is the stable, predictable
signal; users on loaded machines can always pass an explicit smaller
`--budget`.
"""

from __future__ import annotations

import os

_FRACTION = 0.75
_BASELINE_BYTES = 8 * 2**30  # fixed OS-plus-apps cost, subtracted before the fraction
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


def auto_budget_bytes(ram_bytes: int | None = None) -> int | None:
    """Derive a resident-weight budget from total unified RAM, or None when
    RAM is unknowable (non-macOS without sysconf support, sandboxed CI, etc.).
    `0.75 * (RAM - 8 GB)`, floored at 2 GB -- see the module docstring for
    the calibration data."""
    if ram_bytes is None:
        ram_bytes = unified_ram_bytes()
    if ram_bytes is None:
        return None

    return max(int(_FRACTION * (ram_bytes - _BASELINE_BYTES)), _FLOOR_BYTES)


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
