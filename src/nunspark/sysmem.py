"""Unified-memory sizing helpers shared by the CLI and the web UI.

`auto_budget_bytes` derives a default resident-weight budget from total
unified RAM so a user who never passes `--budget` still gets a cache sized
for their machine instead of a fixed, tiny default. The formula --
`0.75 * RAM - 4 GB`, floored at 2 GB -- is calibrated against real runs:

- 16 GB  -> 8 GB   (proven local setting running Qwen3-30B-A3B on this repo)
- 64 GB  -> 44 GB  (community report: an M1 Max ran comfortably at 48-58 GB)
- 128 GB -> 92 GB  (community report: an M5 Max ran fine at 90 GB resident,
                     peak unified memory around 99 GB)

The 4 GB subtracted off the top is headroom for the OS, KV cache,
activations, and everything else that shares unified memory with the
resident-weight cache.
"""

from __future__ import annotations

import os

_FRACTION = 0.75
_HEADROOM_BYTES = 4 * 2**30
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
    """Derive a resident-weight budget from unified RAM, or None when RAM
    is unknowable (non-macOS without sysconf support, sandboxed CI, etc.)."""
    if ram_bytes is None:
        ram_bytes = unified_ram_bytes()
    if ram_bytes is None:
        return None

    return max(int(_FRACTION * ram_bytes) - _HEADROOM_BYTES, _FLOOR_BYTES)


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
