"""Unified-memory sizing helpers shared by the CLI and the web UI.

`auto_budget_bytes` derives a default resident-weight budget from unified
RAM so a user who never passes `--budget` still gets a cache sized for
their machine instead of a fixed, tiny default. The formula is the total-RAM
ceiling `0.75 * (RAM - 8 GB)`, floored at 2 GB, CLAMPED to
`available RAM - 1 GB` when the machine is already using memory:

- 16 GB fresh -> 6 GB  (measured optimum for Qwen3-30B-A3B on this repo:
                        greedy 4/5/6/8/10 GB sweep = 1.79 / 2.52 / 3.23 /
                        2.84 / 1.33 tok/s, post scatter fix)
- 16 GB busy  -> less  (the clamp; see below)
- 64 GB  -> 42 GB  (community report: an M1 Max ran comfortably at 44-58 GB)
- 128 GB -> 90 GB  (community report: an M5 Max ran fine at 90 GB resident,
                     peak unified memory around 99 GB)

Subtracting a FIXED 8 GB before taking the fraction is what fits all three
calibration machines at once: the OS-plus-apps baseline is an absolute cost,
not proportional to RAM, so a 16 GB machine's optimum is ~37% of total while
a 128 GB machine's is ~70%. (The previous `0.75 * RAM - 4 GB` picked 8 GB on
16 GB machines — measurably past the optimum once the scatter fix made
decode fast enough to feel the compressor.)

Why the clamp exists (2026-07-17, scripts/results/decode_time_attribution
.json + a live 6/8/10 GB sweep): the memory cliff is ASYMMETRIC. On a 16 GB
machine running the 30B, 2 GB under the optimum cost ~13% (2.52 vs 2.84
tok/s) while 2 GB over it cost ~55% (10 GB budget -> 1.33 tok/s: the
resident cache fights the macOS compressor and every miss slows ~6x). The
cliff's location depends on what else is using RAM at launch, so total RAM
alone can't see it -- the clamp derives it from memory actually available
(free + reclaimable) when the engine starts. On a freshly booted machine
the clamp is inactive and the total-RAM ceiling wins unchanged.
"""

from __future__ import annotations

import os

_FRACTION = 0.75
_BASELINE_BYTES = 8 * 2**30  # fixed OS-plus-apps cost, subtracted before the fraction
_FLOOR_BYTES = 2 * 2**30
# Headroom subtracted from AVAILABLE memory for the clamp: the engine peaks
# at roughly budget + ~0.9 GB beyond the weight cache (activations, KV, slot;
# measured post scatter fix). The kernel's free-percentage estimate is itself
# conservative (the compressor can still grow), so 1 GB suffices. Calibrated:
# on the 16 GB dev machine mid-session (58% free ~= 9.3 GB) the clamp yields
# 8.3 GB, comfortably above the 6 GB ceiling -- the ceiling (the measured
# optimum) wins in any ordinary session; the clamp exists for heavy load.
_AVAILABLE_HEADROOM_BYTES = 1 * 2**30


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


def available_ram_bytes() -> int | None:
    """Memory actually available right now (free + reclaimable), in bytes,
    or None when it can't be determined. Never raises.

    macOS: `memory_pressure -Q`'s "memory free percentage" times total RAM.
    That is the kernel's own effective-availability estimate and correctly
    counts reclaimable file cache and compressor headroom -- a raw
    free+inactive vm_stat sum was measured to undercount by ~2x on a machine
    whose RAM was full of a previous run's page cache. vm_stat (free +
    inactive + purgeable + speculative) stays as the fallback when the tool
    is missing. Linux: /proc/meminfo MemAvailable (also a kernel estimate)."""
    try:  # macOS, primary: kernel's own free-percentage estimate
        import re
        import subprocess

        out = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=5,
        ).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        total = unified_ram_bytes()
        if m and total:
            return int(total * int(m.group(1)) / 100)
    except Exception:
        pass

    try:  # macOS, fallback: crude reclaimable-page sum
        import re
        import subprocess

        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5,
        ).stdout
        m = re.search(r"page size of (\d+) bytes", out)
        page = int(m.group(1)) if m else 16384
        pages = 0
        for name in ("Pages free", "Pages inactive", "Pages purgeable",
                     "Pages speculative"):
            m = re.search(rf"{re.escape(name)}:\s+(\d+)\.", out)
            if m:
                pages += int(m.group(1))
        if pages > 0:
            return pages * page
    except Exception:
        pass

    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass

    return None


def auto_budget_bytes(
    ram_bytes: int | None = None, available_bytes: int | None = None
) -> int | None:
    """Derive a resident-weight budget from unified RAM, or None when RAM
    is unknowable (non-macOS without sysconf support, sandboxed CI, etc.).

    The total-RAM ceiling (`0.75 * (RAM - 8 GB)`) is clamped to
    `available - 1 GB` so an already-loaded machine gets a smaller cache
    instead of one that fights the macOS compressor (see module docstring).
    Both floors at 2 GB. Availability is only probed on the real auto path
    (both arguments None): callers/tests passing an explicit `ram_bytes`
    stay deterministic unless they also pass `available_bytes`."""
    probe_available = ram_bytes is None and available_bytes is None
    if ram_bytes is None:
        ram_bytes = unified_ram_bytes()
    if ram_bytes is None:
        return None

    budget = max(int(_FRACTION * (ram_bytes - _BASELINE_BYTES)), _FLOOR_BYTES)
    if probe_available:
        available_bytes = available_ram_bytes()
    if available_bytes is not None:
        clamp = available_bytes - _AVAILABLE_HEADROOM_BYTES
        budget = max(min(budget, clamp), _FLOOR_BYTES)
    return budget


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
