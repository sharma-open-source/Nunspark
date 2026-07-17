from nunspark import sysmem
from nunspark.cli import _parse_size


def test_auto_budget_bytes_16gib():
    # 0.75 * (16 - 8) = 6 GiB -- the measured 30B optimum on a fresh 16 GB
    # machine (greedy 4/5/6/8/10 GB sweep = 1.79/2.52/3.23/2.84/1.33 tok/s).
    ram = 16 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 6 * 2**30


def test_auto_budget_bytes_64gib():
    # 0.75 * (64 - 8) = 42 GiB (community M1 Max ran comfortably at 44-58 GB).
    ram = 64 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 42 * 2**30


def test_auto_budget_bytes_128gib():
    # 0.75 * (128 - 8) = 90 GiB (community M5 Max ran fine at 90 GB resident).
    ram = 128 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 90 * 2**30


def test_auto_budget_bytes_floor():
    # 4 GiB RAM: 0.75*(4-8) is negative, must clamp to the 2 GiB floor.
    ram = 4 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 2 * 2**30


def test_auto_budget_bytes_none_ram_returns_none(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: None)
    assert sysmem.auto_budget_bytes() is None


def test_auto_budget_depends_only_on_total_ram(monkeypatch):
    # The 2026-07-17 availability clamp was reverted: auto must be a pure
    # function of total RAM, never probing live memory state (a mid-session
    # probe once starved the budget to the 2 GiB floor on a machine that ran
    # fine at 6 GiB).
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: 16 * 2**30)
    assert sysmem.auto_budget_bytes() == 6 * 2**30


def test_unified_ram_bytes_returns_int_or_none():
    result = sysmem.unified_ram_bytes()
    assert result is None or (isinstance(result, int) and result > 0)


def test_resolve_budget_auto(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: 64 * 2**30)
    assert sysmem.resolve_budget("auto") == 42 * 2**30
    # case-insensitive / whitespace-tolerant
    assert sysmem.resolve_budget("  Auto  ") == 42 * 2**30


def test_resolve_budget_passthrough():
    assert sysmem.resolve_budget("8GB") == _parse_size("8GB")
    assert sysmem.resolve_budget("512MB") == _parse_size("512MB")


def test_resolve_budget_auto_falls_back_when_ram_unknowable(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: None)
    assert sysmem.resolve_budget("auto") == _parse_size("4GB")
    assert sysmem.resolve_budget("auto", fallback="2GB") == _parse_size("2GB")
