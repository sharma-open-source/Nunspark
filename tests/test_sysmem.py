from nunspark import sysmem
from nunspark.cli import _parse_size


def test_auto_budget_bytes_16gib():
    ram = 16 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 8 * 2**30


def test_auto_budget_bytes_128gib():
    ram = 128 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 92 * 2**30


def test_auto_budget_bytes_floor():
    # 4 GiB RAM: 0.75*4 - 4 = -1 GiB, must clamp to the 2 GiB floor.
    ram = 4 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 2 * 2**30


def test_auto_budget_bytes_none_ram_returns_none(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: None)
    assert sysmem.auto_budget_bytes() is None


def test_unified_ram_bytes_returns_int_or_none():
    result = sysmem.unified_ram_bytes()
    assert result is None or (isinstance(result, int) and result > 0)


def test_resolve_budget_auto(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: 64 * 2**30)
    assert sysmem.resolve_budget("auto") == 44 * 2**30
    # case-insensitive / whitespace-tolerant
    assert sysmem.resolve_budget("  Auto  ") == 44 * 2**30


def test_resolve_budget_passthrough():
    assert sysmem.resolve_budget("8GB") == _parse_size("8GB")
    assert sysmem.resolve_budget("512MB") == _parse_size("512MB")


def test_resolve_budget_auto_falls_back_when_ram_unknowable(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: None)
    assert sysmem.resolve_budget("auto") == _parse_size("4GB")
    assert sysmem.resolve_budget("auto", fallback="2GB") == _parse_size("2GB")
