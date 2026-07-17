from nunspark import sysmem
from nunspark.cli import _parse_size


def test_auto_budget_bytes_16gib():
    # 0.75 * (16 - 8) = 6 GiB -- the measured 30B optimum on a fresh 16 GB
    # machine (greedy 4/5/6/8/10 GB sweep = 1.79/2.52/3.23/2.84/1.33 tok/s).
    ram = 16 * 2**30
    assert sysmem.auto_budget_bytes(ram) == 6 * 2**30


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


def test_unified_ram_bytes_returns_int_or_none():
    result = sysmem.unified_ram_bytes()
    assert result is None or (isinstance(result, int) and result > 0)


def test_resolve_budget_auto(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: 64 * 2**30)
    # Pin availability high so the availability clamp (tested separately
    # below) stays inactive and this exercises the total-RAM ceiling.
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 60 * 2**30)
    assert sysmem.resolve_budget("auto") == 42 * 2**30
    # case-insensitive / whitespace-tolerant
    assert sysmem.resolve_budget("  Auto  ") == 42 * 2**30


def test_auto_budget_availability_clamp_inactive_when_fresh():
    # Plenty available: the total-RAM ceiling wins unchanged.
    ram = 16 * 2**30
    assert sysmem.auto_budget_bytes(ram, available_bytes=12 * 2**30) == 6 * 2**30


def test_auto_budget_availability_clamp_active_when_busy():
    # 5 GiB available on a 16 GiB machine: clamp to available - 1 GiB.
    ram = 16 * 2**30
    assert sysmem.auto_budget_bytes(ram, available_bytes=5 * 2**30) == 4 * 2**30


def test_auto_budget_availability_clamp_floors_at_2gib():
    ram = 16 * 2**30
    assert sysmem.auto_budget_bytes(ram, available_bytes=1 * 2**30) == 2 * 2**30


def test_auto_budget_explicit_ram_skips_availability_probe(monkeypatch):
    # Passing ram_bytes must stay deterministic: the real availability probe
    # is never consulted.
    def boom():
        raise AssertionError("availability probed despite explicit ram_bytes")
    monkeypatch.setattr(sysmem, "available_ram_bytes", boom)
    assert sysmem.auto_budget_bytes(16 * 2**30) == 6 * 2**30


def test_available_ram_bytes_returns_int_or_none():
    result = sysmem.available_ram_bytes()
    assert result is None or (isinstance(result, int) and result > 0)


def test_resolve_budget_passthrough():
    assert sysmem.resolve_budget("8GB") == _parse_size("8GB")
    assert sysmem.resolve_budget("512MB") == _parse_size("512MB")


def test_resolve_budget_auto_falls_back_when_ram_unknowable(monkeypatch):
    monkeypatch.setattr(sysmem, "unified_ram_bytes", lambda: None)
    assert sysmem.resolve_budget("auto") == _parse_size("4GB")
    assert sysmem.resolve_budget("auto", fallback="2GB") == _parse_size("2GB")
