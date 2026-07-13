import json
from pathlib import Path

from nunspark.webapp.models import ModelRegistry


def _make_packed(dir_: Path) -> None:
    dir_.mkdir(parents=True)
    (dir_ / "manifest.json").write_text(json.dumps({"num_layers": 2}))


def test_scans_packed_folders(tmp_path):
    packs = tmp_path / "packs"
    _make_packed(packs / "model-a")
    _make_packed(packs / "model-b")
    (packs / "not-a-model").mkdir()  # no manifest.json -> ignored

    reg = ModelRegistry(packed_root=packs, hf_cache=tmp_path / "nohf")
    entries = reg.list_models()
    packed = [e for e in entries if e["kind"] == "packed"]
    names = sorted(e["name"] for e in packed)
    assert names == ["model-a", "model-b"]
    assert all(Path(e["path"]).is_absolute() for e in packed)


def test_scans_hf_cache_as_unpacked(tmp_path):
    hf = tmp_path / "hub"
    snap = hf / "models--mlx-community--Tiny" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")

    reg = ModelRegistry(packed_root=tmp_path / "nopacks", hf_cache=hf)
    entries = reg.list_models()
    unpacked = [e for e in entries if e["kind"] == "unpacked"]
    assert len(unpacked) == 1
    assert unpacked[0]["name"] == "mlx-community/Tiny"
    assert Path(unpacked[0]["path"]) == snap


def test_is_packed(tmp_path):
    packs = tmp_path / "packs"
    _make_packed(packs / "m")
    reg = ModelRegistry(packed_root=packs, hf_cache=tmp_path / "nohf")
    assert reg.is_packed(str((packs / "m").resolve())) is True
    assert reg.is_packed(str(tmp_path / "missing")) is False
