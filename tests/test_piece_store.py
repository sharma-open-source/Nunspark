import mlx.core as mx
from nunspark.manifest import Manifest
from nunspark.packer import pack
from nunspark.piece_store import PieceStore


def test_piece_store_loads_layer(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    store = PieceStore(out, manifest)
    w = store.load("layer_000")

    assert isinstance(w, dict)
    assert "self_attn.q_proj.weight" in w
    assert isinstance(w["self_attn.q_proj.weight"], mx.array)


def test_path_for_points_at_the_loaded_file(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    store = PieceStore(out, manifest)

    p = store.path_for("layer_000")
    assert p == out / manifest.piece_for("layer_000").file
    assert p.exists()
    # path_for points at exactly the file load() reads
    assert set(store.load("layer_000")) == set(mx.load(str(p)))
