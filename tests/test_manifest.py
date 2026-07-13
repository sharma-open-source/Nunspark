import json
from nunspark.manifest import Manifest, Piece


def test_manifest_roundtrip(tmp_path):
    m = Manifest(
        model_type="llama",
        config={"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 320},
        tie_word_embeddings=True,
        num_layers=2,
        pieces=[
            Piece(piece_id="embed", file="embed.safetensors", keys=["embed_tokens.weight"]),
            Piece(piece_id="layer_000", file="layer_000.safetensors", keys=["self_attn.q_proj.weight"]),
            Piece(piece_id="norm_head", file="norm_head.safetensors", keys=["norm.weight"]),
        ],
    )
    path = tmp_path / "manifest.json"
    m.save(path)

    raw = json.loads(path.read_text())
    assert raw["num_layers"] == 2
    assert raw["pieces"][0]["piece_id"] == "embed"

    loaded = Manifest.load(path)
    assert loaded == m
    assert loaded.layer_piece_id(1) == "layer_001"


def test_layer_core_and_expert_piece_ids():
    assert Manifest.layer_core_piece_id(7) == "layer_007_core"
    assert Manifest.layer_expert_piece_id(7, 3) == "layer_007_expert_3"
    assert Manifest.layer_expert_piece_id(12, 0) == "layer_012_expert_0"


def test_has_piece():
    m = Manifest(
        model_type="qwen3_moe",
        config={},
        tie_word_embeddings=True,
        num_layers=1,
        pieces=[Piece(piece_id="layer_000_core", file="f.safetensors", keys=[])],
    )
    assert m.has_piece("layer_000_core") is True
    assert m.has_piece("layer_000_expert_0") is False
