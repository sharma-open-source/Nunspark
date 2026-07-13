import json

import mlx.core as mx
from nunspark.manifest import Manifest
from nunspark.packer import pack


def test_pack_creates_pieces_and_manifest(tiny_model_dir, tmp_path):
    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)

    manifest = Manifest.load(out / "manifest.json")
    assert manifest.num_layers == 4
    # one embed piece + 4 layer pieces + one norm_head piece
    assert {p.piece_id for p in manifest.pieces} == {
        "embed", "layer_000", "layer_001", "layer_002", "layer_003", "norm_head",
    }

    # layer piece keys are relative to the TransformerBlock module
    layer0 = manifest.piece_for("layer_000")
    assert "self_attn.q_proj.weight" in layer0.keys
    assert (out / layer0.file).exists()

    # piece files actually contain the arrays
    w = mx.load(str(out / layer0.file))
    assert "self_attn.q_proj.weight" in w


def test_pack_from_in_memory_weights_matches_file_path(tiny_model_dir, tmp_path):
    # File-based pack (the default path).
    from_file = pack(tiny_model_dir, tmp_path / "from-file")

    # In-memory pack: hand pack() the weights + config directly, model_dir=None.
    weights = mx.load(str(tiny_model_dir / "model.safetensors"))
    config = json.loads((tiny_model_dir / "config.json").read_text())
    from_mem = pack(None, tmp_path / "from-mem", weights=weights, config=config)

    # Identical manifests (same pieces, same keys per piece).
    assert from_mem.num_layers == from_file.num_layers
    assert {(p.piece_id, tuple(p.keys)) for p in from_mem.pieces} == \
        {(p.piece_id, tuple(p.keys)) for p in from_file.pieces}

    # Identical tensor bytes per piece.
    for piece in from_file.pieces:
        a = mx.load(str(tmp_path / "from-file" / piece.file))
        b = mx.load(str(tmp_path / "from-mem" / piece.file))
        assert a.keys() == b.keys()
        for k in a:
            assert mx.array_equal(a[k], b[k])


def test_pack_quantized_captures_scales_and_biases(tiny_quant_model_dir, tmp_path):
    out = tmp_path / "tiny-q4.nunspark"
    pack(tiny_quant_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    # embed piece must carry the quantized triplet, not just the packed weight
    embed = manifest.piece_for("embed")
    assert set(embed.keys) == {
        "embed_tokens.weight", "embed_tokens.scales", "embed_tokens.biases",
    }
    embed_w = mx.load(str(out / embed.file))
    assert "embed_tokens.scales" in embed_w and "embed_tokens.biases" in embed_w

    # per-layer pieces already capture the projection triplets
    layer0 = manifest.piece_for("layer_000")
    assert "self_attn.q_proj.scales" in layer0.keys
    assert "self_attn.q_proj.biases" in layer0.keys

    # the quantization config rides along in the manifest for the engine to read
    assert manifest.config["quantization"] == {"group_size": 64, "bits": 4}


def test_pack_quantized_untied_captures_lm_head_triplet(tiny_quant_untied_model_dir, tmp_path):
    out = tmp_path / "tiny-q4-untied.nunspark"
    pack(tiny_quant_untied_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")
    assert manifest.tie_word_embeddings is False
    norm_head = manifest.piece_for("norm_head")
    assert {"lm_head.weight", "lm_head.scales", "lm_head.biases"} <= set(norm_head.keys)
    assert "norm.weight" in norm_head.keys


def test_pack_moe_layer_splits_into_core_and_experts(tiny_qwen3_moe_model_dir, tmp_path):
    out = tmp_path / "moe.nunspark"
    pack(tiny_qwen3_moe_model_dir, out)
    manifest = Manifest.load(out / "manifest.json")

    ids = {p.piece_id for p in manifest.pieces}
    # 4 MoE layers (mlp_only_layers=[]), 8 experts each -> core + 8 expert pieces/layer
    for layer in range(4):
        assert Manifest.layer_core_piece_id(layer) in ids
        for e in range(8):
            assert Manifest.layer_expert_piece_id(layer, e) in ids
        # the old whole-layer piece must NOT exist for a split layer
        assert Manifest.layer_piece_id(layer) not in ids

    # core carries attention + router gate, NOT the stacked experts
    core = manifest.piece_for(Manifest.layer_core_piece_id(0))
    assert "self_attn.q_proj.weight" in core.keys
    assert "mlp.gate.weight" in core.keys
    assert not any(k.startswith("mlp.switch_mlp.") for k in core.keys)

    # an expert piece carries only its own switch_mlp rows (no expert axis in the key)
    e3 = manifest.piece_for(Manifest.layer_expert_piece_id(0, 3))
    assert "mlp.switch_mlp.gate_proj.weight" in e3.keys
    assert "mlp.switch_mlp.up_proj.weight" in e3.keys
    assert "mlp.switch_mlp.down_proj.weight" in e3.keys


def test_pack_moe_expert_piece_equals_stacked_row(tiny_qwen3_moe_model_dir, tmp_path):
    out = tmp_path / "moe.nunspark"
    pack(tiny_qwen3_moe_model_dir, out)

    src = mx.load(str(tiny_qwen3_moe_model_dir / "model.safetensors"))
    # check every projection, for two different expert rows, equals its stacked row
    for e in (3, 0):
        piece = mx.load(str(out / f"{Manifest.layer_expert_piece_id(0, e)}.safetensors"))
        assert set(piece.keys()) == {
            "mlp.switch_mlp.gate_proj.weight",
            "mlp.switch_mlp.up_proj.weight",
            "mlp.switch_mlp.down_proj.weight",
        }
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = src[f"model.layers.0.mlp.switch_mlp.{proj}.weight"]   # [num_experts, ...]
            assert mx.array_equal(piece[f"mlp.switch_mlp.{proj}.weight"], stacked[e])


def test_pack_mixed_moe_keeps_dense_layers_whole(tmp_path):
    # mlp_only_layers=[0,2] -> layers 0,2 dense (whole-layer), layers 1,3 MoE (split)
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_moe import Model, ModelArgs

    config = {
        "model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 4,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "num_experts": 8, "num_experts_per_tok": 2,
        "decoder_sparse_step": 1, "mlp_only_layers": [0, 2], "moe_intermediate_size": 64,
        "norm_topk_prob": True, "rms_norm_eps": 1e-5, "vocab_size": 320,
        "max_position_embeddings": 2048, "rope_theta": 10000.0, "tie_word_embeddings": True,
    }
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    src = tmp_path / "mixed"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(src / "model.safetensors"), dict(tree_flatten(model.parameters())))

    out = tmp_path / "mixed.nunspark"
    pack(src, out)
    manifest = Manifest.load(out / "manifest.json")
    ids = {p.piece_id for p in manifest.pieces}
    assert Manifest.layer_piece_id(0) in ids        # dense -> whole-layer
    assert Manifest.layer_piece_id(2) in ids        # dense -> whole-layer
    assert Manifest.layer_core_piece_id(1) in ids   # moe -> split
    assert Manifest.layer_core_piece_id(3) in ids   # moe -> split
    assert Manifest.layer_core_piece_id(0) not in ids


def test_pack_copies_tokenizer_files(tiny_model_dir, tmp_path):
    (tiny_model_dir / "tokenizer.json").write_text('{"fake": "tokenizer"}')
    (tiny_model_dir / "tokenizer_config.json").write_text('{"fake": "config"}')
    (tiny_model_dir / "special_tokens_map.json").write_text('{"fake": "map"}')
    (tiny_model_dir / "README.md").write_text("not a tokenizer file")

    out = tmp_path / "tiny.nunspark"
    pack(tiny_model_dir, out)

    assert (out / "tokenizer.json").read_text() == '{"fake": "tokenizer"}'
    assert (out / "tokenizer_config.json").read_text() == '{"fake": "config"}'
    assert (out / "special_tokens_map.json").read_text() == '{"fake": "map"}'
    assert not (out / "README.md").exists()


def test_pack_skips_missing_tokenizer_files(tiny_model_dir, tmp_path):
    # tiny_model_dir ships no tokenizer files at all -- pack must not error
    out = tmp_path / "tiny.nunspark"
    manifest = pack(tiny_model_dir, out)
    assert manifest.num_layers == 4
    assert not (out / "tokenizer.json").exists()


def test_pack_from_in_memory_weights_skips_tokenizer_copy(tiny_model_dir, tmp_path):
    weights = mx.load(str(tiny_model_dir / "model.safetensors"))
    config = json.loads((tiny_model_dir / "config.json").read_text())

    out = tmp_path / "from-mem"
    manifest = pack(None, out, weights=weights, config=config)   # model_dir=None: nothing to copy from
    assert manifest.num_layers == 4
    assert not (out / "tokenizer.json").exists()

