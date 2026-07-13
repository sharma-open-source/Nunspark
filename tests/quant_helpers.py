"""Shared helpers for the quantized-model tests."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import Model, ModelArgs


def load_quantized_model(model_dir: str | Path) -> tuple[Model, dict]:
    """Full-load a packed quantized mlx-lm model dir, the way mlx-lm would.

    Returns the live (model, config) so each test can run whatever forward /
    decode it needs against an independent, non-streaming reference.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    model = Model(args)
    q = config["quantization"]
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model, config
