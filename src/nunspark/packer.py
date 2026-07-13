from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .architectures import get_architecture, supported_model_types
from .manifest import Manifest, Piece


def _resolve_config(model_arg: str | Path) -> dict:
    """Return the model's config.json as a dict (local dir or HF snapshot)."""
    p = Path(model_arg)
    if p.exists():
        return json.loads((p / "config.json").read_text())
    from huggingface_hub import snapshot_download
    local = snapshot_download(str(model_arg), allow_patterns=["config.json"])
    return json.loads((Path(local) / "config.json").read_text())

# mlx-lm Llama parameter prefixes (verified against the actual model).
_EMBED_PREFIX = "model.embed_tokens."
_NORM_PREFIX = "model.norm."
_HEAD_PREFIX = "lm_head."
_LAYER_PREFIX = "model.layers."

# Known HF tokenizer artifact filenames -- copied verbatim into the packed
# output (when present) so a packed directory is self-contained and loadable
# by mlx_lm.tokenizer_utils.load(). Different tokenizers ship different
# subsets (sentencepiece carries tokenizer.model, BPE carries
# vocab.json/merges.txt, fast tokenizers carry tokenizer.json, newer releases
# split out chat_template.jinja) -- copying is purely additive.
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
]


def _copy_tokenizer_files(model_dir: Path | None, out_dir: Path) -> None:
    """Copy whichever known tokenizer files exist in `model_dir` into `out_dir`.

    `model_dir` is None for the in-memory pack path (caller already holds
    weights/config and has no source directory to copy from)."""
    if model_dir is None:
        return
    for name in _TOKENIZER_FILES:
        src = model_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, out_dir / name)


def _strip_layer_prefix(key: str) -> tuple[int, str]:
    # "model.layers.3.self_attn.q_proj.weight" -> (3, "self_attn.q_proj.weight")
    rest = key[len(_LAYER_PREFIX):]
    idx_str, sub = rest.split(".", 1)
    return int(idx_str), sub


def _collect(weights: dict, prefix: str, strip: str) -> dict:
    """Grab every weight whose key starts with `prefix`, re-keyed without `strip`.

    Captures the quantized triplet (.weight/.scales/.biases) as well as a plain
    .weight, so the same code packs both fp16 and 4-bit models.
    """
    return {k[len(strip):]: v for k, v in weights.items() if k.startswith(prefix)}


def _pack_moe_layer(out_dir: Path, layer: int, layer_weights: dict, expert_prefix: str,
                    num_experts: int, pieces: list[Piece]) -> None:
    """Split a MoE layer into a core piece (attn + norms + router) plus one piece
    per expert (row e of each stacked expert-module tensor — incl. any per-expert
    bias — stored WITHOUT the leading expert axis). `expert_prefix` is the
    architecture's expert-module key prefix, e.g. "mlp.switch_mlp." (qwen3_moe)
    or "mlp.experts." (gpt_oss)."""
    core = {sub: arr for sub, arr in layer_weights.items()
            if not sub.startswith(expert_prefix)}
    core_pid = Manifest.layer_core_piece_id(layer)
    core_fname = f"{core_pid}.safetensors"
    mx.save_safetensors(str(out_dir / core_fname), core)
    pieces.append(Piece(core_pid, core_fname, sorted(core)))

    switch = {sub: arr for sub, arr in layer_weights.items()
              if sub.startswith(expert_prefix)}
    for e in range(num_experts):
        expert_weights = {sub: arr[e] for sub, arr in switch.items()}
        epid = Manifest.layer_expert_piece_id(layer, e)
        efname = f"{epid}.safetensors"
        mx.save_safetensors(str(out_dir / efname), expert_weights)
        pieces.append(Piece(epid, efname, sorted(expert_weights)))


def pack(
    model_dir: str | Path | None,
    out_dir: str | Path,
    *,
    weights: dict | None = None,
    config: dict | None = None,
) -> Manifest:
    """Convert an mlx-lm model into NunSpark per-piece files + manifest.

    By default loads the model via ``mlx_lm.load()`` which handles both local
    paths and Hugging Face repo IDs (e.g. 'mlx-community/SmolLM2-360M-Instruct-bf16').
    The model is loaded into memory, then packed without a safetensors round-trip.

    Callers that already hold the model in memory (e.g. after ``mlx_lm.load``)
    can instead pass ``weights`` (a flat ``{key: mx.array}`` dict, as produced by
    ``tree_flatten(model.parameters())``) and ``config``, skipping the
    safetensors round-trip entirely; ``model_dir`` may then be ``None``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model directory for tokenizer copying
    resolved_model_dir = None
    if model_dir is not None:
        resolved_model_dir = Path(model_dir)
        # If it's not a local path, resolve it via HF snapshot_download
        if not resolved_model_dir.exists():
            from huggingface_hub import snapshot_download
            resolved_model_dir = Path(snapshot_download(str(model_dir)))

    _copy_tokenizer_files(resolved_model_dir, out_dir)

    if config is None:
        if model_dir is None:
            raise ValueError("config must be provided when model_dir is None")

        model_path = Path(model_dir)
        is_local = model_path.exists()

        if is_local:
            # Local path: try direct safetensors loading first (faster, no tokenizer needed)
            config = _resolve_config(model_dir)
            try:
                weights = mx.load(str(model_path / "model.safetensors"))
            except Exception:
                # Fall back to mlx_lm.load() for sharded or complex formats
                print(f"Loading {model_dir} ...", file=sys.stderr, flush=True)
                from mlx_lm import load as load_mlxlm
                model, _tokenizer = load_mlxlm(model_dir)
                print(f"  packing {len(dict(tree_flatten(model.parameters())))} in-memory tensors",
                      file=sys.stderr, flush=True)
                weights = dict(tree_flatten(model.parameters()))
        else:
            # HF repo: use mlx_lm.load() which handles download + loading
            config = _resolve_config(model_dir)
            print(f"Loading {model_dir} ...", file=sys.stderr, flush=True)
            from mlx_lm import load as load_mlxlm
            model, _tokenizer = load_mlxlm(model_dir)
            print(f"  packing {len(dict(tree_flatten(model.parameters())))} in-memory tensors",
                  file=sys.stderr, flush=True)
            weights = dict(tree_flatten(model.parameters()))

    # Handle gemma4's language_model.model.* prefix (multimodal wrapper)
    # Normalize weights: language_model.model.* -> model.*
    if any(k.startswith("language_model.model.") for k in weights.keys()):
        print("  normalizing gemma4 weight keys (language_model.model.* -> model.*)",
              file=sys.stderr, flush=True)
        normalized = {}
        for k, v in weights.items():
            if k.startswith("language_model.model."):
                normalized[k.replace("language_model.model.", "model.")] = v
            elif k.startswith("language_model.lm_head."):
                normalized[k.replace("language_model.lm_head.", "lm_head.")] = v
            else:
                normalized[k] = v
        weights = normalized

    # Handle multimodal gemma3/gemma4 configs where transformer config is nested under text_config
    if config.get("model_type") in ("gemma3", "gemma4") and "text_config" in config:
        # For packer consumption: extract fields from text_config for num_layers, etc.
        text_config = config["text_config"]
        # Merge text_config fields, overriding top-level None values
        config["_text_config_merged"] = True
        for key, value in text_config.items():
            config[key] = value

    # Special handling for gemma4_assistant (MTP drafter)
    if config.get("model_type") == "gemma4_assistant" and "text_config" in config:
        # gemma4_assistant also has nested text_config
        text_config = config["text_config"]
        config["_text_config_merged"] = True
        for key, value in text_config.items():
            config[key] = value

        # Load the model to call sanitize() for _token_ordering buffer installation
        from . import gemma4_assistant
        assistant = gemma4_assistant.Model(gemma4_assistant.ModelArgs.from_dict(config))
        weights = assistant.sanitize(weights)

    num_layers = config["num_hidden_layers"]
    tied = config.get("tie_word_embeddings", False)

    # Detect MoE layers so their experts pack as individually streamable pieces.
    # Only architectures whose ArchSpec opts into selective expert streaming
    # (selective_moe=True — qwen3_moe, gpt_oss) split; everything else — including
    # archs that carry a layer_key_fn purely for structural variants (e.g. gemma3
    # global/sliding rope) and unregistered types — packs whole-layer as before.
    layer_key_fn = None
    arch_args = None
    expert_prefix = None
    num_experts = None
    if config["model_type"] in supported_model_types():
        spec = get_architecture(config["model_type"])
        if spec.selective_moe:
            layer_key_fn = spec.layer_key_fn
            arch_args = spec.args_cls.from_dict(config)
            expert_prefix = f"mlp.{spec.expert_attr}."
            num_experts = spec.num_experts(arch_args) if spec.num_experts else arch_args.num_experts

    pieces: list[Piece] = []

    # embed piece — grabs embed_tokens.weight plus .scales/.biases if quantized
    embed = _collect(weights, _EMBED_PREFIX, "model.")
    mx.save_safetensors(str(out_dir / "embed.safetensors"), embed)
    pieces.append(Piece("embed", "embed.safetensors", sorted(embed)))

    # per-layer pieces
    for layer in range(num_layers):
        layer_weights = {}
        for key, arr in weights.items():
            if not key.startswith(_LAYER_PREFIX):
                continue
            idx, sub = _strip_layer_prefix(key)
            if idx == layer:
                layer_weights[sub] = arr

        is_moe = layer_key_fn is not None and layer_key_fn(arch_args, layer) == "moe"
        if is_moe:
            _pack_moe_layer(out_dir, layer, layer_weights, expert_prefix, num_experts, pieces)
        else:
            pid = Manifest.layer_piece_id(layer)
            fname = f"{pid}.safetensors"
            mx.save_safetensors(str(out_dir / fname), layer_weights)
            pieces.append(Piece(pid, fname, sorted(layer_weights)))

    # norm + (optional) head piece — both grab .scales/.biases if quantized
    norm_head = _collect(weights, _NORM_PREFIX, "model.")
    if not tied:
        norm_head.update(_collect(weights, _HEAD_PREFIX, ""))

    # Special handling for gemma4_assistant masked_embedding
    if config.get("model_type") == "gemma4_assistant":
        masked_embed = _collect(weights, "masked_embedding.", "")
        if masked_embed:
            # Pack masked_embedding as a separate piece
            mx.save_safetensors(str(out_dir / "masked_embed.safetensors"), masked_embed)
            pieces.append(Piece("masked_embed", "masked_embed.safetensors", sorted(masked_embed)))

    mx.save_safetensors(str(out_dir / "norm_head.safetensors"), norm_head)
    pieces.append(Piece("norm_head", "norm_head.safetensors", sorted(norm_head)))

    manifest = Manifest(
        model_type=config["model_type"],
        config=config,
        tie_word_embeddings=tied,
        num_layers=num_layers,
        pieces=pieces,
    )
    manifest.save(out_dir / "manifest.json")
    return manifest
