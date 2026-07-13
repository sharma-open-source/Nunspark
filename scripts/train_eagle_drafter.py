#!/usr/bin/env python3
"""Train an EAGLE feature-level drafter for a NunSpark-packed target model.

Collects (h_t, t_{t+1}, h_{t+1}, t_{t+2}) quadruples by running the target
model over text, then trains the 2-layer EAGLE drafter with MSE + CE loss.

Usage:
  python scripts/train_eagle_drafter.py                      \
      --model /path/to/packed/                               \
      --data training_data.txt                               \
      --output checkpoints/eagle_drafter                     \
      --seq-len 256 --batch-size 8 --lr 1e-4 --num-epochs 3

The script works in *online* mode (collects and trains on-the-fly) to
avoid storing large hidden-state arrays on disk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from nunspark.eagle_drafter import EagleConfig, EagleDrafterModel
from nunspark.engine import StreamingEngine
from nunspark.manifest import Manifest


def load_and_tokenize(text_path: str | Path, tokenizer, seq_len: int, stride: int | None = None):
    """Load a text file, tokenize, and yield (overlapping) chunks.

    Args:
        text_path: Path to a plain-text file.
        tokenizer: HuggingFace tokenizer.
        seq_len: Target sequence length per chunk.
        stride: Overlap between consecutive chunks (defaults to seq_len).

    Yields:
        Lists of token IDs, each of length seq_len.
    """
    if stride is None:
        stride = seq_len
    text = Path(text_path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    if not tokens:
        return
    for i in range(0, len(tokens) - seq_len + 1, stride):
        yield tokens[i : i + seq_len]


def collect_training_data(
    engine: StreamingEngine,
    chunk: list[int],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run the target model on `chunk` and extract sequence training data.

    Instead of extracting individual (h_t, t_{t+1}) pairs, returns the full
    sequence so the drafter processes L positions together — enabling its
    self-attention to learn contextual patterns.

    Returns:
        h_seq:  (1, L, hidden_size) — target hidden states at positions 0..L-1
        t_seq:  (1, L) — token IDs at positions 1..L  (shifted right by 1)
        h_target: (1, L, hidden_size) — target hidden states at positions 1..L
        t_target: (1, L) — token IDs at positions 2..L+1
    where L = len(chunk) - 2 (we lose the last 2 positions as targets).
    """
    L = len(chunk) - 2
    tokens = mx.array(chunk)[None]  # (1, seq_len)
    _ = engine.forward(tokens, kv=None)
    hidden = engine.last_hidden_state()  # (1, seq_len, hidden_size)

    h_seq = hidden[:, :L, :]             # (1, L, hidden_size)  — h_0..h_{L-1}
    t_seq = mx.array([chunk[1:L+1]])     # (1, L) — t_1..t_L
    h_target = hidden[:, 1:L+1, :]       # (1, L, hidden_size)  — h_1..h_L
    t_target = mx.array([chunk[2:L+2]])   # (1, L) — t_2..t_{L+1}
    return h_seq, t_seq, h_target, t_target


def sequence_loss(
    model: EagleDrafterModel,
    h_seq: mx.array,
    t_seq: mx.array,
    h_target: mx.array,
    t_target: mx.array,
    causal_mask: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute MSE + CE loss over a sequence of L positions.

    The drafter processes all L positions in one forward, enabling self-attention
    to attend to previous positions.

    Returns (loss, mse, ce) — each a scalar array.
    """
    pred_h, logits = model(h_seq, t_seq, mask=causal_mask)
    mse = ((pred_h - h_target) ** 2).mean()
    ce = nn.losses.cross_entropy(logits, t_target, reduction="mean")
    lambda_mse = 0.1
    loss = lambda_mse * mse + (1.0 - lambda_mse) * ce
    return loss, mse, ce


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train an EAGLE feature-level drafter."
    )
    ap.add_argument("--model", required=True,
                    help="Path to packed model directory (contains manifest.json + pieces)")
    ap.add_argument("--data", required=True,
                    help="Plain-text file for training data (will be tokenized with target's tokenizer)")
    ap.add_argument("--output", default="checkpoints/eagle_drafter",
                    help="Output directory for checkpoints (default: checkpoints/eagle_drafter)")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="Token sequence length per chunk (default: 256)")
    ap.add_argument("--stride", type=int, default=None,
                    help="Chunk stride for overlapping windows (default: equal to --seq-len)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Drafter training batch size (default: 8)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Peak learning rate (default: 1e-4)")
    ap.add_argument("--num-epochs", type=int, default=3,
                    help="Number of passes over the training data (default: 3)")
    ap.add_argument("--save-every", type=int, default=200,
                    help="Save a checkpoint every N training steps (default: 200)")
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="Linear LR warmup steps (default: 100)")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay (default: 0.01)")
    ap.add_argument("--freeze-embed", action="store_true", default=False,
                    help="Freeze embedding table after initialization (recommended for large vocab)")
    ap.add_argument("--init-embed", default=None,
                    help="Path to packed model for embedding initialization (default: uses --model)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Limit number of chunks for debugging (default: unlimited)")
    args = ap.parse_args()

    packed = Path(args.model)
    manifest_path = packed / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Is this a packed model directory?")
        return 1

    manifest = Manifest.load(manifest_path)
    print(f"Target model: {manifest.num_layers} layers, "
          f"hidden={manifest.config['hidden_size']}, "
          f"vocab={manifest.config['vocab_size']}")

    # ---- Load target model as StreamingEngine ----
    # Budget: enough to fit the norm+head pieces (resident). Layer weights are
    # streamed per forward pass; we give a generous budget so the page cache
    # can help on repeated reads.
    print("Loading target model (StreamingEngine) …", end=" ", flush=True)
    t0 = time.perf_counter()
    engine = StreamingEngine(packed, manifest, budget_bytes=2 * 10**9)
    print(f"{time.perf_counter() - t0:.1f}s")

    # ---- Create EAGLE drafter (random init) ----
    eagle_config = EagleConfig.from_target_config(manifest.config)
    print(f"Drafter config: {eagle_config.num_hidden_layers} layers, "
          f"hidden={eagle_config.hidden_size}, "
          f"intermediate={eagle_config.intermediate_size}, "
          f"heads={eagle_config.num_attention_heads}, "
          f"kv_heads={eagle_config.num_key_value_heads}")
    drafter = EagleDrafterModel(eagle_config)
    # Evaluate parameters to materialize them
    mx.eval(drafter.parameters())
    param_count = sum(v.size for _, v in tree_flatten(drafter.parameters()))
    print(f"Drafter parameters: {param_count:,} ({param_count * 4 / 1e6:.1f} MB in fp32)")

    # Optionally initialize embedding from target model (much faster convergence)
    if args.init_embed or eagle_config.vocab_size > 32000:
        init_path = args.init_embed or str(packed)
        try:
            embed_data = mx.load(f"{init_path}/embed.safetensors")
            dequant = manifest.config.get("quantization")
            if "embed_tokens.scales" in embed_data and dequant:
                dq = mx.dequantize(
                    embed_data["embed_tokens.weight"],
                    embed_data["embed_tokens.scales"],
                    embed_data["embed_tokens.biases"],
                    group_size=dequant["group_size"],
                    bits=dequant["bits"],
                )
                # Cast to fp32 — fp16 + AdamW eps underflows to 0 when grad=0 → NaN
                drafter.embed_tokens.weight = dq.astype(mx.float32)
                print(f"  initialized embed_tokens from target (dequantized fp32)")
            elif "embed_tokens.weight" in embed_data:
                drafter.embed_tokens.weight = embed_data["embed_tokens.weight"]
                print(f"  initialized embed_tokens from target (fp16)")
            mx.eval(drafter.embed_tokens.parameters())
        except Exception as e:
            print(f"  warning: could not init embed from target: {e}")

    # ---- Load tokenizer ----
    from transformers import AutoTokenizer

    # Try to infer tokenizer from model location or HF name
    model_path_or_name = manifest.config.get("_name_or_path", str(packed))
    print(f"Loading tokenizer ({model_path_or_name}) …", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    print("ok")

    # ---- Data ----
    print(f"Loading data from {args.data} …", end=" ", flush=True)
    chunks = list(
        load_and_tokenize(args.data, tokenizer, args.seq_len, args.stride)
    )
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    total_pairs = sum(len(c) - 2 for c in chunks)
    print(f"{len(chunks)} chunks, ~{total_pairs} training pairs")

    # ---- Optimiser + learning-rate schedule ----
    # For models with large vocabulary (128K+), freeze the embedding table
    # (initialized from the target) and train only the transformer layers with
    # AdamW. This avoids the ~3s/step overhead of AdamW on 263M embedding params.
    wd = 0.0 if args.freeze_embed else args.weight_decay
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=wd)
    if args.freeze_embed:
        print(f"Optimizer: AdamW (frozen embed, weight_decay=0, lr={args.lr})")
    else:
        print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={wd})")

    loss_and_grad_fn = nn.value_and_grad(drafter, lambda m, *xs: sequence_loss(m, *xs)[0])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save config
    (out_dir / "config.json").write_text(json.dumps(eagle_config.__dict__, indent=2))

    # ---- Training loop ----
    step = 0
    best_loss = float("inf")
    print(f"\n{'step':>6} | {'loss':>8} | {'mse':>8} | {'ce':>8} | {'lr':>10} | {'tok/s':>7}")
    print("-" * 60)

    for epoch in range(args.num_epochs):
        for chunk_idx, chunk in enumerate(chunks):
            h_seq, t_seq, h_target, t_target = collect_training_data(engine, chunk)

            # Create a causal mask for the sequence
            L = h_seq.shape[1]
            # (1, 1, L, L) causal mask: lower-triangular
            causal_mask = mx.tril(mx.ones((1, 1, L, L), dtype=mx.float32))

            t0_step = time.perf_counter()
            loss_val, grads = loss_and_grad_fn(
                drafter, h_seq, t_seq, h_target, t_target, causal_mask
            )
            if args.freeze_embed:
                grads["embed_tokens"]["weight"] = mx.zeros_like(
                    grads["embed_tokens"]["weight"]
                )
            optimizer.update(drafter, grads)
            mx.eval(drafter.parameters(), optimizer.state)

            dt = time.perf_counter() - t0_step
            tokens_per_sec = L / dt if dt > 0 else 0.0

            if step % 5 == 0:
                _, mse_v, ce_v = sequence_loss(
                    drafter, h_seq, t_seq, h_target, t_target, causal_mask
                )
                print(f"{step:>6} | {loss_val.item():>8.4f} | {mse_v.item():>8.6f} | "
                      f"{ce_v.item():>8.4f} | {args.lr:>10.2e} | {tokens_per_sec:>7.1f}")

            if step > 0 and step % args.save_every == 0:
                # Cast embed to fp16 for saving (disk space); restore fp32
                # after because AdamW eps underflows to 0 in fp16 when grad=0
                if args.freeze_embed:
                    saved_embed = drafter.embed_tokens.weight
                    if saved_embed.dtype == mx.float32:
                        drafter.embed_tokens.weight = saved_embed.astype(mx.float16)
                        mx.eval(drafter.embed_tokens.weight)
                weights = dict(tree_flatten(drafter.parameters()))
                mx.save_safetensors(
                    str(out_dir / f"checkpoint_{step}.safetensors"), weights
                )
                if args.freeze_embed and saved_embed.dtype == mx.float32:
                    drafter.embed_tokens.weight = saved_embed
                    mx.eval(drafter.embed_tokens.weight)

            step += 1

        # End of epoch: save checkpoint
        if args.freeze_embed:
            saved_embed = drafter.embed_tokens.weight
            if saved_embed.dtype == mx.float32:
                drafter.embed_tokens.weight = saved_embed.astype(mx.float16)
                mx.eval(drafter.embed_tokens.weight)
        weights = dict(tree_flatten(drafter.parameters()))
        mx.save_safetensors(str(out_dir / f"epoch_{epoch}.safetensors"), weights)
        if args.freeze_embed and saved_embed.dtype == mx.float32:
            drafter.embed_tokens.weight = saved_embed
            mx.eval(drafter.embed_tokens.weight)
        print(f"  --- epoch {epoch} done, checkpoint saved ---")

    # Final save
    weights = dict(tree_flatten(drafter.parameters()))
    mx.save_safetensors(str(out_dir / "final.safetensors"), weights)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {out_dir}")
    print(f"  config.json  — model configuration")
    print(f"  final.safetensors — trained weights (tree-flattened dict)")
    print(f"\nTo load: drafter = EagleDrafterModel(EagleConfig.from_dict(config))")
    print(f"         drafter.update(mx.load('final.safetensors'))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
