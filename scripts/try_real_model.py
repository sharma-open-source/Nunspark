#!/usr/bin/env python
"""Try NunSpark on a real Llama-architecture model (e.g. a 1B).

What it does:
  1. Resolves a model: a local MLX dir, or a Hugging Face repo id (downloaded
     + cached via mlx_lm).
  2. Re-saves it into the single-file mlx-lm layout NunSpark's packer expects
     (config.json + model.safetensors), regardless of how the source was sharded.
  3. Packs it into NunSpark's per-layer pieces.
  4. Streams generation with a chosen --budget, reporting peak unified memory,
     tokens/sec, and cache hit/miss stats.
  5. Sanity-checks the first few greedy tokens against the full-load mlx-lm run.

REQUIREMENTS for the model:
  - A supported decoder-only architecture. Many families are registered, incl.
    Llama (SmolLM2, TinyLlama, Llama-3.2-1B...), Mistral, Phi-3, Qwen2,
    Qwen3 dense (Qwen3-0.6B ... Qwen3-32B), Qwen3-MoE with selective-expert
    streaming (e.g. Qwen3-30B-A3B), Gemma3/Gemma4, GLM/GLM-4, OLMo-2, InternLM3,
    gpt-oss, and more. For the live list, see
    nunspark.architectures.supported_model_types(); the packer will reject an
    unsupported model_type. Multimodal models are NOT supported (text decoders
    only). For an MoE model, (re)pack with --repack to get the selective-expert
    layout (only the fired experts are streamed per sweep).
  - Quantized (4-bit) OR fp16/bf16 MLX models are both supported.
    For 4-bit, use an mlx-community *-4bit repo, or convert:
      python -m mlx_lm convert --hf-path <repo> -q --mlx-path <dir>

Usage:
  uv run python scripts/try_real_model.py --model mlx-community/SmolLM2-360M-Instruct-bf16 \
      --prompt "Explain streaming LLM inference in one sentence." \
      --budget 256MB --max-tokens 60

  # or point at a locally converted dir:
  uv run python scripts/try_real_model.py --model ./smol-mlx --budget 1GB

  # A/B linear vs tree speculative decoding (needs a draft sharing the tokenizer):
  uv run python scripts/try_real_model.py --model <qwen3-moe> --repack \
      --draft Qwen/Qwen3-0.6B --ab-spec --temp 0.0 --budget 4GB
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from nunspark.packer import pack
from nunspark.engine import StreamingEngine
from nunspark.generate import stream_generate, speculative_generate, SpecStats
from nunspark.tree_spec import tree_speculative_generate
from nunspark.tree_shape import TreeShape
from nunspark.kv_store import KVStore
from transformers import AutoTokenizer


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mult in [("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000),
                         ("T", 1000**4), ("G", 1000**3), ("M", 1000**2), ("K", 1000), ("B", 1)]:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def _resolve_config(model_arg: str) -> dict:
    """Return the model's config.json as a dict (local dir or HF snapshot)."""
    p = Path(model_arg)
    if p.exists():
        return json.loads((p / "config.json").read_text())
    from huggingface_hub import snapshot_download
    local = snapshot_download(model_arg, allow_patterns=["config.json"])
    return json.loads((Path(local) / "config.json").read_text())


def _run_one_spec(packed, manifest, draft_model, ids, mode, args, eos,
                  work, budget, kv_budget) -> dict:
    """Run ONE speculative mode ('linear' or 'tree') on a FRESH engine (cold cache)
    so the A/B is fair, and return its metrics. Disk reads == engine.cache.misses."""
    engine = StreamingEngine(packed, manifest, budget_bytes=budget,
                             prefetch=not args.no_prefetch,
                             io_threads=args.io_threads, warm_window=args.warm_window)
    try:
        kv = KVStore(work / f"kv_ab_{mode}", budget_bytes=kv_budget,
                     prefetch=not args.no_prefetch)
        try:
            stats = SpecStats()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            out_ids: list[int] = []
            if mode == "tree":
                shape = TreeShape([int(x) for x in args.tree_branching.split(",")])
                it = tree_speculative_generate(
                    engine, draft_model, ids, shape=shape, max_tokens=args.max_tokens,
                    temp=args.temp, kv=kv, eos_id=eos, stats=stats)
            else:
                it = speculative_generate(
                    engine, draft_model, ids, max_tokens=args.max_tokens,
                    num_draft_tokens=args.draft_tokens, accept_top_k=args.accept_top_k,
                    kv=kv, eos_id=eos, stats=stats)
            for tok in it:
                out_ids.append(tok)
                if tok == eos:
                    break
            dt = time.perf_counter() - t0
            peak_gb = mx.get_peak_memory() / 1e9
            misses = engine.cache.misses
        finally:
            kv.close()
    finally:
        engine.close()
    return {
        "mode": mode, "tokens": len(out_ids), "dt": dt,
        "tok_s": len(out_ids) / dt if dt > 0 else 0.0,
        "M": stats.multiplier, "passes": stats.target_passes,
        "peak_gb": peak_gb, "misses": misses,
    }


def generation_worker(token_iter, out_queue, eos=None):
    """Fetch tokens from the model and put them into the queue."""
    for tok in token_iter:
        out_queue.put(tok)          # Non‑blocking (queue size may be large enough)
        if tok == eos:
            break
    out_queue.put(None)             # Sentinel to signal end of generation


def printer_worker(out_queue, tokenizer, out_ids_list, delay_per_char=1.0):
    """Consume tokens from the queue and print them with delayed characters.
    out_ids_list is a shared list that will be populated with output token IDs."""
    decoded_so_far = ""
    while True:
        tok = out_queue.get()        # Blocks until a token is available
        if tok is None:              # End of generation
            break
        out_ids_list.append(tok)
        current = tokenizer.decode(out_ids_list)
        new_chars = current[len(decoded_so_far):]
        for ch in new_chars:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay_per_char)
        decoded_so_far = current
    # Ensure final newline if needed
    sys.stdout.write('\n')


def _run_ab_spec(packed, manifest, draft_model, ids, args, eos, work,
                 budget, kv_budget, is_moe_pack) -> int:
    """A/B the prompt under linear vs tree speculative decoding; print a comparison."""
    print("\n===== A/B: linear vs tree speculative decoding =====")
    if args.temp > 0.0:
        print("note: linear speculation is greedy-only (it ignores --temp); at temp>0 it "
              "decodes greedily while tree samples, so their OUTPUTS differ -- compare "
              "speed/acceptance, not the text. Use --temp 0 for an apples-to-apples run.")
    results = []
    for mode in ("linear", "tree"):
        print(f"running {mode} ...")
        results.append(_run_one_spec(packed, manifest, draft_model, ids, mode, args,
                                     eos, work, budget, kv_budget))

    print(f"\n{'mode':>8} | {'tok/s':>7} | {'M':>5} | {'sweeps':>6} | "
          f"{'disk reads':>10} | {'peak GB':>7} | {'tokens':>6}")
    print("-" * 72)
    for r in results:
        print(f"{r['mode']:>8} | {r['tok_s']:>7.2f} | {r['M']:>5.2f} | "
              f"{r['passes']:>6} | {r['misses']:>10} | {r['peak_gb']:>7.2f} | "
              f"{r['tokens']:>6}")

    best = max(results, key=lambda r: r["tok_s"])
    other = next(r for r in results if r["mode"] != best["mode"])
    if other["tok_s"] > 0:
        gain = (best["tok_s"] / other["tok_s"] - 1) * 100
        print(f"-> winner: {best['mode']} ({gain:+.0f}% tok/s vs {other['mode']})")
    if is_moe_pack:
        print("note: on a MoE target, tree reassembles the full layer (reads ALL experts/"
              "sweep -- see 'disk reads'); linear keeps the selective-expert path.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF repo id (e.g. mlx-community/SmolLM2-360M-Instruct-bf16) or local MLX dir")
    ap.add_argument("--packed-dir", default=None,
                    help="Path to pre-packed NunSpark model directory (contains manifest.json and *.safetensors)")
    ap.add_argument("--tokenizer-model", default=None,
                    help="HF repo id or local path for tokenizer (required when using --packed-dir)")
    ap.add_argument("--prompt", default="Explain streaming LLM inference in one sentence.")
    ap.add_argument("--budget", default="512MB", help="resident weight budget, e.g. 256MB, 2GB")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature (0.0 = greedy; default 0.0)")
    ap.add_argument("--workdir", default=".nunspark_tmp", help="scratch dir for repacking")
    ap.add_argument("--repack", action="store_true",
                    help="force re-pack even if packed pieces already exist in workdir")
    ap.add_argument("--pack-only", action="store_true",
                    help="download + pack the model into NunSpark pieces, then exit "
                         "(no generation). Use to prep a large target before a speculative run.")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--io-threads", type=int, default=1,
                    help="parallel raw-reader threads warming the page cache (1 = off)")
    ap.add_argument("--warm-window", type=int, default=1,
                    help="layers prefetched ahead per step (set >1 with --io-threads >1)")
    ap.add_argument("--kv-budget", default="1TB",
                    help="resident KV-cache budget, e.g. 256MB (default: all resident)")
    ap.add_argument("--draft", default=None,
                    help="HF repo id of a small draft model for speculative decoding. "
                         "Must share the target's tokenizer/vocab (use the same model "
                         "family). For a 'thinking'/reasoning target, give the draft the "
                         "SAME thinking mode so it proposes reasoning tokens too -- a "
                         "mismatch stays correct but slows the think phase. Enables "
                         "speculative mode.")
    ap.add_argument("--draft-tokens", type=int, default=16,
                    help="number of tokens the draft proposes per target sweep (default 16)")
    ap.add_argument("--accept-top-k", type=int, default=1,
                    help="relaxed-acceptance fast mode for LINEAR speculative decoding: "
                         "accept a draft token if it is within the target's top-k logits "
                         "(1 = lossless greedy, the default). Higher = faster but lossy; "
                         "see 'deviation rate' in the summary. Ignored by --tree.")
    ap.add_argument("--tree", action="store_true",
                    help="use tree speculative decoding (SpecInfer sampling over a fixed "
                         "tree of root-to-leaf paths) instead of the linear draft chain")
    ap.add_argument("--tree-branching", default="4,2,2,1,1",
                    help="comma-separated branching factor per tree level (default 4,2,2,1,1)")
    ap.add_argument("--ab-spec", action="store_true",
                    help="run the prompt under BOTH linear and tree speculative decoding "
                         "(fresh cold-cache engine each) and print a tok/s / acceptance / "
                         "disk-read / peak-memory comparison. Requires --draft; uses "
                         "--temp, --draft-tokens, --tree-branching.")
    ap.add_argument("--check-tokens", type=int, default=8,
                    help="how many greedy tokens to verify against full-load mlx-lm (0 to skip)")
    args = ap.parse_args()

    # Validate that either --model or --packed-dir is provided
    if not args.model and not args.packed_dir:
        ap.error("Either --model or --packed-dir must be specified")
    if args.model and args.packed_dir:
        ap.error("Cannot specify both --model and --packed-dir")

    work = Path(args.workdir)

    # Handle --packed-dir: use existing packed model directly
    if args.packed_dir:
        packed = Path(args.packed_dir)
        if not packed.exists():
            print(f"ERROR: Packed directory {packed} does not exist")
            return 2
        manifest_path = packed / "manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: manifest.json not found in {packed}")
            return 2
        reuse = True  # Force reuse mode for pre-packed models
        print(f"Using pre-packed model from {packed}")
    else:
        packed = work / "packed"
        work.mkdir(parents=True, exist_ok=True)
        manifest_path = packed / "manifest.json"
        reuse = manifest_path.exists() and not args.repack
    reuse = manifest_path.exists() and not args.repack

    if reuse:
        # Reuse the existing pack: skip the full weight load entirely and pull
        # only the tokenizer. config comes from the manifest the packer wrote.
        from huggingface_hub import snapshot_download
        from mlx_lm.tokenizer_utils import load as load_tokenizer
        from mlx_lm.utils import hf_repo_to_path
        from nunspark.manifest import Manifest

        manifest = Manifest.load(manifest_path)
        config = manifest.config
        model = None  # not loaded; correctness check is unavailable in reuse mode
        print(f"Reusing packed model at {packed} ({manifest.num_layers} layers) "
              f"-- pass --repack to force a fresh pack.")

        # Determine tokenizer path: from --tokenizer-model, --model, or manifest hints
        tok_path = None
        if args.tokenizer_model:
            if Path(args.tokenizer_model).exists():
                tok_path = Path(args.tokenizer_model)
            else:
                # Try to resolve HF repo to local cache
                try:
                    tok_path = Path(snapshot_download(args.tokenizer_model, local_files_only=True))
                except Exception:
                    print(f"WARNING: Could not find tokenizer for {args.tokenizer_model} in local cache")
                    print(f"Attempting online lookup...")
                    try:
                        tok_path = Path(snapshot_download(args.tokenizer_model, local_files_only=False))
                    except Exception as e:
                        print(f"ERROR: Failed to download tokenizer: {e}")
                        return 2
        elif args.model:
            if Path(args.model).exists():
                tok_path = Path(args.model)
            else:
                try:
                    tok_path = Path(snapshot_download(args.model, local_files_only=True))
                except Exception:
                    try:
                        tok_path = Path(snapshot_download(args.model, local_files_only=False))
                    except Exception as e:
                        print(f"ERROR: Failed to get tokenizer: {e}")
                        return 2
        else:
            # Try to infer from HF cache based on model_type in manifest
            model_type = config.get("model_type", "")
            try:
                if "qwen2" in model_type:
                    tok_path = Path(snapshot_download("mlx-community/Qwen2.5-32B-4bit", local_files_only=True))
                elif "qwen3_moe" in model_type:
                    tok_path = Path(snapshot_download("mlx-community/Qwen3-30B-A3B-4bit", local_files_only=True))
                else:
                    print(f"ERROR: Cannot infer tokenizer path for model_type={model_type}. "
                          f"Please specify --tokenizer-model")
                    return 2
            except Exception as e:
                print(f"ERROR: Cannot infer tokenizer from cache: {e}")
                return 2

        tokenizer = load_tokenizer(Path(tok_path))
    else:
        # 1) Load (downloads + merges shards) and confirm architecture.
        print(f"Loading {args.model} ...")
        model, tokenizer = load(args.model)
        config = _resolve_config(args.model)
        from nunspark.architectures import supported_model_types
        if config.get("model_type") not in supported_model_types():
            print(f"ERROR: model_type={config.get('model_type')!r} is not supported. "
                  f"NunSpark streams: {', '.join(supported_model_types())}.")
            return 2

        # 2) Pack straight from the in-memory weights (no safetensors round-trip).
        is_quant = "quantization" in config
        if is_quant:
            q = config["quantization"]
            print(f"  quantized model detected: {q['bits']}-bit, group_size={q['group_size']}")
        flat = dict(tree_flatten(model.parameters()))
        print(f"  packing {len(flat)} in-memory tensors")

        # 3) Pack into NunSpark pieces.
        manifest = pack(None, packed, weights=flat, config=config)

    total = sum(f.stat().st_size for f in packed.glob("*.safetensors"))
    print(f"Packed {manifest.num_layers} layers, {total/1e9:.2f} GB on disk at {packed}")

    # A MoE pack has per-layer `core` pieces (selective-expert layout); detect it so
    # we can flag the --tree tradeoff. has_piece is O(1), so scanning layers is cheap.
    is_moe_pack = any(
        manifest.has_piece(manifest.layer_core_piece_id(layer))
        for layer in range(manifest.num_layers)
    )

    if args.pack_only:
        print("(--pack-only) packed pieces ready; skipping generation.")
        return 0

    # 4) Tokenize the prompt (chat template if the model has one).
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True)
        if isinstance(ids, str):
            ids = tokenizer.encode(ids)
    except Exception:
        ids = tokenizer.encode(args.prompt)
    ids = list(ids)
    print(f"Prompt -> {len(ids)} tokens")

    if args.tree and not args.draft:
        print("WARNING: --tree has no effect without --draft (no draft model loaded); "
              "falling back to plain streaming.")

    draft_model = None
    if args.draft:
        from mlx_lm import load as _load_full
        print(f"Loading draft {args.draft} ...")
        draft_model, draft_tok = _load_full(args.draft)
        # Speculative decoding needs the draft & target to map text to the SAME token
        # ids (a shared tokenizer/vocab, including any <think> special tokens). Warn --
        # don't hard-fail -- so experimentation is still possible; a real mismatch shows
        # up as a near-zero acceptance multiplier (and verification may error).
        try:
            probe = "The quick brown fox jumps over the lazy dog 0123456789."
            if list(tokenizer.encode(probe)) != list(draft_tok.encode(probe)):
                print("WARNING: draft and target tokenizers produce DIFFERENT token ids "
                      "for a probe string -- they do NOT share a vocab. Use a draft from "
                      "the same model family, or acceptance will be ~0.")
        except Exception:
            d, t = draft_model.args.vocab_size, config["vocab_size"]
            if d != t:
                print(f"WARNING: draft vocab_size={d} != target vocab_size={t}; the draft "
                      f"must share the target's tokenizer/vocab for speculative decoding.")

    # 5) Stream-generate with live token printing.
    budget = _parse_size(args.budget)
    kv_budget = _parse_size(args.kv_budget)
    eos = getattr(tokenizer, "eos_token_id", None)

    if args.ab_spec:
        if draft_model is None:
            print("ERROR: --ab-spec requires --draft (both modes are speculative).")
            return 2
        return _run_ab_spec(packed, manifest, draft_model, ids, args, eos, work,
                            budget, kv_budget, is_moe_pack)

    engine = StreamingEngine(packed, manifest, budget_bytes=budget,
                             prefetch=not args.no_prefetch,
                             io_threads=args.io_threads, warm_window=args.warm_window)
    try:
        kv = KVStore(work / "kv", budget_bytes=kv_budget, prefetch=not args.no_prefetch)
        try:
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            out_ids: list[int] = []
            decoded_so_far = ""
            print("\n===== OUTPUT =====")
            spec_stats = SpecStats() if draft_model is not None else None
            if draft_model is not None and args.tree:
                shape = TreeShape([int(x) for x in args.tree_branching.split(",")])
                token_iter = tree_speculative_generate(
                    engine, draft_model, ids, shape=shape,
                    max_tokens=args.max_tokens, temp=args.temp, kv=kv,
                    eos_id=eos, stats=spec_stats)
            elif draft_model is not None:
                token_iter = speculative_generate(
                    engine, draft_model, ids, max_tokens=args.max_tokens,
                    num_draft_tokens=args.draft_tokens, accept_top_k=args.accept_top_k,
                    kv=kv, eos_id=eos, stats=spec_stats)
            else:
                token_iter = stream_generate(engine, ids, max_tokens=args.max_tokens,
                                          temp=args.temp, kv=kv)
            # Thread-safe queue for async printing
            out_queue = queue.Queue()
            out_ids = []  # Shared list for printer_thread to populate

            # Start printer thread (daemon so it exits when main thread ends)
            delay = 0.01
            printer_thread = threading.Thread(
                target=printer_worker,
                args=(out_queue, tokenizer, out_ids, delay)
            )
            printer_thread.daemon = True
            printer_thread.start()

            # Run generation in the main thread
            generation_worker(token_iter, out_queue, eos)

            # Wait for printer to finish (will happen after sentinel is processed)
            printer_thread.join()
            dt = time.perf_counter() - t0   # capture before cleanup
            print("\n==================")
        finally:
            kv.close()
    finally:
        engine.close()
    peak_gb = mx.get_peak_memory() / 1e9
    print(f"budget                : {args.budget}  ({budget/1e9:.2f} GB)")
    print(f"generated tokens      : {len(out_ids)} in {dt:.1f}s  "
          f"({len(out_ids)/dt:.2f} tok/s, prefetch={'off' if args.no_prefetch else 'on'})")
    print(f"peak unified memory   : {peak_gb:.2f} GB")
    print(f"resident weight peak  : {engine.cache.peak_bytes/1e9:.2f} GB "
          f"(of {total/1e9:.2f} GB model)")
    print(f"cache hits / misses   : {engine.cache.hits} / {engine.cache.misses}")
    print(f"kv-budget             : {args.kv_budget}  ({kv_budget/1e9:.2f} GB)")
    print(f"kv resident peak      : {kv.peak_bytes/1e9:.2f} GB")
    print(f"kv hits / misses      : {kv.hits} / {kv.misses}")
    if spec_stats is not None:
        print(f"draft model           : {args.draft}")
        if args.tree:
            print(f"tree branching        : {args.tree_branching}")
        else:
            print(f"draft tokens / sweep  : {args.draft_tokens}")
            print(f"accept-top-k          : {args.accept_top_k}  "
                  f"({'lossless' if args.accept_top_k <= 1 else 'fast mode'})")
            print(f"deviation rate        : {spec_stats.deviation_rate:.3f}  "
                  f"(off-path {spec_stats.accepted_offpath}/{spec_stats.accepted_total} accepted)")
        print(f"target passes (sweeps): {spec_stats.target_passes}")
        print(f"acceptance multiplier : {spec_stats.multiplier:.2f}  "
              f"(tokens {spec_stats.tokens_emitted} / passes {spec_stats.target_passes})")
        eff = len(out_ids) / dt if dt > 0 else 0.0
        print(f"effective speed       : {eff:.2f} tok/s "
              f"(vs ~{eff/spec_stats.multiplier:.2f} tok/s naive-stream equivalent)")
        if args.tree and is_moe_pack:
            print("note                  : --tree on a MoE target reads ALL experts/sweep "
                  "(no selective-expert saving); drop --tree for linear spec to keep it.")

    # 6) Correctness: compare first K greedy tokens to a full-load mlx-lm run.
    # Only valid at temp=0.0 (greedy); sampling is non-deterministic.
    if args.check_tokens > 0:
        if model is None:
            print(f"\ncorrectness check skipped (reused packed model; "
                  f"full-load reference not available -- use --repack to enable)")
        elif args.temp > 0.0:
            print(f"\ncorrectness check skipped (temp={args.temp} > 0; "
                  f"sampling is non-deterministic)")
        else:
            k = args.check_tokens
            cache = [KVCache() for _ in range(config["num_hidden_layers"])]
            logits = model(mx.array(ids)[None], cache=cache)[:, -1, :]
            ref = []
            for _ in range(k):
                nxt = int(mx.argmax(logits, axis=-1).item())
                ref.append(nxt)
                logits = model(mx.array([nxt])[None], cache=cache)[:, -1, :]
            match = out_ids[:k] == ref
            print(f"\ncorrectness (first {k} greedy tokens vs full-load): "
                  f"{'MATCH ✓' if match else 'MISMATCH ✗'}")
            if not match:
                print("  nunspark:", out_ids[:k])
                print("  full-load:", ref)

    print(f"\n(scratch dir {work} can be deleted; packed model is reusable.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
