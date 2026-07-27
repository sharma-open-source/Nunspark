from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .manifest import Manifest
from .packer import pack as pack_model
from .engine import StreamingEngine
from .generate import generate as run_generate, PREFILL_CHUNK
from .server import run_server
from .sysmem import auto_budget_bytes, resolve_budget as _resolve_budget

_UNITS = [("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("KB", 1000),
          ("T", 1000**4), ("G", 1000**3), ("M", 1000**2), ("K", 1000), ("B", 1)]


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mult in _UNITS:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def _kv_quant_from_args(args):
    from .archspec import KVQuant
    if args.kv_bits is None:
        return None
    return KVQuant(bits=args.kv_bits, group_size=args.kv_group_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nunspark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pack = sub.add_parser("pack", help="pack an mlx-lm model into NunSpark pieces")
    p_pack.add_argument("model_dir",
                        help="local path or Hugging Face repo id (e.g. mlx-community/SmolLM2-360M-Instruct-bf16)")
    p_pack.add_argument("out_dir")

    p_gen = sub.add_parser("generate", help="stream-generate from a packed model")
    p_gen.add_argument("packed_dir")
    p_gen.add_argument("--prompt", default=None, help="text prompt to generate from")
    p_gen.add_argument("--prompt-ids", default=None, help="comma-separated token ids (alternative to --prompt)")
    p_gen.add_argument("--max-tokens", type=int, default=32)
    p_gen.add_argument("--temp", type=float, default=0.0)
    p_gen.add_argument("--budget", default="auto",
                       help='resident weight budget, e.g. 512MB, 8GB, or "auto" '
                            '(default: 75% of RAM minus 4GB)')
    p_gen.add_argument("--no-prefetch", action="store_true",
                       help="disable background prefetch overlap")
    p_gen.add_argument("--io-threads", type=int, default=1,
                       help="parallel raw-reader threads warming the page cache "
                            "(1 = off; try 8 with --warm-window 4)")
    p_gen.add_argument("--warm-window", type=int, default=1,
                       help="layers prefetched ahead per step (set >1 with --io-threads >1)")
    p_gen.add_argument("--kv-budget", default="1TB",
                       help="resident KV-cache budget, e.g. 256MB, 4GB "
                            "(default keeps all layers resident)")
    p_gen.add_argument("--kv-bits", type=int, default=None, choices=[4, 8],
                       help="quantize the KV cache to N bits (default: fp16)")
    p_gen.add_argument("--kv-group-size", type=int, default=64,
                       help="quantization group size for --kv-bits (default 64)")
    p_gen.add_argument("--draft-model", default=None,
                       help="Path to draft model for speculative decoding (mlx_lm format)")
    p_gen.add_argument("--eagle-drafter", default=None,
                       help="Path to trained EAGLE drafter weights (feature-level speculation, no full draft model needed)")
    p_gen.add_argument("--ngram-draft", action="store_true",
                       help="prompt-lookup (n-gram) speculative decoding: model-free drafter "
                            "that proposes --num-draft-tokens tokens by finding the most recent "
                            "prior occurrence of the context suffix (lossless; no draft model). "
                            "Mutually exclusive with --draft-model / --eagle-drafter")
    p_gen.add_argument("--ngram-max", type=int, default=3,
                       help="longest suffix n-gram tried by --ngram-draft (default: 3)")
    p_gen.add_argument("--no-ngram-adaptive", dest="ngram_adaptive",
                       action="store_false", default=True,
                       help="disable adaptive proposal-length shrink/grow for "
                            "--ngram-draft; always propose the fixed --num-draft-tokens "
                            "(default: adaptive on)")
    p_gen.add_argument("--num-draft-tokens", type=int, default=16,
                       help="Number of draft tokens to propose per iteration (default: 16)")
    p_gen.add_argument("--accept-top-k", type=int, default=1,
                       help="Acceptance threshold: 1=lossless, >1=fast mode (default: 1)")
    p_gen.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK,
                       help="prompt tokens prefilled per forward window; smaller caps "
                            "peak activation memory on long prompts (default: "
                            f"{PREFILL_CHUNK})")
    p_gen.add_argument("--no-chat-template", action="store_true",
                       help="skip chat template; encode prompt as-is (default: apply if available)")
    p_gen.add_argument("--metrics", action="store_true",
                       help="print tok/s, peak memory, cache and KV stats after generation")
    p_gen.add_argument("--expert-trace", default=None,
                       help="write a JSONL trace of fired MoE experts per layer/call to this "
                            "path (M1 locality measurement; off by default, no overhead when unset)")
    p_gen.add_argument("--expert-cache-frac", type=float, default=0.9,
                       help="fraction of --budget reserved for the LRU expert-piece cache "
                            "region on selectively-packed MoE models (default 0.9; ignored "
                            "for models without expert pieces)")
    p_gen.add_argument("--expert-prefetch", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="temporal expert prefetch (plan4 M3a): overlap fired-expert "
                            "loads with attention/router compute by speculatively "
                            "prefetching the previous pass's per-layer expert sets "
                            "(default on; --no-expert-prefetch to disable)")
    p_gen.add_argument("--lookahead", action="store_true",
                       help="experimental cross-layer MoE expert prefetch (plan7): on "
                            "decode passes, replicate the next layer's router on the "
                            "current residual stream and speculatively stage its "
                            "predicted experts. Output-identical to off; prefetch only. "
                            "Opt-in, off by default")
    p_gen.add_argument("--compact-scatter", action="store_true",
                       help="experimental compact fired-only expert scatter (plan8): "
                            "pack the ~k fired experts into a [k, ...] buffer instead of "
                            "scattering into full [num_experts, ...] buffers. "
                            "Output-identical to off; the scatter was the #1 decode "
                            "bucket (backlog #16). Opt-in, off by default")
    p_gen.add_argument("--eval-window", type=int, default=1, metavar="W",
                       help="drain the per-layer eval barrier every W-th layer "
                            "instead of every layer (plan9), pipelining ~W layers. "
                            "Output-identical to W=1; measured +7.6%% decode at W=3 "
                            "on the 30B (wire-guarded so it can't churn). Opt-in, "
                            "default 1 (off)")
    p_gen.add_argument("--no-wire", dest="wire", action="store_false",
                       help="don't raise the MLX wired-memory limit at startup. "
                            "Default is to wire it to the device's recommended "
                            "working-set size so macOS can't compress the piece "
                            "cache out from under the engine (backlog #14); "
                            "memory policy only, output-identical either way")

    p_serve = sub.add_parser("serve", help="serve a packed model behind an OpenAI v1-compatible API")
    p_serve.add_argument("packed_dir")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--model-name", default=None,
                         help="model id reported to clients (defaults to the packed dir's name)")
    p_serve.add_argument("--budget", default="auto",
                         help='resident weight budget, e.g. 512MB, 8GB, or "auto" '
                              '(default: 75% of RAM minus 4GB)')
    p_serve.add_argument("--no-prefetch", action="store_true",
                         help="disable background prefetch overlap")
    p_serve.add_argument("--io-threads", type=int, default=1,
                         help="parallel raw-reader threads warming the page cache "
                              "(1 = off; try 8 with --warm-window 4)")
    p_serve.add_argument("--warm-window", type=int, default=1,
                         help="layers prefetched ahead per step (set >1 with --io-threads >1)")
    p_serve.add_argument("--kv-budget", default="1TB",
                         help="resident KV-cache budget, e.g. 256MB, 4GB "
                              "(default keeps all layers resident)")
    p_serve.add_argument("--kv-bits", type=int, default=None, choices=[4, 8],
                         help="quantize the KV cache to N bits (default: fp16)")
    p_serve.add_argument("--kv-group-size", type=int, default=64,
                         help="quantization group size for --kv-bits (default 64)")
    p_serve.add_argument("--draft-model", default=None,
                         help="Path to draft model for speculative decoding (mlx_lm format)")
    p_serve.add_argument("--num-draft-tokens", type=int, default=16,
                         help="Number of draft tokens to propose per iteration (default: 16)")
    p_serve.add_argument("--accept-top-k", type=int, default=1,
                         help="Acceptance threshold: 1=lossless, >1=fast mode (default: 1)")
    p_serve.add_argument("--no-prefix-cache", dest="use_prefix_cache",
                         action="store_false",
                         help="disable single-slot prompt-prefix KV reuse (default: on)")
    p_serve.add_argument("--no-wire", dest="wire", action="store_false",
                         help="don't raise the MLX wired-memory limit at startup "
                              "(default: wire to the device's recommended working-set "
                              "size; see nunspark generate --help)")
    p_serve.add_argument("--lookahead", action="store_true",
                         help="experimental cross-layer MoE expert prefetch (plan7): on "
                              "decode passes, replicate the next layer's router on the "
                              "current residual stream and speculatively stage its "
                              "predicted experts. Output-identical to off; prefetch only. "
                              "Opt-in, off by default")
    p_serve.add_argument("--compact-scatter", action="store_true",
                         help="experimental compact fired-only expert scatter (plan8): "
                              "pack the ~k fired experts into a [k, ...] buffer instead "
                              "of full [num_experts, ...] buffers. Output-identical to "
                              "off (backlog #16). Opt-in, off by default")
    p_serve.add_argument("--eval-window", type=int, default=1, metavar="W",
                         help="drain the per-layer eval barrier every W-th layer "
                              "instead of every layer (plan9), pipelining ~W layers. "
                              "Output-identical to W=1; measured +7.6%% decode at W=3 "
                              "on the 30B (wire-guarded so it can't churn). Opt-in, "
                              "default 1 (off)")

    p_web = sub.add_parser("web", help="launch the local web interface")
    p_web.add_argument("--packed-root", default="models",
                       help="directory scanned for packed models (default: ./models)")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)

    p_bench = sub.add_parser(
        "bench", help="benchmark NunSpark on this machine and print a shareable report")
    p_bench.add_argument("model", nargs="?", default="mlx-community/Qwen3-30B-A3B-4bit",
                         help="HF repo id, local model dir, or already-packed dir "
                              "(default: mlx-community/Qwen3-30B-A3B-4bit)")
    p_bench.add_argument("--packed-root", default="./packed",
                         help="directory auto-packed models are written under "
                              "(default: ./packed)")
    p_bench.add_argument("--draft", default="Qwen/Qwen3-0.6B",
                         help="draft model for speculative runs (default: Qwen/Qwen3-0.6B)")
    p_bench.add_argument("--no-spec", action="store_true",
                         help="skip speculative runs entirely (greedy only)")
    p_bench.add_argument("--ngram", action="store_true",
                         help="run the speculative arm with the model-free prompt-lookup "
                              "(n-gram) drafter instead of a draft model (mode 'ngram-spec'); "
                              "uses --draft-tokens as K, no draft download")
    p_bench.add_argument("--draft-tokens", type=int, default=24)
    p_bench.add_argument("--no-ngram-adaptive", dest="ngram_adaptive",
                         action="store_false", default=True,
                         help="disable adaptive proposal-length shrink/grow for --ngram "
                              "(default: adaptive on)")
    p_bench.add_argument("--budget", default="auto",
                         help='PieceCache byte budget, e.g. 512MB, 8GB, or "auto" '
                              '(default: 75% of RAM minus 4GB)')
    p_bench.add_argument("--max-tokens", type=int, default=100)
    p_bench.add_argument("--out", default=None, help="optional path to write raw JSON results")
    p_bench.add_argument("--quick", action="store_true",
                         help="fast smoke run: single prose workload, greedy only, 50 tokens")
    p_bench.add_argument("--yes", action="store_true",
                         help="skip the confirmation prompt before packing an HF repo")
    p_bench.add_argument("--lookahead", action="store_true",
                         help="experimental cross-layer MoE expert prefetch (plan7): on "
                              "decode passes, replicate the next layer's router on the "
                              "current residual stream and speculatively stage its "
                              "predicted experts. Output-identical to off; prefetch only. "
                              "Opt-in, off by default")
    p_bench.add_argument("--no-wire", dest="wire", action="store_false",
                         help="don't raise the MLX wired-memory limit for the bench "
                              "runs (default: wired; used to A/B the unwired legacy "
                              "behavior — see nunspark generate --help)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "pack":
        manifest = pack_model(args.model_dir, args.out_dir)
        print(f"packed {manifest.num_layers} layers into {args.out_dir}",
              file=sys.stderr)
        return 0

    if args.command == "generate":
        if args.ngram_draft and (args.draft_model or args.eagle_drafter):
            print("Error: --ngram-draft is mutually exclusive with "
                  "--draft-model / --eagle-drafter", file=sys.stderr)
            return 1
        manifest = Manifest.load(Path(args.packed_dir) / "manifest.json")
        engine = StreamingEngine(
            args.packed_dir, manifest,
            budget_bytes=_resolve_budget(args.budget),
            prefetch=not args.no_prefetch,
            io_threads=args.io_threads,
            warm_window=args.warm_window,
            expert_trace=args.expert_trace,
            expert_cache_frac=args.expert_cache_frac,
            expert_prefetch=args.expert_prefetch,
            lookahead_prefetch=args.lookahead,
            wire_limit=args.wire,
            compact_scatter=args.compact_scatter,
            eval_window=args.eval_window,
        )
        try:
            from .generate import check_kv_quant_support
            check_kv_quant_support(engine, _kv_quant_from_args(args))
        except ValueError as e:
            engine.close()
            print(f"Error: {e}", file=sys.stderr)
            return 1
        try:
            from mlx_lm.tokenizer_utils import load as load_tokenizer
            from .kv_store import KVStore
            import tempfile

            tokenizer = load_tokenizer(Path(args.packed_dir))
            eos = getattr(tokenizer, "eos_token_id", None)

            # Encode prompt — apply chat template unless suppressed
            if args.prompt:
                if not args.no_chat_template:
                    try:
                        prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": args.prompt}],
                            add_generation_prompt=True, tokenize=True)
                        if isinstance(prompt, str):
                            prompt = tokenizer.encode(prompt)
                        prompt = list(prompt)
                    except Exception:
                        prompt = tokenizer.encode(args.prompt)
                else:
                    prompt = tokenizer.encode(args.prompt)
            elif args.prompt_ids:
                prompt = [int(x) for x in args.prompt_ids.split(",")]
            else:
                print("Error: either --prompt or --prompt-ids is required", file=sys.stderr)
                return 1

            print(f"Prompt -> {len(prompt)} tokens", file=sys.stderr)

            # Build the model-free n-gram drafter if requested
            ngram_drafter = None
            if args.ngram_draft:
                from .ngram_drafter import NGramDrafter
                ngram_drafter = NGramDrafter(
                    max_ngram=args.ngram_max,
                    num_draft_tokens=args.num_draft_tokens,
                    adaptive=args.ngram_adaptive,
                )
                print(f"  n-gram drafter (max_ngram={args.ngram_max}, "
                      f"K={args.num_draft_tokens}, "
                      f"adaptive={args.ngram_adaptive})", file=sys.stderr)

            # Load EAGLE drafter if provided (feature-level speculation)
            eagle_drafter = None
            if args.eagle_drafter:
                from .eagle_drafter import EagleConfig, EagleDrafter, EagleDrafterModel

                eagle_path = Path(args.eagle_drafter)
                if eagle_path.is_dir():
                    config_path = eagle_path / "config.json"
                    if config_path.exists():
                        import json
                        cfg = json.loads(config_path.read_text())
                    else:
                        # Build config from target model config
                        cfg = EagleConfig.from_target_config(manifest.config).__dict__
                    config = EagleConfig(**cfg)
                    model = EagleDrafterModel(config)
                    # Load weights if safetensors exist
                    safetensors = sorted(eagle_path.glob("*.safetensors"))
                    if safetensors:
                        weights = {}
                        for f in safetensors:
                            weights.update(mx.load(str(f)))
                        model.update(weights)
                        mx.eval(model.parameters())
                    eagle_drafter = EagleDrafter(model)
                    print(f"  loaded EAGLE drafter from {args.eagle_drafter}", file=sys.stderr)
                else:
                    print(f"Error: eagle drafter path {args.eagle_drafter} not found", file=sys.stderr)
                    return 1

            # Load draft model if provided
            draft_model = None
            is_assistant = False
            if args.draft_model:
                from mlx_lm import load as load_mlxlm
                print(f"Loading draft model {args.draft_model} ...", file=sys.stderr)

                # Check if this is a gemma4_assistant model (custom NunSpark model)
                try:
                    import json
                    # Try to read config to determine model type
                    model_path = Path(args.draft_model)
                    if model_path.exists():
                        config_path = model_path / "config.json"
                    else:
                        # It's an HF repo, fetch config
                        from huggingface_hub import snapshot_download
                        local = snapshot_download(args.draft_model, allow_patterns=["config.json"])
                        config_path = Path(local) / "config.json"

                    if config_path.exists():
                        config = json.loads(config_path.read_text())
                        model_type = config.get("model_type", "")
                        if model_type == "gemma4_assistant":
                            is_assistant = True
                except Exception as e:
                    print(f"  Warning: Could not detect model type: {e}", file=sys.stderr)

                if is_assistant:
                    # Load using our custom implementation
                    from . import gemma4_assistant

                    # Re-read config for our use and get local path
                    model_path = Path(args.draft_model)
                    if model_path.exists():
                        config_path = model_path / "config.json"
                        config = json.loads(config_path.read_text())
                        load_from_path = model_path
                    else:
                        from huggingface_hub import snapshot_download
                        local = snapshot_download(args.draft_model, allow_patterns=["*.safetensors"])
                        # Also download config
                        snapshot_download(args.draft_model, allow_patterns=["config.json"])
                        config_path = Path(local) / "config.json"
                        config = json.loads(config_path.read_text())
                        load_from_path = Path(local)  # Use downloaded local path

                    # Manually load weights from safetensors files (bypasses mlx_lm model instantiation)
                    weights = {}
                    safetensors_files = list(load_from_path.glob("*.safetensors"))
                    for st_file in safetensors_files:
                        file_weights = mx.load(str(st_file))
                        weights.update(file_weights)

                    # Normalize weight keys (strip language_model.model. prefix like gemma4)
                    normalized_weights = {}
                    for k, v in weights.items():
                        if k.startswith("language_model.model."):
                            normalized_weights[k.replace("language_model.model.", "model.")] = v
                        elif k.startswith("language_model."):
                            normalized_weights[k.replace("language_model.", "")] = v
                        else:
                            normalized_weights[k] = v
                    weights = normalized_weights

                    # Create gemma4_assistant.Model from our implementation
                    args_cls = gemma4_assistant.ModelArgs.from_dict(config)
                    draft_model = gemma4_assistant.Model(args_cls)

                    # Call sanitize to handle _token_ordering buffer (removes it from weights)
                    # This also removes quantization parameters (.biases, .scales) that don't match the model structure
                    weights = draft_model.sanitize(weights)

                    # Load only the weights that match the model structure
                    # (Sanitize removes quantization params and _token_ordering, leaving only .weight)
                    try:
                        draft_model.load_weights(weights)
                    except ValueError as e:
                        print(f"  Warning: Could not load all weights: {e}", file=sys.stderr)
                        # Fallback: try loading only .weight parameters
                        weight_only = {k: v for k, v in weights.items() if k.endswith(".weight")}
                        draft_model.load_weights(weight_only)

                    from .assistant import AssistantDrafter
                    draft_model = AssistantDrafter(draft_model)

                    print(f"  loaded gemma4_assistant from NunSpark", file=sys.stderr)
                else:
                    # Standard mlx_lm model
                    draft_model, _ = load_mlxlm(args.draft_model)

            kv_quant = _kv_quant_from_args(args)

            # Set up KVStore explicitly so we can read its stats
            tmp = tempfile.TemporaryDirectory(prefix="nunspark_kv_")
            kv = KVStore(tmp.name, budget_bytes=_parse_size(args.kv_budget),
                         prefetch=not args.no_prefetch,
                         cache_kinds=engine.cache_kinds,
                         kv_quant=kv_quant)
            try:
                out_ids: list[int] = []

                from .generate import SpecStats
                use_spec = (draft_model is not None or eagle_drafter is not None
                            or ngram_drafter is not None)
                spec_stats = SpecStats() if use_spec else None

                if ngram_drafter is not None:
                    from .generate import ngram_speculative_generate
                    token_iter = ngram_speculative_generate(
                        engine, ngram_drafter, prompt,
                        max_tokens=args.max_tokens,
                        kv=kv, eos_id=eos, stats=spec_stats,
                        kv_quant=kv_quant,
                        prefill_chunk=args.prefill_chunk,
                    )
                elif eagle_drafter is not None:
                    from .generate import eagle_speculative_generate
                    token_iter = eagle_speculative_generate(
                        engine, eagle_drafter, prompt,
                        max_tokens=args.max_tokens,
                        num_draft_tokens=args.num_draft_tokens,
                        kv=kv, eos_id=eos, stats=spec_stats,
                        kv_quant=kv_quant,
                    )
                elif is_assistant:
                    from .generate import gemma4_mtp_speculative_generate
                    token_iter = gemma4_mtp_speculative_generate(
                        engine, draft_model, prompt,
                        max_tokens=args.max_tokens,
                        num_draft_tokens=args.num_draft_tokens,
                        accept_top_k=args.accept_top_k,
                        kv=kv, eos_id=eos, stats=spec_stats,
                        kv_quant=kv_quant,
                    )
                elif draft_model is not None:
                    from .generate import speculative_generate
                    token_iter = speculative_generate(
                        engine, draft_model, prompt,
                        max_tokens=args.max_tokens,
                        num_draft_tokens=args.num_draft_tokens,
                        accept_top_k=args.accept_top_k,
                        kv=kv, eos_id=eos, stats=spec_stats,
                        kv_quant=kv_quant,
                        prefill_chunk=args.prefill_chunk,
                    )
                else:
                    from .generate import stream_generate
                    token_iter = stream_generate(
                        engine, prompt,
                        max_tokens=args.max_tokens, temp=args.temp,
                        kv=kv,
                        kv_quant=kv_quant,
                        prefill_chunk=args.prefill_chunk,
                    )

                # Decode + print on a background thread so terminal I/O does
                # not block the next forward. The generation loop only enqueues
                # tokens, so the measured dt reflects generation, not stdout.
                import queue as _queue
                import threading

                tok_q: _queue.Queue = _queue.Queue()

                def _printer() -> None:
                    decoded_so_far = ""
                    while True:
                        tok = tok_q.get()
                        if tok is None:
                            break
                        out_ids.append(tok)
                        current = tokenizer.decode(out_ids)
                        sys.stdout.write(current[len(decoded_so_far):])
                        sys.stdout.flush()
                        decoded_so_far = current

                printer = threading.Thread(target=_printer, daemon=True)
                printer.start()

                mx.reset_peak_memory()
                t0 = time.perf_counter()
                for token in token_iter:
                    tok_q.put(token)
                    if token == eos:
                        break
                dt = time.perf_counter() - t0   # excludes terminal drain below

                tok_q.put(None)
                printer.join()
                sys.stdout.write('\n')
                sys.stdout.flush()
            finally:
                kv_peak = kv.peak_bytes
                kv_hits, kv_misses = kv.hits, kv.misses
                kv.close()
                tmp.cleanup()
        finally:
            cache_peak = engine.cache.peak_bytes
            cache_hits, cache_misses = engine.cache.hits, engine.cache.misses
            lookahead_issued = engine.lookahead_issued
            lookahead_skipped_core_missing = engine.lookahead_skipped_core_missing
            engine.close()

        if args.metrics:
            peak_gb = mx.get_peak_memory() / 1e9
            budget_bytes = _resolve_budget(args.budget)
            total_model_gb = sum(
                f.stat().st_size for f in Path(args.packed_dir).glob("*.safetensors")
            ) / 1e9
            print(f"\n--- metrics ---")
            print(f"budget                : {args.budget}  ({budget_bytes/1e9:.2f} GB)")
            print(f"generated tokens      : {len(out_ids)} in {dt:.1f}s  "
                  f"({len(out_ids)/dt:.2f} tok/s, prefetch={'off' if args.no_prefetch else 'on'})")
            print(f"peak unified memory   : {peak_gb:.2f} GB")
            print(f"resident weight peak  : {cache_peak/1e9:.2f} GB  (of {total_model_gb:.2f} GB model)")
            print(f"cache hits / misses   : {cache_hits} / {cache_misses}")
            print(f"kv-budget             : {args.kv_budget}  ({_parse_size(args.kv_budget)/1e9:.2f} GB)")
            print(f"kv resident peak      : {kv_peak/1e9:.2f} GB")
            print(f"kv hits / misses      : {kv_hits} / {kv_misses}")
            if args.lookahead:
                print(f"lookahead issued      : {lookahead_issued}")
                print(f"lookahead skip (core) : {lookahead_skipped_core_missing}")
            if spec_stats is not None:
                print(f"draft model           : {args.draft_model}")
                print(f"draft tokens / sweep  : {args.num_draft_tokens}")
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
        return 0

    if args.command == "serve":
        try:
            run_server(
                args.packed_dir, args.host, args.port,
                model_name=args.model_name,
                budget_bytes=_resolve_budget(args.budget),
                kv_budget=_parse_size(args.kv_budget),
                prefetch=not args.no_prefetch,
                io_threads=args.io_threads,
                warm_window=args.warm_window,
                draft_model_path=args.draft_model,
                num_draft_tokens=args.num_draft_tokens,
                accept_top_k=args.accept_top_k,
                kv_quant=_kv_quant_from_args(args),
                use_prefix_cache=args.use_prefix_cache,
                lookahead_prefetch=args.lookahead,
                wire_limit=args.wire,
                compact_scatter=args.compact_scatter,
                eval_window=args.eval_window,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.command == "bench":
        from .bench import run_bench, system_info, format_report

        model_arg = args.model
        model_path = Path(model_arg)
        default_model = "mlx-community/Qwen3-30B-A3B-4bit"

        # Already a packed dir (local path carrying a manifest.json)?
        if model_path.exists() and (model_path / "manifest.json").exists():
            packed_dir = model_path
        else:
            # Local model dir or HF repo id -- pack it under --packed-root,
            # named after the model id/dir, unless already packed there.
            dir_name = model_path.name if model_path.exists() else model_arg.replace("/", "__")
            packed_dir = Path(args.packed_root) / dir_name
            if (packed_dir / "manifest.json").exists():
                print(f"using existing packed model at {packed_dir}", file=sys.stderr)
            else:
                if model_arg == default_model and not args.yes:
                    resp = input(
                        f"This will download {default_model} (~16 GB) and write a packed "
                        f"copy to {packed_dir} (~16 GB more, ~35 GB total). Continue? [y/N] ")
                    if resp.strip().lower() not in ("y", "yes"):
                        print("aborted", file=sys.stderr)
                        return 1
                print(f"packing {model_arg} -> {packed_dir} ...", file=sys.stderr)
                pack_model(model_arg, packed_dir)

        draft = None if args.no_spec else args.draft
        ngram = args.ngram and not args.no_spec
        if args.quick:
            workloads = ["prose"]
            max_tokens = 50
            draft = None    # --quick is a greedy-only smoke run (matches the help text)
            ngram = False
        else:
            workloads = None
            max_tokens = args.max_tokens

        out_path = Path(args.out) if args.out else None
        results = run_bench(
            packed_dir, draft=draft, draft_tokens=args.draft_tokens,
            budget=_resolve_budget(args.budget), max_tokens=max_tokens,
            workloads=workloads, out=out_path, ngram=ngram,
            ngram_adaptive=args.ngram_adaptive,
            lookahead=args.lookahead,
            wire_limit=args.wire,
        )

        info = system_info()
        report = format_report(info, results, model_arg)
        print("\n----- BEGIN SHAREABLE REPORT -----")
        print(report)
        print("----- END SHAREABLE REPORT -----\n")
        # stdout, not stderr: stderr is unbuffered and jumps ahead of the report
        # when output is piped, putting this line before the run it refers to.
        print("Share your results: paste the block above into a NunSpark GitHub "
              "issue or discussion -- it helps calibrate expectations across machines.")
        return 0

    if args.command == "web":
        try:
            import uvicorn

            from .webapp.app import create_app
        except ModuleNotFoundError as e:
            print(f"Error: the web interface needs the optional 'web' extra "
                  f"({e.name} is missing). Install it with: "
                  f"pip install 'nunspark[web]'  (or: uv run --extra web nunspark web ...)",
                  file=sys.stderr)
            return 1
        app = create_app(packed_root=args.packed_root)
        print(f"nunspark web UI at http://{args.host}:{args.port}", file=sys.stderr)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1
