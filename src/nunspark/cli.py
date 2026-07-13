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
from .generate import generate as run_generate
from .server import run_server

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
    p_gen.add_argument("--budget", default="4GB",
                       help="resident weight budget, e.g. 512MB, 4GB")
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
    p_gen.add_argument("--num-draft-tokens", type=int, default=16,
                       help="Number of draft tokens to propose per iteration (default: 16)")
    p_gen.add_argument("--accept-top-k", type=int, default=1,
                       help="Acceptance threshold: 1=lossless, >1=fast mode (default: 1)")
    p_gen.add_argument("--no-chat-template", action="store_true",
                       help="skip chat template; encode prompt as-is (default: apply if available)")
    p_gen.add_argument("--metrics", action="store_true",
                       help="print tok/s, peak memory, cache and KV stats after generation")

    p_serve = sub.add_parser("serve", help="serve a packed model behind an OpenAI v1-compatible API")
    p_serve.add_argument("packed_dir")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--model-name", default=None,
                         help="model id reported to clients (defaults to the packed dir's name)")
    p_serve.add_argument("--budget", default="4GB",
                         help="resident weight budget, e.g. 512MB, 4GB")
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

    p_web = sub.add_parser("web", help="launch the local web interface")
    p_web.add_argument("--packed-root", default="models",
                       help="directory scanned for packed models (default: ./models)")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)

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
        manifest = Manifest.load(Path(args.packed_dir) / "manifest.json")
        engine = StreamingEngine(
            args.packed_dir, manifest,
            budget_bytes=_parse_size(args.budget),
            prefetch=not args.no_prefetch,
            io_threads=args.io_threads,
            warm_window=args.warm_window,
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
                use_spec = draft_model is not None or eagle_drafter is not None
                spec_stats = SpecStats() if use_spec else None

                if eagle_drafter is not None:
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
                    )
                else:
                    from .generate import stream_generate
                    token_iter = stream_generate(
                        engine, prompt,
                        max_tokens=args.max_tokens, temp=args.temp,
                        kv=kv,
                        kv_quant=kv_quant,
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
            engine.close()

        if args.metrics:
            peak_gb = mx.get_peak_memory() / 1e9
            budget_bytes = _parse_size(args.budget)
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
                budget_bytes=_parse_size(args.budget),
                kv_budget=_parse_size(args.kv_budget),
                prefetch=not args.no_prefetch,
                io_threads=args.io_threads,
                warm_window=args.warm_window,
                draft_model_path=args.draft_model,
                num_draft_tokens=args.num_draft_tokens,
                accept_top_k=args.accept_top_k,
                kv_quant=_kv_quant_from_args(args),
                use_prefix_cache=args.use_prefix_cache,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
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
