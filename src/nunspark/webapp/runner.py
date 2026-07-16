from __future__ import annotations

import json
import tempfile
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from ..archspec import KVQuant
from ..generate import SpecStats, speculative_generate, stream_generate
from ..kv_store import KVStore
from ..sysmem import resolve_budget
from ..sysmem import unified_ram_bytes as _unified_ram_bytes
from .engine_pool import EnginePool
from .schemas import Job, JobStatus, preset_params

_METRICS_EVERY = 8
_KV_BUDGET_BYTES = 1_000_000_000_000

# --- Input guards -----------------------------------------------------------
# A binary upload (e.g. a PDF) decoded with errors="replace" becomes millions
# of garbage characters; tokenized, their KV cache alone (~98 KB/token on
# Qwen3-30B) can exceed machine RAM and swap-storm the whole Mac. So: reject
# binary files outright, and cap prompt tokens against unified memory.

# PDF is called out separately because it is the common accidental upload.
_PDF_MAGIC = b"%PDF"
_BINARY_MAGICS: tuple[bytes, ...] = (
    b"PK\x03\x04",  # zip (also docx/xlsx/pptx)
    b"\x89PNG",
    b"\xff\xd8",    # jpeg
    b"GIF8",
    b"\x7fELF",
)
_NUL_SCAN_BYTES = 8 * 1024
# Real text has essentially zero U+FFFD; 5% means the bytes are not UTF-8.
_REPLACEMENT_RATIO_MAX = 0.05

_PROMPT_CAP_MIN = 4096
_PROMPT_CAP_MAX = 65536
# Used when RAM or model geometry can't be determined -- conservative but
# enough for typical documents.
_PROMPT_CAP_FALLBACK = 8192
# Left out of the KV budget for the OS, activations, and everything else.
_PROMPT_CAP_HEADROOM_BYTES = 4 * 1024**3


def _read_file_text(path: Path) -> str:
    """Read an upload as text, rejecting binary files loudly.

    Reads bytes first so binary content is caught before errors="replace"
    can turn it into garbage that looks like a (huge) valid prompt.
    """
    raw = path.read_bytes()

    if raw.startswith(_PDF_MAGIC):
        raise ValueError(
            "PDF files are not supported — extract the text first "
            "and upload it as .txt/.md"
        )

    if raw.startswith(_BINARY_MAGICS):
        raise ValueError(
            "binary file detected — upload plain text (.txt/.md) instead"
        )

    if b"\x00" in raw[:_NUL_SCAN_BYTES]:
        raise ValueError(
            "binary file detected (NUL bytes) — upload plain text "
            "(.txt/.md) instead"
        )

    text = raw.decode("utf-8", errors="replace")

    if text and text.count("�") / len(text) > _REPLACEMENT_RATIO_MAX:
        raise ValueError(
            "file does not decode as UTF-8 text (>5% invalid characters) "
            "— upload plain text (.txt/.md) instead"
        )

    return text


def _arg_int(args: Any, *names: str) -> int | None:
    """First integer attribute (or dict key) among `names`, else None.

    The isinstance check guards against Mock objects and non-numeric config
    values sneaking into the KV-size arithmetic.
    """
    for name in names:
        value = getattr(args, name, None)
        if value is None and isinstance(args, dict):
            value = args.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _per_token_kv_bytes(engine: Any) -> int | None:
    """Estimated fp16 KV-cache bytes per prompt token, or None if the model
    geometry can't be read off the engine's args."""
    args = getattr(engine, "args", None)
    if args is None:
        return None

    layers = _arg_int(args, "num_hidden_layers")
    heads = _arg_int(args, "num_key_value_heads", "num_attention_heads")
    head_dim = _arg_int(args, "head_dim")

    if head_dim is None:
        hidden = _arg_int(args, "hidden_size")
        attn_heads = _arg_int(args, "num_attention_heads")
        if hidden is not None and attn_heads:
            head_dim = hidden // attn_heads

    if layers is None or heads is None or head_dim is None:
        return None

    # K and V, 2 bytes each (fp16) -- ignores kv_bits quantization on
    # purpose: the cap should hold even for the unquantized worst case.
    return layers * heads * head_dim * 2 * 2


def _check_prompt_cap(
    prompt_len: int,
    *,
    engine: Any,
    budget_bytes: int,
    advanced: dict[str, Any],
) -> None:
    """Fail loudly (never truncate) when the prompt's KV cache can't fit.

    `advanced.max_prompt_tokens` overrides the derived cap entirely, so a
    user who understands the swap risk is never blocked.
    """
    override = advanced.get("max_prompt_tokens")
    per_token = _per_token_kv_bytes(engine)
    ram = _unified_ram_bytes()

    if override is not None:
        cap = int(override)
    elif per_token is None or ram is None:
        cap = _PROMPT_CAP_FALLBACK
    else:
        allowed = (ram - budget_bytes - _PROMPT_CAP_HEADROOM_BYTES) // per_token
        cap = max(_PROMPT_CAP_MIN, min(_PROMPT_CAP_MAX, int(allowed)))

    if prompt_len <= cap:
        return

    message = f"prompt is {prompt_len} tokens, over the cap of {cap}"

    if per_token is not None and ram is not None:
        message += (
            f": its KV cache alone would need "
            f"~{prompt_len * per_token / 1e9:.1f} GB "
            f"against {ram / 1e9:.0f} GB unified RAM"
        )

    message += "; set advanced.max_prompt_tokens to override"

    raise ValueError(message)


def build_prompt(
    instruction: str,
    file_text: str,
    *,
    tokenizer: Any,
    use_chat_template: bool,
) -> str | list[int]:
    """Build a prompt for generation.

    Returns token IDs when using a chat template, otherwise returns raw text.
    """
    combined = (
        f"{instruction.strip()}\n\n{file_text}"
        if instruction.strip()
        else file_text
    )

    if not use_chat_template or tokenizer is None:
        return combined

    with suppress(AttributeError, TypeError, ValueError):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": combined}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )

    return combined


def _kv_quant(advanced: dict[str, Any]) -> KVQuant | None:
    bits = advanced.get("kv_bits")
    if bits is None:
        return None

    return KVQuant(
        bits=bits,
        group_size=advanced.get("kv_group_size", 64),
    )


def _parse_size(size: str) -> int:
    from ..cli import _parse_size as parse_size

    return parse_size(size)


def _unique_base(out_dir: Path, file_name: str) -> str:
    """Generate a unique output basename."""
    candidate = file_name
    index = 1

    while (out_dir / f"{candidate}.out.txt").exists():
        candidate = f"{file_name}.{index}"
        index += 1

    return candidate


def run_generation(
    job: Job,
    *,
    pool: EnginePool,
    emit: Callable[[dict[str, Any]], None],
    should_cancel: Callable[[], bool],
) -> None:
    """Execute a generation job."""

    job.status = JobStatus.RUNNING
    emit({"type": "started", "job": job.public()})

    advanced = job.advanced or {}

    budget = resolve_budget(advanced.get("budget", "auto"))
    kv_quant = _kv_quant(advanced)

    accept_top_k, num_draft = preset_params(
        job.preset,
        advanced.get("accept_top_k"),
        advanced.get("num_draft_tokens"),
    )

    out_dir = Path(job.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _unique_base(out_dir, job.file_name)
    out_path = out_dir / f"{base}.out.txt"

    job.output_path = str(out_path)

    t0 = time.perf_counter()

    try:
        with ExitStack() as stack:
            out_fh = stack.enter_context(
                out_path.open("w", encoding="utf-8")
            )

            kv_tmp = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="nunspark_web_kv_")
            )

            # Before acquiring an engine: a rejected upload should not cost
            # a (potentially multi-GB) model load.
            file_text = _read_file_text(Path(job.file_path))

            handle = pool.acquire(
                job.model,
                job.draft,
                budget_bytes=budget,
                kv_quant=kv_quant,
            )

            engine = handle.engine
            tokenizer = handle.tokenizer
            draft_model = handle.draft_model

            prompt = build_prompt(
                job.instruction,
                file_text,
                tokenizer=tokenizer,
                use_chat_template=job.use_chat_template,
            )

            if isinstance(prompt, str):
                prompt = tokenizer.encode(prompt)

            _check_prompt_cap(
                len(prompt),
                engine=engine,
                budget_bytes=budget,
                advanced=advanced,
            )

            eos = getattr(tokenizer, "eos_token_id", None)

            cache_baseline = _cache_snapshot(engine)

            kv = KVStore(
                kv_tmp,
                budget_bytes=_KV_BUDGET_BYTES,
                prefetch=True,
                cache_kinds=engine.cache_kinds,
                kv_quant=kv_quant,
            )
            stack.callback(kv.close)

            if draft_model is not None:
                spec_stats = SpecStats()

                tokens = speculative_generate(
                    engine,
                    draft_model,
                    list(prompt),
                    max_tokens=job.max_tokens,
                    num_draft_tokens=num_draft,
                    accept_top_k=accept_top_k,
                    kv=kv,
                    kv_quant=kv_quant,
                    eos_id=eos,
                    stats=spec_stats,
                )
            else:
                spec_stats = None

                tokens = stream_generate(
                    engine,
                    list(prompt),
                    max_tokens=job.max_tokens,
                    temp=job.temperature,
                    kv=kv,
                    kv_quant=kv_quant,
                )

            stack.callback(lambda: getattr(tokens, "close", lambda: None)())

            mx.reset_peak_memory()

            out_ids: list[int] = []
            cancelled = False

            for tok in tokens:
                if should_cancel():
                    cancelled = True
                    break

                if eos is not None and tok == eos:
                    break

                out_ids.append(tok)
                job.tokens_done = len(out_ids)

                # Incremental decoding if supported.
                delta = tokenizer.decode([tok])

                if delta:
                    out_fh.write(delta)
                    out_fh.flush()

                    emit(
                        {
                            "type": "token",
                            "job_id": job.id,
                            "text": delta,
                        }
                    )

                if len(out_ids) % _METRICS_EVERY == 0:
                    emit(
                        {
                            "type": "metrics",
                            "job_id": job.id,
                            "metrics": _metrics(
                                out_ids,
                                t0,
                                engine,
                                kv,
                                spec_stats,
                                cache_baseline,
                            ),
                        }
                    )

            job.metrics = _metrics(
                out_ids,
                t0,
                engine,
                kv,
                spec_stats,
                cache_baseline,
            )

            job.status = (
                JobStatus.CANCELLED
                if cancelled
                else JobStatus.DONE
            )

    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = f"{type(exc).__name__}: {exc}"

    with suppress(OSError):
        _write_sidecar(out_dir, job, base)

    event_type = {
        JobStatus.DONE: "done",
        JobStatus.CANCELLED: "cancelled",
        JobStatus.ERROR: "error",
    }[job.status]

    emit({"type": event_type, "job": job.public()})


def _cache_snapshot(engine: Any) -> dict[str, dict[str, int]]:
    """Point-in-time copy of the piece cache counters, used as a baseline so
    `_metrics` can report deltas for the current job on a cache/engine that
    is reused (and thus already warm) across jobs."""
    stats = engine.cache.stats()

    return {
        "hits": dict(stats["hits"]),
        "misses": dict(stats["misses"]),
        "bytes_loaded": dict(stats["bytes_loaded"]),
    }


def _metrics(
    out_ids: list[int],
    t0: float,
    engine: Any,
    kv: KVStore | None,
    spec_stats: SpecStats | None,
    cache_baseline: dict[str, dict[str, int]] | None = None,
) -> dict[str, float]:
    elapsed = max(time.perf_counter() - t0, 1e-9)

    metrics: dict[str, float] = {
        "tokens": len(out_ids),
        "tok_per_s": len(out_ids) / elapsed,
        "peak_mem_gb": mx.get_peak_memory() / 1e9,
        "weight_peak_gb": engine.cache.peak_bytes / 1e9,
        "kv_peak_gb": (kv.peak_bytes / 1e9) if kv else 0.0,
    }

    if spec_stats is not None:
        metrics["multiplier"] = spec_stats.multiplier
        metrics["deviation_rate"] = spec_stats.deviation_rate

    if cache_baseline is not None:
        stats = engine.cache.stats()

        eh = stats["hits"].get("expert", 0) - cache_baseline["hits"].get("expert", 0)
        em = stats["misses"].get("expert", 0) - cache_baseline["misses"].get("expert", 0)

        if eh + em > 0:
            metrics["expert_hit_pct"] = 100.0 * eh / (eh + em)

            bytes_loaded_delta = sum(
                stats["bytes_loaded"].get(kind, 0)
                - cache_baseline["bytes_loaded"].get(kind, 0)
                for kind in stats["bytes_loaded"]
            )

            if out_ids:
                metrics["mb_per_token"] = (
                    bytes_loaded_delta / len(out_ids) / 1e6
                )

    return metrics


def _write_sidecar(
    out_dir: Path,
    job: Job,
    base: str,
) -> None:
    sidecar = out_dir / f"{base}.meta.json"

    payload = {
        "file_name": job.file_name,
        "model": job.model,
        "draft": job.draft,
        "preset": job.preset,
        "instruction": job.instruction,
        "max_tokens": job.max_tokens,
        "temperature": job.temperature,
        "status": job.status.value,
        "error": job.error,
        "metrics": job.metrics,
    }

    sidecar.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )