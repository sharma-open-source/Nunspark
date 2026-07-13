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
from .engine_pool import EnginePool
from .schemas import Job, JobStatus, preset_params

_METRICS_EVERY = 8
_KV_BUDGET_BYTES = 1_000_000_000_000


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

    budget = _parse_size(advanced.get("budget", "4GB"))
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

            handle = pool.acquire(
                job.model,
                job.draft,
                budget_bytes=budget,
                kv_quant=kv_quant,
            )

            # Replace with stack.callback(...) if EnginePool
            # does not provide a context manager.
            stack.callback(pool.release, handle)

            engine = handle.engine
            tokenizer = handle.tokenizer
            draft_model = handle.draft_model

            file_text = Path(job.file_path).read_text(
                encoding="utf-8",
                errors="replace",
            )

            prompt = build_prompt(
                job.instruction,
                file_text,
                tokenizer=tokenizer,
                use_chat_template=job.use_chat_template,
            )

            if isinstance(prompt, str):
                prompt = tokenizer.encode(prompt)

            eos = getattr(tokenizer, "eos_token_id", None)

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
                            ),
                        }
                    )

            job.metrics = _metrics(
                out_ids,
                t0,
                engine,
                kv,
                spec_stats,
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


def _metrics(
    out_ids: list[int],
    t0: float,
    engine: Any,
    kv: KVStore | None,
    spec_stats: SpecStats | None,
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