from __future__ import annotations

import json
import tempfile
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from ..archspec import KVQuant
from ..generate import SpecStats, batched_generate, speculative_generate, stream_generate
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


def _fail_job(job: Job, exc: Exception, emit: Callable[[str, dict[str, Any]], None]) -> None:
    """Mark `job` ERROR from a validation exception and report it on its own
    event stream. Mirrors run_generation's except-block behavior (status,
    error message, sidecar) so a job that fails before/outside the batched
    call looks the same to the UI as one that failed inside run_generation."""
    job.status = JobStatus.ERROR
    job.error = f"{type(exc).__name__}: {exc}"

    with suppress(OSError):
        out_dir = Path(job.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = _unique_base(out_dir, job.file_name)
        out_path = out_dir / f"{base}.out.txt"
        out_path.touch()
        job.output_path = str(out_path)
        _write_sidecar(out_dir, job, base)

    emit(job.id, {"type": "error", "job": job.public()})


def _run_one(
    job: Job,
    *,
    pool: EnginePool,
    emit: Callable[[str, dict[str, Any]], None],
    should_cancel: Callable[[str], bool],
) -> None:
    """Adapt the job-id-keyed batch emit/should_cancel to run_generation's
    per-job signature, so the single-job sequential path can be reused
    unchanged both for group-of-1 batches and for the fallback when
    batched_generate itself raises."""
    run_generation(
        job, pool=pool,
        emit=lambda e: emit(job.id, e),
        should_cancel=lambda: should_cancel(job.id),
    )


def run_generation_batch(
    jobs: list[Job],
    *,
    pool: EnginePool,
    emit: Callable[[str, dict[str, Any]], None],
    should_cancel: Callable[[str], bool],
) -> None:
    """Execute a group of param-compatible jobs as ONE batched_generate() call.

    Grouping (which jobs land here, and the MAX_BATCH=8 cap from
    docs/plan5-m3-design.md §16) is the caller's (webapp/jobs.py) job; this
    function only assumes the group shares model/budget/kv_quant/
    use_chat_template/temperature/max_tokens and has no draft model.

    Validation runs in two stages BEFORE any batched forward, so a bad upload
    or an over-cap prompt is isolated onto its own job (ERROR status, its own
    event) and never sinks the rest of the group:
      1. `_read_file_text` (no engine needed) -- catches binary uploads cheaply,
         before paying for a model load, same guard as run_generation.
      2. prompt construction + `_check_prompt_cap` (needs the shared engine's
         tokenizer / geometry).
    If fewer than 2 jobs survive validation, the survivor (if any) runs the
    ordinary sequential path (`run_generation`) -- a group of 1 gets no benefit
    from batching. If `batched_generate` itself raises (e.g. a sliding-window
    arch slipped through the group-compatibility check upstream), the whole
    surviving group falls back to running sequentially rather than failing.
    """
    if not jobs:
        return

    advanced0 = jobs[0].advanced or {}
    budget = resolve_budget(advanced0.get("budget", "auto"))
    kv_quant = _kv_quant(advanced0)

    # Stage 1: upload guard, before any engine load.
    texts: dict[str, str] = {}
    survivors: list[Job] = []
    for job in jobs:
        try:
            texts[job.id] = _read_file_text(Path(job.file_path))
            survivors.append(job)
        except Exception as exc:  # noqa: BLE001 -- isolate, never sink the group
            _fail_job(job, exc, emit)

    if not survivors:
        return
    if len(survivors) == 1:
        _run_one(survivors[0], pool=pool, emit=emit, should_cancel=should_cancel)
        return

    handle = pool.acquire(
        jobs[0].model, jobs[0].draft, budget_bytes=budget, kv_quant=kv_quant,
    )
    engine = handle.engine
    tokenizer = handle.tokenizer

    # Stage 2: prompt build + token cap (needs the shared tokenizer/engine).
    prompts: dict[str, list[int]] = {}
    survivors2: list[Job] = []
    for job in survivors:
        try:
            prompt = build_prompt(
                job.instruction, texts[job.id],
                tokenizer=tokenizer, use_chat_template=job.use_chat_template,
            )
            if isinstance(prompt, str):
                prompt = tokenizer.encode(prompt)
            _check_prompt_cap(
                len(prompt), engine=engine, budget_bytes=budget,
                advanced=job.advanced or {},
            )
            prompts[job.id] = list(prompt)
            survivors2.append(job)
        except Exception as exc:  # noqa: BLE001 -- isolate, never sink the group
            _fail_job(job, exc, emit)

    if not survivors2:
        return
    if len(survivors2) == 1:
        _run_one(survivors2[0], pool=pool, emit=emit, should_cancel=should_cancel)
        return

    # Cancellation cannot interrupt batched_generate mid-decode (it is not a
    # generator -- all rows' tokens arrive together only once the whole lock-
    # step loop finishes, see module docstring note below). The only place a
    # cancel can take effect is before the call: drop already-cancelled jobs
    # here rather than running (and discarding) their share of the batch.
    live = [j for j in survivors2 if not should_cancel(j.id)]
    live_ids = {j.id for j in live}
    for job in survivors2:
        if job.id not in live_ids:
            job.status = JobStatus.CANCELLED
            emit(job.id, {"type": "cancelled", "job": job.public()})

    if not live:
        return
    if len(live) == 1:
        _run_one(live[0], pool=pool, emit=emit, should_cancel=should_cancel)
        return

    for job in live:
        job.status = JobStatus.RUNNING
        emit(job.id, {"type": "started", "job": job.public()})

    try:
        _run_batched(
            live, prompts, engine=engine, tokenizer=tokenizer,
            kv_quant=kv_quant, emit=emit, should_cancel=should_cancel,
        )
    except Exception:  # noqa: BLE001
        # batched_generate raised (e.g. a rotating arch slipped through the
        # group key upstream) -- fall back to the sequential path per job
        # rather than failing everyone in the group. run_generation re-does
        # its own (cheap, cached-engine) validation and emits its own
        # started/done/error events.
        for job in live:
            _run_one(job, pool=pool, emit=emit, should_cancel=should_cancel)


def _run_batched(
    jobs: list[Job],
    prompts: dict[str, list[int]],
    *,
    engine: Any,
    tokenizer: Any,
    kv_quant: KVQuant | None,
    emit: Callable[[str, dict[str, Any]], None],
    should_cancel: Callable[[str], bool],
) -> None:
    """Run ONE batched_generate() call for `jobs` and fan the per-row results
    back to each job's own output file / event stream by row index.

    batched_generate() is not a generator (Plan 5 M3b-1, generate.py): the
    lock-step decode loop returns only after every row has finished. So unlike
    the sequential path's live per-token streaming, a batched job's "token"
    events are emitted in one rapid burst, right after the whole group's
    decode completes, in generation order -- the UI still sees the full text
    appear, just not incrementally while it is being produced. This is a
    direct, honest consequence of the shared weight-sweep design (§7 of
    docs/plan5-m3-design.md): all rows' compute is inseparable until the batch
    is done.
    """
    max_tokens = jobs[0].max_tokens
    temp = jobs[0].temperature
    eos = getattr(tokenizer, "eos_token_id", None)

    t0 = time.perf_counter()
    cache_baseline = _cache_snapshot(engine)

    out_dirs: list[Path] = []
    out_paths: list[Path] = []
    bases: list[str] = []
    for job in jobs:
        out_dir = Path(job.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = _unique_base(out_dir, job.file_name)
        out_path = out_dir / f"{base}.out.txt"
        job.output_path = str(out_path)
        out_dirs.append(out_dir)
        out_paths.append(out_path)
        bases.append(base)

    with ExitStack() as stack:
        out_fhs = [
            stack.enter_context(p.open("w", encoding="utf-8")) for p in out_paths
        ]
        kv_tmp = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="nunspark_web_kv_batch_")
        )
        kv = KVStore(
            kv_tmp,
            budget_bytes=_KV_BUDGET_BYTES,
            prefetch=True,
            cache_kinds=engine.cache_kinds,
            kv_quant=kv_quant,
        )
        stack.callback(kv.close)

        mx.reset_peak_memory()

        prompt_list = [prompts[job.id] for job in jobs]
        # Exactness contract (docs/plan5-m2-mismatch-investigation.md, and the
        # §14/§15 orchestrator amendments in docs/plan5-m3-design.md): each
        # row is target-greedy correct but not guaranteed byte-identical to a
        # solo run() of that prompt -- left-padded rows evaluate RoPE at
        # shifted absolute positions and batched GEMM tiling can differ from
        # single-row GEMM, so rare argmax near-tie flips can diverge from the
        # sequential run (self-healing, documented).
        token_lists = batched_generate(
            engine, prompt_list, max_tokens=max_tokens, temp=temp,
            kv=kv, kv_quant=kv_quant, eos_id=eos,
        )

        for idx, job in enumerate(jobs):
            out_ids: list[int] = []
            for tok in token_lists[idx]:
                if eos is not None and tok == eos:
                    break
                out_ids.append(tok)
                delta = tokenizer.decode([tok])
                if delta:
                    out_fhs[idx].write(delta)
                    out_fhs[idx].flush()
                    emit(job.id, {"type": "token", "job_id": job.id, "text": delta})

            job.tokens_done = len(out_ids)
            metrics = _metrics(out_ids, t0, engine, kv, None, cache_baseline)
            metrics["batched"] = True
            metrics["batch_size"] = len(jobs)
            job.metrics = metrics

            job.status = (
                JobStatus.CANCELLED if should_cancel(job.id) else JobStatus.DONE
            )

            with suppress(OSError):
                _write_sidecar(out_dirs[idx], job, bases[idx])

            event_type = {
                JobStatus.DONE: "done",
                JobStatus.CANCELLED: "cancelled",
            }[job.status]
            emit(job.id, {"type": event_type, "job": job.public()})


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