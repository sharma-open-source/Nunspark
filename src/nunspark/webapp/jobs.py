from __future__ import annotations

import queue
import threading
import uuid
from collections import deque
from typing import Callable

from .engine_pool import EnginePool
from .runner import run_generation, run_generation_batch
from .schemas import BatchRequest, Job, JobStatus

# Plan 5 M3b-3 (docs/plan5-m3-design.md §16): the bit-identical/token-identical
# gate landed at B=8 (2.30x speedup, +8.4% peak memory over B=1), so that is
# the operational default batch size for both the web UI and CLI.
MAX_BATCH = 8


def _compat_key(job: Job) -> tuple | None:
    """The batch-compatibility key for `job`, or None if it can never be
    grouped (opted out, or requires a draft/speculative arm -- batched_generate
    is greedy/temperature-only, no draft model in v1, per
    docs/plan5-m3-design.md §9). Two jobs are batchable together iff their
    keys compare equal: same packed model, no draft, same budget setting, same
    kv-quant setting, same chat-template setting, same temp, same max_tokens.
    Per-row prompt LENGTH is deliberately excluded -- that is exactly the
    left-pad path batched_generate handles. Architecture (sliding-window)
    compatibility is NOT checked here (the engine doesn't exist yet at queue
    time); batched_generate raises for it and the caller falls back to
    sequential for that group."""
    advanced = job.advanced or {}
    if job.draft is not None:
        return None
    if not advanced.get("batch", True):
        return None
    return (
        job.model,
        advanced.get("budget", "auto"),
        advanced.get("kv_bits"),
        advanced.get("kv_group_size", 64),
        job.use_chat_template,
        job.temperature,
        job.max_tokens,
    )


class JobManager:
    """A FIFO job queue with a single background worker. The engine is
    single-stream/disk-bound, so (absent batching) exactly one job runs at a
    time. Each job streams events to any subscribers (the SSE endpoint) via
    per-subscriber queues; a per-job event log lets a late subscriber replay
    what it missed.

    Batch dispatch: when the worker is about to run a job, it opportunistically
    (non-blocking) grabs up to MAX_BATCH-1 more ALREADY-QUEUED jobs that are
    param-compatible (`_compat_key`) with it and hands the whole group to
    `batch_runner` (default `run_generation_batch`) instead of running each one
    through `runner`. This never blocks waiting for more jobs to arrive -- it
    only groups what's sitting in the queue right now, so a lone job is never
    delayed. Batching only ever engages for the production `run_generation`
    runner (checked by identity): a caller supplying a custom single-job runner
    (as the tests do, to record per-job dispatch order) gets the unmodified
    one-job-at-a-time behavior unless it also opts in via `batch_runner`."""

    def __init__(self, runner: Callable = run_generation, pool: EnginePool | None = None,
                 batch_runner: Callable | None = None, start_paused: bool = False):
        self._runner = runner
        self._batch_runner = (
            batch_runner if batch_runner is not None
            else (run_generation_batch if runner is run_generation else None)
        )
        self._pool = pool or EnginePool()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._files: dict[str, str] = {}          # file_id -> path
        self._queue: queue.Queue[str | None] = queue.Queue()
        # Job ids pulled off self._queue while peeking ahead for a batch group
        # but not (yet) consumed -- drained front-first before self._queue, so
        # FIFO order is preserved across peek/defer.
        self._peeked: deque[str] = deque()
        self._cancels: set[str] = set()
        self._subs: dict[str, list[queue.Queue]] = {}
        self._log: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._pause = threading.Event()           # set => worker paused
        # start_paused avoids the inherent race of calling pause() AFTER the
        # worker thread has already started: without it, the worker could
        # reach its first blocking queue.get() before pause() runs, then that
        # get() unblocks on the first submitted job regardless of the pause
        # flag (pause is only checked BEFORE get(), never during a call
        # already in flight) -- exactly the kind of interleaving batch-
        # grouping tests need to rule out deterministically.
        if start_paused:
            self._pause.set()
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -- registration --------------------------------------------------------

    def register_files(self, mapping: dict[str, str]) -> None:
        with self._lock:
            self._files.update(mapping)

    def submit_batch(self, req: BatchRequest) -> list[Job]:
        batch_id = uuid.uuid4().hex
        created: list[Job] = []
        with self._lock:
            for fid in req.file_ids:
                path = self._files.get(fid, fid)  # accept a raw path too
                name = path.rsplit("/", 1)[-1]
                job = Job(
                    id=uuid.uuid4().hex, batch_id=batch_id,
                    file_name=name, file_path=path,
                    model=req.model, draft=req.draft, preset=req.preset,
                    instruction=req.instruction, use_chat_template=req.use_chat_template,
                    max_tokens=req.max_tokens, temperature=req.temperature,
                    output_dir=req.output_dir, advanced=req.advanced.model_dump(),
                )
                self._jobs[job.id] = job
                self._order.append(job.id)
                self._log[job.id] = []
                created.append(job)
        for job in created:
            self._queue.put(job.id)
        return created

    # -- queries -------------------------------------------------------------

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._jobs[jid].public() for jid in self._order]

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    # -- control -------------------------------------------------------------

    def cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancels.add(job_id)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def shutdown(self) -> None:
        self._stop = True
        self._queue.put(None)
        self._worker.join(timeout=10)
        self._pool.close()

    # -- event bus -----------------------------------------------------------

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for ev in self._log.get(job_id, []):   # replay history
                q.put(ev)
            self._subs.setdefault(job_id, []).append(q)
        return q

    def _emit(self, job_id: str, event: dict) -> None:
        with self._lock:
            self._log.setdefault(job_id, []).append(event)
            for q in self._subs.get(job_id, []):
                q.put(event)

    # -- worker --------------------------------------------------------------

    def _next_job_id(self) -> str | None:
        """Blocking pop: the peeked buffer (already pulled off self._queue
        while grouping a previous batch) drains before new arrivals."""
        if self._peeked:
            return self._peeked.popleft()
        return self._queue.get()

    def _peek_job_id(self) -> str | None:
        """Non-blocking pop, for opportunistically grabbing more jobs to group
        into the current batch. Returns None if nothing is available RIGHT
        NOW -- never waits for a job that hasn't arrived yet."""
        if self._peeked:
            return self._peeked.popleft()
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _defer(self, job_id: str) -> None:
        """Push a peeked-but-not-grouped job id back to the front of the
        queue, preserving FIFO order for the next dispatch round."""
        self._peeked.appendleft(job_id)

    def _run(self) -> None:
        while not self._stop:
            while self._pause.is_set() and not self._stop:
                threading.Event().wait(0.02)
            job_id = self._next_job_id()
            if job_id is None:
                break
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job_id in self._cancels:
                job.status = JobStatus.CANCELLED
                self._emit(job_id, {"type": "cancelled", "job": job.public()})
                continue

            group = [job]
            key = _compat_key(job) if self._batch_runner is not None else None
            if key is not None:
                while len(group) < MAX_BATCH:
                    nid = self._peek_job_id()
                    if nid is None:
                        break
                    njob = self._jobs.get(nid)
                    if njob is None:
                        continue
                    if nid in self._cancels:
                        njob.status = JobStatus.CANCELLED
                        self._emit(nid, {"type": "cancelled", "job": njob.public()})
                        continue
                    if _compat_key(njob) != key:
                        self._defer(nid)
                        break
                    group.append(njob)

            if self._batch_runner is not None and len(group) > 1:
                self._batch_runner(
                    group, pool=self._pool,
                    emit=self._emit,
                    should_cancel=lambda jid: jid in self._cancels,
                )
            else:
                for j in group:
                    self._runner(
                        j, pool=self._pool,
                        emit=lambda e, jid=j.id: self._emit(jid, e),
                        should_cancel=lambda jid=j.id: jid in self._cancels,
                    )
