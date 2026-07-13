from __future__ import annotations

import queue
import threading
import uuid
from typing import Callable

from .engine_pool import EnginePool
from .runner import run_generation
from .schemas import BatchRequest, Job, JobStatus


class JobManager:
    """A FIFO job queue with a single background worker. The engine is
    single-stream/disk-bound, so exactly one job runs at a time. Each job
    streams events to any subscribers (the SSE endpoint) via per-subscriber
    queues; a per-job event log lets a late subscriber replay what it missed."""

    def __init__(self, runner: Callable = run_generation, pool: EnginePool | None = None):
        self._runner = runner
        self._pool = pool or EnginePool()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._files: dict[str, str] = {}          # file_id -> path
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._cancels: set[str] = set()
        self._subs: dict[str, list[queue.Queue]] = {}
        self._log: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._pause = threading.Event()           # set => worker paused
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

    def _run(self) -> None:
        while not self._stop:
            while self._pause.is_set() and not self._stop:
                threading.Event().wait(0.02)
            job_id = self._queue.get()
            if job_id is None:
                break
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job_id in self._cancels:
                job.status = JobStatus.CANCELLED
                self._emit(job_id, {"type": "cancelled", "job": job.public()})
                continue
            self._runner(
                job, pool=self._pool,
                emit=lambda e, jid=job_id: self._emit(jid, e),
                should_cancel=lambda jid=job_id: jid in self._cancels,
            )
