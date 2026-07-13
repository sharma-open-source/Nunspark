from __future__ import annotations

import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import mlx.core as mx


def _piece_bytes(weights: dict) -> int:
    return sum(int(v.nbytes) for v in weights.values())


class PieceCache:
    """Thread-safe, byte-budget MRU (scan-resistant) cache of weight pieces with background prefetch.

    - get(pid): returns the piece's weights, loading on miss. Resident pieces are
      reused across calls (the cross-token hot-set), bounded by a byte budget with
      MRU eviction of unpinned pieces (keeps a stable early-layer prefix resident across the engine's cyclic layer scan).
    - prefetch(pids): hands upcoming pieces to a background worker that loads AND
      evals them (mx.load is lazy, so eval forces the disk read off the compute
      thread), so they are already resident when get() asks for them.

    Loads are de-duplicated via per-piece in-flight events: a piece requested by
    both get() and the worker is loaded exactly once. Eviction never drops the
    piece being loaded (`protect`) or any pinned piece.
    """

    def __init__(
        self,
        loader: Callable[[str], dict],
        budget_bytes: int,
        pinned: Iterable[str] = (),
        io_threads: int = 1,
        pather: Callable[[str], Path] | None = None,
    ):
        self._loader = loader
        self._budget = int(budget_bytes)
        self._pinned = set(pinned)
        self._resident: "OrderedDict[str, dict]" = OrderedDict()
        self._bytes = 0
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        # Optional parallel page-cache warmer. When active, prefetch() raw-reads a
        # piece's file off the (serialized) mlx load path so the single materialize
        # worker below hits warm pages. io_threads<=1 or no pather => no pool, and
        # behavior is behavior-identical to the pre-warmer path.
        self._pather = pather
        self._warm_pool = (
            ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="nunspark-warm")
            if io_threads > 1 and pather is not None
            else None
        )
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---- public API ----
    def get(self, pid: str) -> dict:
        with self._lock:
            if pid in self._resident:
                self._resident.move_to_end(pid)
                self.hits += 1
                return self._resident[pid]
            ev = self._inflight.get(pid)
            if ev is None:
                ev = threading.Event()
                self._inflight[pid] = ev   # reserve under lock; we own the load
                self.misses += 1
                owner = True
            else:
                owner = False
        if not owner:
            ev.wait()
            with self._lock:
                if pid in self._resident:
                    self._resident.move_to_end(pid)
                    self.hits += 1
                    return self._resident[pid]
            # The owning load failed and cleaned up its in-flight entry; retry
            # from scratch (we become the owner and surface any error to caller).
            return self.get(pid)
        return self._materialize(pid, ev)

    def prefetch(self, pids: Iterable[str]) -> None:
        with self._lock:
            for pid in pids:
                if pid in self._resident or pid in self._inflight:
                    continue
                self._inflight[pid] = threading.Event()  # reserve so get() waits
                self.misses += 1  # every scheduled prefetch is a disk-load miss
                if self._warm_pool is None:
                    self._queue.put(pid)
                else:
                    self._warm_pool.submit(self._warm_then_queue, pid)

    def _warm_then_queue(self, pid: str) -> None:
        # Warm the OS page cache for this piece's file (parallel, off the mlx load
        # path), then hand the pid to the single materialize worker, whose
        # mx.load+mx.eval now reads warm pages instead of doing a cold disk read.
        try:
            self._warm(pid)
        except Exception:
            pass  # best-effort: a failed warm just means the materialize reads cold
        self._queue.put(pid)

    def _warm(self, pid: str) -> None:
        path = self._pather(pid)
        fd = os.open(os.fspath(path), os.O_RDONLY)
        try:
            while os.read(fd, 1 << 23):  # 8 MiB chunks; populate the page cache
                pass
        finally:
            os.close(fd)

    def close(self) -> None:
        # Swap the pool out under the lock so a concurrent prefetch() either sees a
        # live pool (and submits) or None (and uses the queue) — never submits to a
        # pool being shut down. shutdown() blocks, so it runs OUTSIDE the lock.
        # Drains in-flight warms (each then queues its materialize), then stops the
        # materialize worker with the sentinel. Safe to call twice.
        with self._lock:
            pool, self._warm_pool = self._warm_pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        self._queue.put(None)
        self._worker.join()

    @property
    def resident_ids(self) -> list[str]:
        with self._lock:
            return list(self._resident)

    # ---- internals ----
    def _run(self) -> None:
        while True:
            pid = self._queue.get()
            if pid is None:
                return
            with self._lock:
                if pid in self._resident:
                    ev = self._inflight.pop(pid, None)
                    if ev:
                        ev.set()
                    continue
                ev = self._inflight.get(pid)
                if ev is None:
                    continue  # already handled by a get() owner
            try:
                self._materialize(pid, ev)
            except Exception:
                # A failed prefetch must not kill the worker; the eventual
                # get() for this pid will retry and surface the error.
                pass

    def _materialize(self, pid: str, ev: threading.Event) -> dict:
        try:
            weights = self._loader(pid)
            mx.eval(list(weights.values()))   # force the lazy mmap read here
            with self._lock:
                self._resident[pid] = weights
                self._bytes += _piece_bytes(weights)
                self.peak_bytes = max(self.peak_bytes, self._bytes)  # transient peak
                self._evict_locked(protect=pid)
            return weights
        finally:
            # Always release the reservation and wake waiters, even on failure,
            # so a loader/eval error can never permanently strand a get() waiter.
            with self._lock:
                self._inflight.pop(pid, None)
            ev.set()

    def _evict_locked(self, protect: str) -> None:
        # Scan-resistant (MRU) eviction: the engine scans layers 0..N-1 in the
        # SAME order every token, so the just-used piece is the one needed again
        # soonest. Evicting from the most-recently-used end (iterate reversed)
        # keeps a stable early-layer prefix resident across tokens; LRU would
        # keep the trailing layers and re-read everything next token.
        # NOTE: tuned for that dense cyclic scan. MoE selective-expert pieces have
        # a sparse, non-cyclic access profile and are not considered here.
        evicted = False
        for pid in reversed(list(self._resident)):
            if self._bytes <= self._budget:
                break
            if pid == protect or pid in self._pinned:
                continue
            w = self._resident.pop(pid)
            self._bytes -= _piece_bytes(w)
            evicted = True
        if evicted:
            mx.clear_cache()  # only reclaim Metal buffers when we actually freed some
