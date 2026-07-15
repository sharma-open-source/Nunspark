from __future__ import annotations

import itertools
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


def _piece_class(pid: str) -> str:
    """Classify a piece id for the per-class counters (M1 locality measurement):
    'expert' for a per-expert MoE piece, 'core' for a selective-MoE layer's
    non-expert piece, 'dense' for everything else (whole dense layers, embed,
    norm/head, whole-layer legacy MoE packs)."""
    if "_expert_" in pid:
        return "expert"
    if pid.endswith("_core"):
        return "core"
    return "dense"


class PieceCache:
    """Thread-safe, byte-budget, two-region cache of weight pieces with background prefetch.

    - get(pid): returns the piece's weights, loading on miss. Resident pieces are
      reused across calls (the cross-token hot-set), bounded by a byte budget.
    - prefetch(pids): hands upcoming pieces to a background worker that loads AND
      evals them (mx.load is lazy, so eval forces the disk read off the compute
      thread), so they are already resident when get() asks for them.

    Two eviction regions (plan4 M2), split by _piece_class:
    - main (dense + core pieces): MRU eviction, exactly as before — the engine
      scans layers 0..N-1 in the same order every token, so evicting the
      most-recently-used piece keeps a stable early-layer prefix resident
      (scan resistance; LRU would thrash the cyclic scan).
    - expert (pids containing "_expert_"): plain LRU eviction — expert access is
      sparse and skewed, where MRU is actively wrong (M1: 57% live hit under the
      shared MRU pool vs 84–94% for LRU in the offline sim at equal capacity;
      LFU-with-decay matched LRU within 0.3%, so no frequency machinery).

    One total byte budget: the expert region is capped at `expert_frac * budget`;
    the main region gets the remainder. The split activates on the first expert
    insert (sticky), so dense-only models keep the full budget and behave exactly
    like the old single-region cache. A region never evicts the other region's
    pieces.

    Loads are de-duplicated via per-piece in-flight events: a piece requested by
    both get() and the worker is loaded exactly once. Eviction never drops the
    piece being loaded (`protect`) or any pinned piece.

    Two prefetch tiers (plan4 M3b): the background worker drains a PriorityQueue
    that always serves demand-critical prefetches (cores, dense layers — the pids
    the cyclic scan needs unconditionally, enqueued by prefetch(speculative=False))
    before speculative expert prefetches (prefetch(speculative=True), the M3a
    temporal expert guesses that only fill I/O slack). A speculative load never
    blocks a demand get(): demand misses materialize on the *calling* thread, not
    via the worker, so an in-flight speculative load on the worker thread can only
    lag the next demand *prefetch*, never a demand get(). If a demanded pid IS the
    speculative one already in flight, get() waits on that same load (dedup) — a
    win, not a stall. Counters (speculative_issued / speculative_used /
    speculative_wasted_bytes) expose the prefetch's hit-rate and wasted bandwidth
    via stats().

    Speculative expert loads go to a separate STAGING buffer (v3), never straight
    into the expert LRU: a wrong or oversized batch of temporal guesses can then
    never evict the demand working set. (One K-token verify pass can touch more
    expert bytes than the whole expert region, so any in-LRU speculative insertion
    is a knife-edge — hot-insert thrashes a large working set, cold-insert starves
    a tight one, and epoch "protection" evicts the demand set first; staging sits
    outside the eviction path entirely, sidestepping all three.) Staging has its
    own byte cap (`spec_staging_bytes`, default 20% of the expert-region budget)
    and shares the expert region's budget dynamically — expert LRU + staging
    together stay within the configured expert share (occupied staging squeezes
    the LRU; empty staging returns the full region to it), so total process bytes
    stay within budget_bytes; peak_bytes includes staging.
    A staged piece survives one pass of grace (its insertion pass plus the next,
    via the begin_pass() epoch counter); an unconsumed staged piece is expired at
    begin_pass() and its bytes counted wasted. If staging is full, the oldest
    staged entries are dropped (and counted wasted) to make room; a piece larger
    than the whole cap is never stored (also counted wasted). A demand get() that
    finds a staged piece counts it "used" once, removes it from staging, and
    inserts it into the expert LRU as a normal hot demand piece. A demand get()
    that instead joins a still-in-flight speculative load (via the dedup Event)
    counts "used" at join time and the load then lands in the expert LRU as a
    plain demand piece (never staged). wasted_bytes = speculative bytes never
    demand-touched (dropped from staging by expiry or cap pressure, or too large
    to stage at all).
    """

    def __init__(
        self,
        loader: Callable[[str], dict],
        budget_bytes: int,
        pinned: Iterable[str] = (),
        io_threads: int = 1,
        pather: Callable[[str], Path] | None = None,
        expert_frac: float = 0.9,
        spec_staging_bytes: int | None = None,
    ):
        self._loader = loader
        self._budget = int(budget_bytes)
        self._expert_budget = int(self._budget * float(expert_frac))
        self._pinned = set(pinned)
        # main region: dense + core pieces, MRU semantics (last = most recent).
        self._resident: "OrderedDict[str, dict]" = OrderedDict()
        # expert region: LRU semantics (first = least recent, evicted first).
        self._experts: "OrderedDict[str, dict]" = OrderedDict()
        self._bytes = 0            # total resident bytes across both regions
        self._main_bytes = 0
        self._expert_bytes = 0
        # Sticky: flips on the first expert insert. Until then the main region
        # gets the WHOLE budget, so dense-only models see no behavior change.
        self._split_active = False
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        # Per-class counters (M1 locality measurement): "expert" / "core" / "dense",
        # classified from the pid by _piece_class. hits/misses above stay the sums
        # of these — no API break for existing callers.
        self._hits_by_class = {"dense": 0, "core": 0, "expert": 0}
        self._misses_by_class = {"dense": 0, "core": 0, "expert": 0}
        self._bytes_by_class = {"dense": 0, "core": 0, "expert": 0}
        # Speculative-prefetch (plan4 M3a/M3b, v3 staging) counters + tracking state.
        # _spec_pending: pids reserved by a speculative prefetch, not yet
        # materialized (a demand get() joining one of these counts "used" at join
        # time and converts the load to a plain demand load). _staging: pid ->
        # (weights, nbytes, insertion_epoch) for speculative loads that finished
        # materializing but no get() has demanded yet — a later get() promotes it
        # into the expert LRU ("used"); expiry or cap pressure drops it ("wasted").
        # _epoch advances via begin_pass() (one call per engine forward pass) and
        # drives the one-pass staging grace.
        self.speculative_issued = 0
        self.speculative_used = 0
        self.speculative_wasted_bytes = 0
        self._spec_pending: set[str] = set()
        self._epoch = 0
        # Speculative staging buffer (v3): speculative expert loads land HERE, never
        # in the expert LRU, so a wrong/oversized batch of guesses can never evict
        # the demand working set. Own byte cap (default 20% of the expert share),
        # tracked separately from the two resident regions (self._bytes) but carved
        # out of the expert budget below. peak_bytes includes staging.
        self._staging: "OrderedDict[str, tuple[dict, int, int]]" = OrderedDict()
        self._staging_bytes = 0
        self._staging_cap = (
            int(spec_staging_bytes) if spec_staging_bytes is not None
            else int(self._expert_budget * 0.2)
        )
        # Staging SHARES the expert budget dynamically (see _evict_locked): expert
        # LRU + staging together never exceed the expert share, so total process
        # bytes stay within budget_bytes — but when staging is empty (prefetch off,
        # greedy decode) the LRU gets the full region back. (Measured: an
        # added-on-top cap pushed a 16 GB machine into memory pressure — peak
        # 11.6 GB — and the paging cost dwarfed every expert-cache win.)
        # Two-tier prefetch queue: (tier, seq, pid). tier 0 = demand-critical,
        # 1 = speculative, 3 = shutdown sentinel (drains after both). seq is a
        # monotonic tiebreaker so the sentinel/pids never compare (None vs str).
        self._pq: "queue.PriorityQueue[tuple[int, int, str | None]]" = queue.PriorityQueue()
        self._seq = itertools.count()
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
        # Separate pool for warm_bulk() (the demand-critical prefill/verify bulk
        # warm), created lazily on first use so single-token decode never pays for
        # it. Kept distinct from _warm_pool: that one belongs to the speculative
        # prefetch path (only when io_threads>1); warm_bulk is a pure page-cache
        # populate that never touches _inflight/queue/staging accounting.
        self._bulk_warm_pool: ThreadPoolExecutor | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _region(self, pid: str) -> "OrderedDict[str, dict]":
        """The resident table a pid lives in (call with self._lock held)."""
        return self._experts if _piece_class(pid) == "expert" else self._resident

    # ---- public API ----
    def begin_pass(self) -> None:
        """Advance the speculative-staging epoch and expire stale staged pieces.
        The engine calls this once at the start of every forward pass; a staged
        speculative piece inserted at epoch E survives while the epoch is <= E+1
        (its insertion pass plus exactly one following pass — the pass its temporal
        prediction is for). Once the epoch advances past that window an unconsumed
        staged piece is dropped and its bytes counted wasted."""
        with self._lock:
            self._epoch += 1
            for pid in [p for p, (_w, _n, e) in self._staging.items()
                        if e < self._epoch - 1]:
                _w, n, _e = self._staging.pop(pid)
                self._staging_bytes -= n
                self.speculative_wasted_bytes += n  # never demanded within its window

    def get(self, pid: str) -> dict:
        with self._lock:
            tbl = self._region(pid)
            if pid in tbl:
                # main: MRU bookkeeping; expert: marks most-recently-used for LRU.
                tbl.move_to_end(pid)
                self.hits += 1
                self._hits_by_class[_piece_class(pid)] += 1
                return tbl[pid]
            if pid in self._staging:
                # Speculative staging hit: a speculative prefetch already
                # materialized this piece. Count it used once, promote it into the
                # expert LRU as a normal hot demand piece, then evict under the cap.
                return self._promote_staged(pid)
            ev = self._inflight.get(pid)
            if ev is None:
                ev = threading.Event()
                self._inflight[pid] = ev   # reserve under lock; we own the load
                self.misses += 1
                self._misses_by_class[_piece_class(pid)] += 1
                owner = True
            else:
                owner = False
                if pid in self._spec_pending:
                    # A demand get() joining a still-in-flight speculative load:
                    # the guess WAS right (the demand arrived before the load even
                    # finished), so count it used now and convert the load to a
                    # plain demand load — it lands hot and is never tracked as
                    # speculative-resident (so it can't be double-counted used or
                    # later counted wasted).
                    self.speculative_used += 1
                    self._spec_pending.discard(pid)
        if not owner:
            ev.wait()
            with self._lock:
                tbl = self._region(pid)
                if pid in tbl:
                    tbl.move_to_end(pid)
                    self.hits += 1
                    self._hits_by_class[_piece_class(pid)] += 1
                    return tbl[pid]
            # The owning load failed and cleaned up its in-flight entry; retry
            # from scratch (we become the owner and surface any error to caller).
            return self.get(pid)
        return self._materialize(pid, ev)

    def _promote_staged(self, pid: str) -> dict:
        """A demand get() found `pid` in the speculative staging buffer: count it
        used once, move it out of staging into the expert LRU as a normal hot
        demand piece, evict under the region cap, and return it (call with
        self._lock held). The disk-materialized bytes were already counted at
        _materialize time, so only the region occupancy is updated here."""
        weights, nbytes, _epoch = self._staging.pop(pid)
        self._staging_bytes -= nbytes
        self.speculative_used += 1
        self._experts[pid] = weights   # hot end (append == most recent)
        self._expert_bytes += nbytes
        self._bytes += nbytes
        self._split_active = True
        self.hits += 1
        self._hits_by_class["expert"] += 1
        self.peak_bytes = max(self.peak_bytes, self._bytes + self._staging_bytes)
        self._evict_locked(protect=pid)
        return weights

    def _stage_locked(self, pid: str, weights: dict, nbytes: int) -> None:
        """Insert a freshly-materialized speculative expert into the staging buffer
        under its own byte cap (call with self._lock held). A piece larger than the
        whole cap is never stored (counted wasted); otherwise the oldest staged
        entries (oldest insertion == most likely already expired) are dropped and
        counted wasted until the new piece fits."""
        if nbytes > self._staging_cap:
            self.speculative_wasted_bytes += nbytes  # never stored -> pure waste
            return
        while self._staging and self._staging_bytes + nbytes > self._staging_cap:
            _old_pid, (_w, old_n, _e) = self._staging.popitem(last=False)  # oldest
            self._staging_bytes -= old_n
            self.speculative_wasted_bytes += old_n
        self._staging[pid] = (weights, nbytes, self._epoch)
        self._staging_bytes += nbytes

    def prefetch(self, pids: Iterable[str], speculative: bool = False) -> None:
        """Enqueue background loads. `speculative=True` marks temporal expert
        guesses (plan4 M3a): they load in the lower-priority tier and only fill
        I/O slack behind demand-critical (core/dense) prefetches. Dedup against
        resident/in-flight pieces is identical for both tiers — a speculative pid
        already demanded (or vice versa) is not re-issued, nor is one already
        held in the speculative staging buffer."""
        tier = 1 if speculative else 0
        with self._lock:
            for pid in pids:
                if (pid in self._region(pid) or pid in self._inflight
                        or pid in self._staging):
                    continue
                self._inflight[pid] = threading.Event()  # reserve so get() waits
                self.misses += 1  # every scheduled prefetch is a disk-load miss
                self._misses_by_class[_piece_class(pid)] += 1
                seq = next(self._seq)
                if speculative:
                    self.speculative_issued += 1
                    self._spec_pending.add(pid)
                    self._pq.put((tier, seq, pid))  # slack tier; never warmed
                elif self._warm_pool is None:
                    self._pq.put((tier, seq, pid))
                else:
                    self._warm_pool.submit(self._warm_then_queue, seq, pid)

    def _warm_then_queue(self, seq: int, pid: str) -> None:
        # Warm the OS page cache for this piece's file (parallel, off the mlx load
        # path), then hand the pid to the single materialize worker, whose
        # mx.load+mx.eval now reads warm pages instead of doing a cold disk read.
        try:
            self._warm(pid)
        except Exception:
            pass  # best-effort: a failed warm just means the materialize reads cold
        self._pq.put((0, seq, pid))  # warmed pids are always demand-tier

    def _warm(self, pid: str) -> None:
        path = self._pather(pid)
        fd = os.open(os.fspath(path), os.O_RDONLY)
        try:
            while os.read(fd, 1 << 23):  # 8 MiB chunks; populate the page cache
                pass
        finally:
            os.close(fd)

    def warm_bulk(self, pids: Iterable[str]) -> None:
        """Fire-and-forget parallel page-cache warm of a known-in-advance bulk of
        pieces (the demand-critical prefill / spec-verify expert set, computed
        after the router but BEFORE the serial get() loop touches any of them).
        Raw-reading these files with a small thread pool reaches SSD bandwidth and
        populates the OS page cache, so the subsequent serial mx.load+mx.eval in
        get()'s materialize path reads warm pages instead of cold-faulting each
        file single-threaded.

        Pure page-cache populate: NO accounting is touched (no _inflight reserve,
        no miss counters, no staging) — get() keeps all bookkeeping, and a get()
        racing a warm of the same file is harmless (concurrent reads). Pids that
        are already resident / staged / in-flight need no disk read (or one is
        already happening), so they are skipped. No-op without a _pather (nothing
        to raw-read). We do NOT wait on the submitted warms: the serial get loop
        that follows overlaps with them, the OS page cache mediating."""
        if self._pather is None:
            return
        with self._lock:
            todo = [
                pid for pid in pids
                if pid not in self._region(pid)
                and pid not in self._staging
                and pid not in self._inflight
            ]
            if not todo:
                return
            if self._bulk_warm_pool is None:
                self._bulk_warm_pool = ThreadPoolExecutor(
                    max_workers=8, thread_name_prefix="nunspark-bulkwarm")
            pool = self._bulk_warm_pool
            for pid in todo:
                pool.submit(self._warm_one_best_effort, pid)

    def _warm_one_best_effort(self, pid: str) -> None:
        # Best-effort raw read (like _warm_then_queue): a failed warm just means
        # the subsequent get() materialize reads that file cold.
        try:
            self._warm(pid)
        except Exception:
            pass

    def close(self) -> None:
        # Swap the pool out under the lock so a concurrent prefetch() either sees a
        # live pool (and submits) or None (and uses the queue) — never submits to a
        # pool being shut down. shutdown() blocks, so it runs OUTSIDE the lock.
        # Drains in-flight warms (each then queues its materialize), then stops the
        # materialize worker with the sentinel. Safe to call twice.
        with self._lock:
            pool, self._warm_pool = self._warm_pool, None
            bulk_pool, self._bulk_warm_pool = self._bulk_warm_pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        # Bulk warms are best-effort raw page-cache reads, so we needn't wait for
        # them; swapped out under the lock (same care as _warm_pool) so a
        # concurrent warm_bulk sees None and never submits to a shut-down pool.
        if bulk_pool is not None:
            bulk_pool.shutdown(wait=False)
        # Sentinel tier 3 sorts after demand (0) and speculative (1), so any
        # already-queued prefetches drain before the worker stops.
        self._pq.put((3, next(self._seq), None))
        self._worker.join()

    @property
    def resident_ids(self) -> list[str]:
        with self._lock:
            return list(self._resident) + list(self._experts)

    def stats(self) -> dict:
        """Per-class ("dense" / "core" / "expert") hits, misses, and bytes loaded
        (disk-materialized, i.e. counted in _materialize — not the cheaper
        already-resident hit path), plus current per-region occupancy in bytes
        ("main" = dense+core MRU region, "expert" = LRU expert region)."""
        with self._lock:
            return {
                "hits": dict(self._hits_by_class),
                "misses": dict(self._misses_by_class),
                "bytes_loaded": dict(self._bytes_by_class),
                "resident_bytes": {
                    "main": self._main_bytes,
                    "expert": self._expert_bytes,
                },
                "speculative": {
                    "issued": self.speculative_issued,
                    "used": self.speculative_used,
                    "wasted_bytes": self.speculative_wasted_bytes,
                },
            }

    # ---- internals ----
    def _run(self) -> None:
        while True:
            _tier, _seq, pid = self._pq.get()
            if pid is None:
                return
            with self._lock:
                if pid in self._region(pid) or pid in self._staging:
                    ev = self._inflight.pop(pid, None)
                    self._spec_pending.discard(pid)
                    if ev:
                        ev.set()
                    continue
                ev = self._inflight.get(pid)
                if ev is None:
                    self._spec_pending.discard(pid)
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
                nbytes = _piece_bytes(weights)
                # Still speculative? A demand get() joining the in-flight load
                # already discarded the pid from _spec_pending (and counted it
                # used), converting this into a plain demand load.
                is_spec = pid in self._spec_pending
                self._spec_pending.discard(pid)
                # Disk read happened for either path — count it once here.
                self._bytes_by_class[_piece_class(pid)] += nbytes
                if is_spec and _piece_class(pid) == "expert":
                    # Speculative expert: land in the STAGING buffer, never the
                    # expert LRU, so a wrong/oversized batch of guesses can never
                    # evict the demand working set. A demand get() promotes it
                    # (used); begin_pass()/cap pressure drops it (wasted).
                    self._stage_locked(pid, weights, nbytes)
                    self.peak_bytes = max(
                        self.peak_bytes, self._bytes + self._staging_bytes)
                    # Staging shares the expert budget: squeeze the LRU tail now
                    # so expert LRU + staging stay within the expert share.
                    self._evict_locked(protect=pid)
                    return weights
                if _piece_class(pid) == "expert":
                    # Demand experts land HOT (append == most recent).
                    self._experts[pid] = weights
                    self._expert_bytes += nbytes
                    self._split_active = True
                else:
                    self._resident[pid] = weights
                    self._main_bytes += nbytes
                self._bytes += nbytes
                self.peak_bytes = max(
                    self.peak_bytes, self._bytes + self._staging_bytes)  # transient peak
                self._evict_locked(protect=pid)
            return weights
        finally:
            # Always release the reservation and wake waiters, even on failure,
            # so a loader/eval error can never permanently strand a get() waiter.
            with self._lock:
                self._inflight.pop(pid, None)
            ev.set()

    def _evict_locked(self, protect: str) -> None:
        # Two independent regions; each is brought under its own cap and never
        # evicts the other's pieces. Before the first expert insert the main
        # region owns the whole budget (single-region behavior, unchanged).
        evicted = False
        main_cap = (self._budget - self._expert_budget) if self._split_active else self._budget

        # main region — scan-resistant (MRU) eviction: the engine scans layers
        # 0..N-1 in the SAME order every token, so the just-used piece is the one
        # needed again soonest. Evicting from the most-recently-used end (iterate
        # reversed) keeps a stable early-layer prefix resident across tokens; LRU
        # would keep the trailing layers and re-read everything next token.
        for pid in reversed(list(self._resident)):
            if self._main_bytes <= main_cap:
                break
            if pid == protect or pid in self._pinned:
                continue
            w = self._resident.pop(pid)
            n = _piece_bytes(w)
            self._main_bytes -= n
            self._bytes -= n
            evicted = True

        # expert region — plain LRU over demand pieces only (first = least recent).
        # Expert reuse is sparse/skewed (M1), which is exactly LRU's home turf.
        # Speculative loads never enter this region (they live in the staging
        # buffer), so eviction here is never POLICY-sacrificed to a speculative
        # guess — the v3 fix for the demand-set thrash. Staging bytes do count
        # against the shared expert budget (memory accounting, bounded by the
        # staging cap), so occupied staging squeezes the LRU tail rather than
        # growing total process bytes past budget_bytes.
        if self._expert_bytes + self._staging_bytes > self._expert_budget:
            for pid in list(self._experts):  # LRU order
                if self._expert_bytes + self._staging_bytes <= self._expert_budget:
                    break
                if pid == protect or pid in self._pinned:
                    continue
                w = self._experts.pop(pid)
                n = _piece_bytes(w)
                self._expert_bytes -= n
                self._bytes -= n
                evicted = True

        if evicted:
            mx.clear_cache()  # only reclaim Metal buffers when we actually freed some
