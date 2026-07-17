import time

from nunspark.webapp.jobs import MAX_BATCH, JobManager, _compat_key
from nunspark.webapp.schemas import Advanced, BatchRequest, Job, JobStatus


def _fake_runner_factory(order):
    def fake_runner(job, *, pool, emit, should_cancel):
        order.append(job.id)
        job.status = JobStatus.RUNNING
        emit({"type": "started", "job": job.public()})
        if should_cancel():
            job.status = JobStatus.CANCELLED
            emit({"type": "cancelled", "job": job.public()})
            return
        job.status = JobStatus.DONE
        job.tokens_done = 3
        emit({"type": "done", "job": job.public()})
    return fake_runner


def _wait(mgr, pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_batch_fans_out_one_job_per_file(tmp_path):
    order = []
    mgr = JobManager(runner=_fake_runner_factory(order))
    mgr.register_files({"f1": str(tmp_path / "a.txt"), "f2": str(tmp_path / "b.txt")})
    for n in ("a.txt", "b.txt"):
        (tmp_path / n).write_text("x")
    req = BatchRequest(model="/m", instruction="go", output_dir=str(tmp_path),
                       file_ids=["f1", "f2"])
    jobs = mgr.submit_batch(req)
    assert len(jobs) == 2
    try:
        assert _wait(mgr, lambda: all(j.status == JobStatus.DONE for j in jobs))
        assert order == [jobs[0].id, jobs[1].id]  # FIFO
    finally:
        mgr.shutdown()


def test_cancel_before_run(tmp_path):
    order = []
    mgr = JobManager(runner=_fake_runner_factory(order))
    (tmp_path / "a.txt").write_text("x")
    mgr.register_files({"f1": str(tmp_path / "a.txt")})
    mgr.pause()
    req = BatchRequest(model="/m", output_dir=str(tmp_path), file_ids=["f1"])
    jobs = mgr.submit_batch(req)
    mgr.cancel(jobs[0].id)
    mgr.resume()
    try:
        assert _wait(mgr, lambda: jobs[0].status == JobStatus.CANCELLED)
    finally:
        mgr.shutdown()


def _fake_batch_runner_factory(calls):
    def fake_batch_runner(jobs, *, pool, emit, should_cancel):
        calls.append([j.id for j in jobs])
        for j in jobs:
            j.status = JobStatus.DONE
            emit(j.id, {"type": "done", "job": j.public()})
    return fake_batch_runner


def _fake_single_runner_factory(order):
    def fake_runner(job, *, pool, emit, should_cancel):
        order.append(job.id)
        job.status = JobStatus.DONE
        emit({"type": "done", "job": job.public()})
    return fake_runner


# --- Plan 5 M3b-3: batch dispatcher (grouping) --------------------------------
# These use mgr.pause()/resume() around submission so every job of interest is
# already sitting in the queue before the worker's first (non-blocking) peek --
# otherwise grouping would race the main thread's `submit_batch` calls.

def test_batch_dispatcher_groups_compatible_jobs(tmp_path):
    batch_calls: list = []
    mgr = JobManager(runner=_fake_single_runner_factory([]),
                      batch_runner=_fake_batch_runner_factory(batch_calls),
                      start_paused=True)
    for n in ("a.txt", "b.txt"):
        (tmp_path / n).write_text("x")
    mgr.register_files({"f1": str(tmp_path / "a.txt"), "f2": str(tmp_path / "b.txt")})
    req = BatchRequest(model="/m", output_dir=str(tmp_path), file_ids=["f1", "f2"])
    jobs = mgr.submit_batch(req)
    mgr.resume()
    try:
        assert _wait(mgr, lambda: all(j.status == JobStatus.DONE for j in jobs))
        assert batch_calls == [[jobs[0].id, jobs[1].id]]
    finally:
        mgr.shutdown()


def test_batch_dispatcher_splits_incompatible_groups(tmp_path):
    """job1/job2 share max_tokens=10 and group; job3 (max_tokens=99) breaks
    the run so job3 and job4 (max_tokens=10 again, but no longer ADJACENT to
    job1/job2) each run alone -- grouping is adjacency-based over the FIFO
    queue, not a global regroup by key."""
    batch_calls: list = []
    single_order: list = []
    mgr = JobManager(runner=_fake_single_runner_factory(single_order),
                      batch_runner=_fake_batch_runner_factory(batch_calls),
                      start_paused=True)
    for n in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (tmp_path / n).write_text("x")
    mgr.register_files({
        "f1": str(tmp_path / "a.txt"), "f2": str(tmp_path / "b.txt"),
        "f3": str(tmp_path / "c.txt"), "f4": str(tmp_path / "d.txt"),
    })
    jobs = []
    jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path),
                                           file_ids=["f1"], max_tokens=10))
    jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path),
                                           file_ids=["f2"], max_tokens=10))
    jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path),
                                           file_ids=["f3"], max_tokens=99))
    jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path),
                                           file_ids=["f4"], max_tokens=10))
    mgr.resume()
    try:
        assert _wait(mgr, lambda: all(j.status == JobStatus.DONE for j in jobs))
    finally:
        mgr.shutdown()

    assert batch_calls == [[jobs[0].id, jobs[1].id]]
    assert single_order == [jobs[2].id, jobs[3].id]


def test_batch_dispatcher_honors_batch_opt_out(tmp_path):
    batch_calls: list = []
    single_order: list = []
    mgr = JobManager(runner=_fake_single_runner_factory(single_order),
                      batch_runner=_fake_batch_runner_factory(batch_calls),
                      start_paused=True)
    for n in ("a.txt", "b.txt"):
        (tmp_path / n).write_text("x")
    mgr.register_files({"f1": str(tmp_path / "a.txt"), "f2": str(tmp_path / "b.txt")})
    jobs = []
    jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path),
                                           file_ids=["f1"], advanced=Advanced(batch=False)))
    jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path), file_ids=["f2"]))
    mgr.resume()
    try:
        assert _wait(mgr, lambda: all(j.status == JobStatus.DONE for j in jobs))
    finally:
        mgr.shutdown()

    assert batch_calls == []
    assert single_order == [jobs[0].id, jobs[1].id]


def test_batch_dispatcher_caps_group_at_max_batch(tmp_path):
    batch_calls: list = []
    mgr = JobManager(runner=_fake_single_runner_factory([]),
                      batch_runner=_fake_batch_runner_factory(batch_calls),
                      start_paused=True)
    n_jobs = MAX_BATCH + 2
    names = [f"f{i}.txt" for i in range(n_jobs)]
    for n in names:
        (tmp_path / n).write_text("x")
    mgr.register_files({f"id{i}": str(tmp_path / n) for i, n in enumerate(names)})
    jobs = []
    for i in range(n_jobs):
        jobs += mgr.submit_batch(BatchRequest(model="/m", output_dir=str(tmp_path),
                                               file_ids=[f"id{i}"]))
    mgr.resume()
    try:
        assert _wait(mgr, lambda: all(j.status == JobStatus.DONE for j in jobs))
    finally:
        mgr.shutdown()

    assert [len(c) for c in batch_calls] == [MAX_BATCH, n_jobs - MAX_BATCH]


def test_compat_key_excludes_draft_and_opt_out():
    base = dict(id="j", batch_id="b", file_name="f", file_path="p", model="/m")
    assert _compat_key(Job(**base, draft="/draft")) is None
    assert _compat_key(Job(**base, advanced={"batch": False})) is None
    assert _compat_key(Job(**base)) is not None


def test_compat_key_differs_on_params():
    base = dict(id="j", batch_id="b", file_name="f", file_path="p", model="/m")
    k1 = _compat_key(Job(**base, max_tokens=10))
    k2 = _compat_key(Job(**base, max_tokens=20))
    assert k1 != k2
    k3 = _compat_key(Job(**base, temperature=0.5))
    k4 = _compat_key(Job(**base, temperature=0.5))
    assert k3 == k4


def test_subscribe_receives_events(tmp_path):
    order = []
    mgr = JobManager(runner=_fake_runner_factory(order))
    (tmp_path / "a.txt").write_text("x")
    mgr.register_files({"f1": str(tmp_path / "a.txt")})
    req = BatchRequest(model="/m", output_dir=str(tmp_path), file_ids=["f1"])
    jobs = mgr.submit_batch(req)
    q = mgr.subscribe(jobs[0].id)
    try:
        seen = []
        end = time.time() + 5
        while time.time() < end:
            ev = q.get(timeout=5)
            seen.append(ev["type"])
            if ev["type"] in ("done", "error", "cancelled"):
                break
        assert "started" in seen and "done" in seen
    finally:
        mgr.shutdown()
