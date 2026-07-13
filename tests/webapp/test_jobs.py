import time

from nunspark.webapp.jobs import JobManager
from nunspark.webapp.schemas import BatchRequest, JobStatus


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
