import json
import threading
from pathlib import Path

from nunspark.webapp.engine_pool import EnginePool
from nunspark.webapp.runner import build_prompt, run_generation
from nunspark.webapp.schemas import Job, JobStatus


def test_build_prompt_with_instruction():
    text = build_prompt("Summarize:", "The body.", tokenizer=None, use_chat_template=False)
    assert "Summarize:" in text and "The body." in text


def test_run_generation_writes_output_and_sidecar(tmp_path, tiny_packed_dir):
    src = tmp_path / "doc.txt"
    src.write_text("Hello world. This is a test document.")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    job = Job(
        id="j1", batch_id="b1", file_name="doc.txt", file_path=str(src),
        model=str(tiny_packed_dir), draft=None, preset="lossless",
        instruction="Continue:", use_chat_template=False,
        max_tokens=4, temperature=0.0, output_dir=str(out_dir),
        advanced={"budget": "4GB"},
    )
    events = []
    pool = EnginePool()
    try:
        run_generation(job, pool=pool,
                        emit=lambda e: events.append(e),
                        should_cancel=lambda: False)
    finally:
        pool.close()

    assert job.status == JobStatus.DONE
    out_file = out_dir / "doc.txt.out.txt"
    assert out_file.is_file()
    meta = json.loads((out_dir / "doc.txt.meta.json").read_text())
    assert meta["file_name"] == "doc.txt"
    assert "tok_per_s" in meta["metrics"]
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "done"


def test_run_generation_cancel_midway(tmp_path, tiny_packed_dir):
    src = tmp_path / "doc.txt"
    src.write_text("Hello world.")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    job = Job(
        id="j2", batch_id="b1", file_name="doc.txt", file_path=str(src),
        model=str(tiny_packed_dir), use_chat_template=False,
        max_tokens=1000, output_dir=str(out_dir), advanced={"budget": "4GB"},
    )
    cancel = threading.Event()
    def should_cancel():
        if job.tokens_done >= 1:
            cancel.set()
        return cancel.is_set()

    pool = EnginePool()
    try:
        run_generation(job, pool=pool, emit=lambda e: None, should_cancel=should_cancel)
    finally:
        pool.close()
    assert job.status == JobStatus.CANCELLED
    assert (out_dir / "doc.txt.out.txt").is_file()


def test_run_generation_unique_output_on_name_collision(tmp_path, tiny_packed_dir):
    src = tmp_path / "doc.txt"; src.write_text("Hello world.")
    out_dir = tmp_path / "out"; out_dir.mkdir()
    from nunspark.webapp.engine_pool import EnginePool
    from nunspark.webapp.runner import run_generation
    from nunspark.webapp.schemas import Job, JobStatus
    pool = EnginePool()
    try:
        for jid in ("ja", "jb"):
            job = Job(id=jid, batch_id="b1", file_name="doc.txt", file_path=str(src),
                      model=str(tiny_packed_dir), use_chat_template=False,
                      max_tokens=3, output_dir=str(out_dir), advanced={"budget": "4GB"})
            run_generation(job, pool=pool, emit=lambda e: None, should_cancel=lambda: False)
            assert job.status == JobStatus.DONE
    finally:
        pool.close()
    assert (out_dir / "doc.txt.out.txt").is_file()
    assert (out_dir / "doc.txt.1.out.txt").is_file()
    assert (out_dir / "doc.txt.meta.json").is_file()
    assert (out_dir / "doc.txt.1.meta.json").is_file()


def test_run_generation_bad_output_dir_does_not_raise(tmp_path, tiny_packed_dir):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")  # a file, so blocker/sub can't be created as a dir
    bad_out = blocker / "sub"
    src = tmp_path / "doc.txt"; src.write_text("hi")
    from nunspark.webapp.engine_pool import EnginePool
    from nunspark.webapp.runner import run_generation
    from nunspark.webapp.schemas import Job, JobStatus
    job = Job(id="j3", batch_id="b1", file_name="doc.txt", file_path=str(src),
              model=str(tiny_packed_dir), use_chat_template=False,
              max_tokens=2, output_dir=str(bad_out), advanced={"budget": "4GB"})
    events = []
    pool = EnginePool()
    try:
        run_generation(job, pool=pool, emit=lambda e: events.append(e),
                        should_cancel=lambda: False)  # must not raise
    finally:
        pool.close()
    assert job.status == JobStatus.ERROR
    assert events[-1]["type"] == "error"
