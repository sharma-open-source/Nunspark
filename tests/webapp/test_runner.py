import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

from nunspark.webapp.engine_pool import EnginePool
from nunspark.webapp.runner import build_prompt, run_generation, run_generation_batch
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


def test_run_generation_does_not_call_missing_pool_release(tmp_path, tiny_packed_dir):
    """Regression test: EnginePool only exposes acquire/close, never release --
    it is an intentional cross-job cache of one loaded engine (see its
    docstring), not something a single job should tear down. run_generation
    used to do `stack.callback(pool.release, handle)`, which raised
    AttributeError the instant it ran against any real (non-mocked) pool.

    We spec the mock to EnginePool so any attribute other than acquire/close
    raises AttributeError immediately, exactly like the real class -- while
    still delegating to a real EnginePool underneath so generation actually
    runs against a real engine/tokenizer.
    """
    src = tmp_path / "doc.txt"
    src.write_text("Hello world. This is a test document.")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    job = Job(
        id="j5", batch_id="b1", file_name="doc.txt", file_path=str(src),
        model=str(tiny_packed_dir), draft=None, preset="lossless",
        instruction="Continue:", use_chat_template=False,
        max_tokens=4, temperature=0.0, output_dir=str(out_dir),
        advanced={"budget": "4GB"},
    )

    real_pool = EnginePool()
    pool = MagicMock(spec=EnginePool)
    pool.acquire.side_effect = real_pool.acquire
    pool.close.side_effect = real_pool.close
    assert not hasattr(pool, "release")

    events = []
    try:
        run_generation(job, pool=pool,
                        emit=lambda e: events.append(e),
                        should_cancel=lambda: False)
    finally:
        real_pool.close()

    assert job.status == JobStatus.DONE
    assert job.error is None
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


# --- Input guards (binary rejection + prompt-token cap) ----------------------
# These use MagicMock(spec=EnginePool) like the release-regression test above,
# so we can assert the pool (and thus the model / generate) is never touched
# for a rejected upload.

def _guard_job(tmp_path, data: bytes, advanced=None, **overrides):
    src = tmp_path / overrides.pop("file_name", "upload.bin")
    src.write_bytes(data)
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    return Job(
        id="jg", batch_id="b1", file_name=src.name, file_path=str(src),
        model="/nonexistent/model", use_chat_template=False,
        max_tokens=4, output_dir=str(out_dir),
        advanced=advanced or {"budget": "4GB"}, **overrides,
    )


def test_run_generation_rejects_pdf(tmp_path):
    job = _guard_job(tmp_path, b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nbinary body",
                     file_name="doc.pdf")
    pool = MagicMock(spec=EnginePool)
    events = []
    run_generation(job, pool=pool, emit=lambda e: events.append(e),
                   should_cancel=lambda: False)
    assert job.status == JobStatus.ERROR
    assert "PDF" in job.error
    assert ".txt/.md" in job.error
    pool.acquire.assert_not_called()  # generate never reached
    assert events[-1]["type"] == "error"


def test_run_generation_rejects_nul_binary(tmp_path):
    job = _guard_job(tmp_path, b"MZ\x90\x00\x03binary\x00tail")
    pool = MagicMock(spec=EnginePool)
    run_generation(job, pool=pool, emit=lambda e: None,
                   should_cancel=lambda: False)
    assert job.status == JobStatus.ERROR
    assert "binary" in job.error
    pool.acquire.assert_not_called()


def test_run_generation_rejects_high_replacement_ratio(tmp_path):
    # No magic, no NUL bytes -- but >5% of the decoded text is U+FFFD.
    job = _guard_job(tmp_path, b"hello " + b"\xfe\xfb" * 4096)
    pool = MagicMock(spec=EnginePool)
    run_generation(job, pool=pool, emit=lambda e: None,
                   should_cancel=lambda: False)
    assert job.status == JobStatus.ERROR
    assert "UTF-8" in job.error
    pool.acquire.assert_not_called()


def _mock_pool_with_prompt(num_tokens: int, engine=None):
    """Pool whose engine tokenizes any text to `num_tokens` ids -- lets the
    cap tests exercise huge prompts without a real tokenizer."""
    handle = MagicMock()
    handle.tokenizer.encode.return_value = list(range(num_tokens))
    handle.draft_model = None
    if engine is not None:
        handle.engine = engine
    pool = MagicMock(spec=EnginePool)
    pool.acquire.return_value = handle
    return pool


def test_run_generation_prompt_cap_override_exceeded(tmp_path):
    job = _guard_job(tmp_path, b"plain text body",
                     advanced={"budget": "4GB", "max_prompt_tokens": 128})
    pool = _mock_pool_with_prompt(50_000)
    run_generation(job, pool=pool, emit=lambda e: None,
                   should_cancel=lambda: False)
    assert job.status == JobStatus.ERROR
    assert "50000" in job.error and "128" in job.error
    assert "max_prompt_tokens" in job.error


def test_run_generation_prompt_cap_override_high_passes(tmp_path, tiny_packed_dir):
    src = tmp_path / "doc.txt"
    src.write_text("Hello world. This is a test document.")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    job = Job(
        id="j6", batch_id="b1", file_name="doc.txt", file_path=str(src),
        model=str(tiny_packed_dir), use_chat_template=False,
        max_tokens=4, output_dir=str(out_dir),
        advanced={"budget": "4GB", "max_prompt_tokens": 100_000},
    )
    pool = EnginePool()
    try:
        run_generation(job, pool=pool, emit=lambda e: None,
                       should_cancel=lambda: False)
    finally:
        pool.close()
    assert job.status == JobStatus.DONE
    assert job.error is None


def _qwen30b_like_args():
    from types import SimpleNamespace
    # 48 layers x 4 KV heads x 128 head_dim x 4 bytes = 98304 B/token,
    # matching the ~98 KB/token Qwen3-30B figure the guard exists for.
    return SimpleNamespace(
        num_hidden_layers=48, num_key_value_heads=4,
        num_attention_heads=32, head_dim=128, hidden_size=4096,
    )


def test_run_generation_derived_cap_arithmetic(tmp_path, monkeypatch):
    from nunspark.webapp import runner as runner_mod

    per_token = 48 * 4 * 128 * 4
    budget = runner_mod._parse_size("4GB")
    # RAM sized so (ram - budget - headroom) / per_token == 10_000 exactly,
    # inside the [4096, 65536] clamp window.
    ram = budget + runner_mod._PROMPT_CAP_HEADROOM_BYTES + per_token * 10_000
    monkeypatch.setattr(runner_mod, "_unified_ram_bytes", lambda: ram)

    engine = MagicMock()
    engine.args = _qwen30b_like_args()
    job = _guard_job(tmp_path, b"plain text body")
    pool = _mock_pool_with_prompt(10_001, engine=engine)
    run_generation(job, pool=pool, emit=lambda e: None,
                   should_cancel=lambda: False)
    assert job.status == JobStatus.ERROR
    assert "10001" in job.error and "10000" in job.error
    assert "GB unified RAM" in job.error


# --- Plan 5 M3b-3: batch dispatcher runner (run_generation_batch) ------------

def _batch_job(tmp_path, out_dir, jid, text, **overrides):
    src = tmp_path / f"{jid}.txt"
    src.write_text(text)
    return Job(
        id=jid, batch_id="b1", file_name=f"{jid}.txt", file_path=str(src),
        use_chat_template=False, max_tokens=4, temperature=0.0,
        output_dir=str(out_dir), advanced={"budget": "4GB"}, **overrides,
    )


def test_run_generation_batch_calls_batched_generate_once(tmp_path, tiny_packed_dir, monkeypatch):
    """A compatible group of jobs must ride ONE batched_generate() call with
    N prompts, not N calls to generate()."""
    from nunspark.webapp import runner as runner_mod

    calls = []

    def fake_batched_generate(engine, prompts, **kwargs):
        calls.append(prompts)
        return [[7, 8] for _ in prompts]

    monkeypatch.setattr(runner_mod, "batched_generate", fake_batched_generate)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = [
        _batch_job(tmp_path, out_dir, f"j{i}", f"Hello world {i}.",
                   model=str(tiny_packed_dir))
        for i in range(3)
    ]

    events = []
    pool = EnginePool()
    try:
        run_generation_batch(jobs, pool=pool,
                              emit=lambda jid, e: events.append((jid, e)),
                              should_cancel=lambda jid: False)
    finally:
        pool.close()

    assert len(calls) == 1
    assert len(calls[0]) == 3
    for job in jobs:
        assert job.status == JobStatus.DONE
        assert job.metrics["batched"] is True
        assert job.metrics["batch_size"] == 3
    assert any(e["type"] == "done" for _, e in events)


def test_run_generation_batch_group_of_one_uses_sequential(tmp_path, tiny_packed_dir):
    """A group of a single job gets no benefit from batching -- it must take
    the ordinary sequential generate() path (no `batched` metrics field)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    job = _batch_job(tmp_path, out_dir, "j1", "Hello world.", model=str(tiny_packed_dir))

    events = []
    pool = EnginePool()
    try:
        run_generation_batch([job], pool=pool,
                              emit=lambda jid, e: events.append((jid, e)),
                              should_cancel=lambda jid: False)
    finally:
        pool.close()

    assert job.status == JobStatus.DONE
    assert "batched" not in job.metrics


def test_run_generation_batch_falls_back_when_batched_generate_raises(
    tmp_path, tiny_packed_dir, monkeypatch,
):
    """If batched_generate() itself raises (e.g. a sliding-window arch slipped
    through the group-compatibility check upstream), the group must fall back
    to running sequentially rather than failing every job in it."""
    from nunspark.webapp import runner as runner_mod

    def boom(*a, **k):
        raise ValueError("sliding-window unsupported")

    monkeypatch.setattr(runner_mod, "batched_generate", boom)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = [
        _batch_job(tmp_path, out_dir, f"j{i}", f"Hello world {i}.",
                   model=str(tiny_packed_dir))
        for i in range(2)
    ]

    events = []
    pool = EnginePool()
    try:
        run_generation_batch(jobs, pool=pool,
                              emit=lambda jid, e: events.append((jid, e)),
                              should_cancel=lambda jid: False)
    finally:
        pool.close()

    for job in jobs:
        assert job.status == JobStatus.DONE
        assert job.error is None
        assert "batched" not in job.metrics


def test_run_generation_batch_isolates_bad_upload(tmp_path, tiny_packed_dir):
    """A single bad upload in a group of 2 errors on its own job; the group
    shrinks to a survivor of 1 and still completes via the sequential path."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    good = _batch_job(tmp_path, out_dir, "good", "Hello world.", model=str(tiny_packed_dir))
    bad = _batch_job(tmp_path, out_dir, "bad", "placeholder", model=str(tiny_packed_dir))
    Path(bad.file_path).write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nbinary body")

    events = []
    pool = EnginePool()
    try:
        run_generation_batch([bad, good], pool=pool,
                              emit=lambda jid, e: events.append((jid, e)),
                              should_cancel=lambda jid: False)
    finally:
        pool.close()

    assert bad.status == JobStatus.ERROR
    assert "PDF" in bad.error
    assert good.status == JobStatus.DONE
    assert good.error is None


def test_run_generation_batch_isolates_bad_upload_with_batched_survivors(
    tmp_path, tiny_packed_dir,
):
    """Same isolation, but with enough survivors (2) to actually exercise the
    real batched_generate() path end to end -- confirms `batched`/`batch_size`
    metadata appears and the bad job never touches the others."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    bad = _batch_job(tmp_path, out_dir, "bad", "placeholder", model=str(tiny_packed_dir))
    Path(bad.file_path).write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nbinary body")
    good_jobs = [
        _batch_job(tmp_path, out_dir, f"good{i}", f"Hello world {i}.",
                   model=str(tiny_packed_dir))
        for i in range(2)
    ]
    jobs = [bad, *good_jobs]

    events = []
    pool = EnginePool()
    try:
        run_generation_batch(jobs, pool=pool,
                              emit=lambda jid, e: events.append((jid, e)),
                              should_cancel=lambda jid: False)
    finally:
        pool.close()

    assert bad.status == JobStatus.ERROR
    for job in good_jobs:
        assert job.status == JobStatus.DONE
        assert job.metrics["batched"] is True
        assert job.metrics["batch_size"] == 2


def test_run_generation_derived_cap_clamps_to_min(tmp_path, monkeypatch):
    from nunspark.webapp import runner as runner_mod

    # RAM below budget + headroom -> raw allowance is negative -> min clamp.
    monkeypatch.setattr(runner_mod, "_unified_ram_bytes", lambda: 6 * 1024**3)

    engine = MagicMock()
    engine.args = _qwen30b_like_args()
    job = _guard_job(tmp_path, b"plain text body")
    pool = _mock_pool_with_prompt(5_000, engine=engine)
    run_generation(job, pool=pool, emit=lambda e: None,
                   should_cancel=lambda: False)
    assert job.status == JobStatus.ERROR
    assert "5000" in job.error and "4096" in job.error
