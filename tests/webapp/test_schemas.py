from nunspark.webapp.schemas import BatchRequest, Job, JobStatus, preset_params


def test_batch_request_defaults():
    req = BatchRequest(
        model="/packs/m",
        instruction="Summarize.",
        output_dir="/out",
        file_ids=["a", "b"],
    )
    assert req.preset == "lossless"
    assert req.use_chat_template is True
    assert req.max_tokens == 2048
    assert req.temperature == 0.0
    assert req.draft is None


def test_preset_params_lossless_and_fast():
    assert preset_params("lossless") == (1, 16)
    assert preset_params("fast") == (3, 24)


def test_preset_params_advanced_override():
    assert preset_params("lossless", accept_top_k=5, num_draft_tokens=8) == (5, 8)


def test_job_defaults():
    job = Job(id="j1", batch_id="b1", file_name="doc.txt", file_path="/tmp/doc.txt")
    assert job.status == JobStatus.QUEUED
    assert job.tokens_done == 0
    assert job.error is None
