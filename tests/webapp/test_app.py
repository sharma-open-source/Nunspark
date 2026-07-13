import io
import json

from fastapi.testclient import TestClient

from nunspark.webapp.app import create_app
from nunspark.webapp.schemas import JobStatus


def _fake_runner(job, *, pool, emit, should_cancel):
    job.status = JobStatus.RUNNING
    emit({"type": "started", "job": job.public()})
    emit({"type": "token", "job_id": job.id, "text": "hi"})
    job.status = JobStatus.DONE
    job.tokens_done = 1
    emit({"type": "done", "job": job.public()})


def _client(tmp_path):
    packs = tmp_path / "packs"
    (packs / "m").mkdir(parents=True)
    (packs / "m" / "manifest.json").write_text(json.dumps({"num_layers": 1}))
    app = create_app(packed_root=packs, hf_cache=tmp_path / "nohf", runner=_fake_runner)
    return TestClient(app), packs


def test_list_models(tmp_path):
    client, packs = _client(tmp_path)
    r = client.get("/api/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "m" in names


def test_upload_and_batch_flow(tmp_path):
    client, packs = _client(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    up = client.post("/api/files",
                     files={"files": ("doc.txt", io.BytesIO(b"hello"), "text/plain")})
    assert up.status_code == 200
    fid = up.json()["files"][0]["id"]

    r = client.post("/api/batch", json={
        "model": str((packs / "m").resolve()),
        "instruction": "go", "output_dir": str(out), "file_ids": [fid],
    })
    assert r.status_code == 200
    assert len(r.json()["jobs"]) == 1

    jobs = client.get("/api/jobs").json()["jobs"]
    assert len(jobs) == 1


def test_batch_rejects_unpacked_model(tmp_path):
    client, packs = _client(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    r = client.post("/api/batch", json={
        "model": str(tmp_path / "not-packed"),
        "output_dir": str(out), "file_ids": ["x"],
    })
    assert r.status_code == 400
    assert "pack" in r.json()["detail"].lower()


def test_batch_rejects_bad_output_dir(tmp_path):
    client, packs = _client(tmp_path)
    r = client.post("/api/batch", json={
        "model": str((packs / "m").resolve()),
        "output_dir": "relative/path", "file_ids": ["x"],
    })
    assert r.status_code == 400


def test_pack_requires_source(tmp_path):
    client, packs = _client(tmp_path)
    r = client.post("/api/pack", json={})
    assert r.status_code == 400
    assert "source" in r.json()["detail"].lower()
