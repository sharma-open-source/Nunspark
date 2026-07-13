import io
import json

from fastapi.testclient import TestClient

from nunspark.webapp.app import create_app


def test_sse_emits_token_and_done(tmp_path, tiny_packed_dir):
    # Real runner (default), real tiny streaming engine.
    app = create_app(packed_root=tiny_packed_dir.parent, hf_cache=tmp_path / "nohf")
    out = tmp_path / "out"; out.mkdir()
    with TestClient(app) as client:
        up = client.post("/api/files",
                         files={"files": ("doc.txt", io.BytesIO(b"Hello there."), "text/plain")})
        fid = up.json()["files"][0]["id"]
        jobs = client.post("/api/batch", json={
            "model": str(tiny_packed_dir.resolve()),
            "instruction": "Continue:", "use_chat_template": False,
            "max_tokens": 4, "output_dir": str(out), "file_ids": [fid],
        }).json()["jobs"]
        job_id = jobs[0]["id"]

        types = []
        with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                ev = json.loads(line[len("data: "):])
                types.append(ev["type"])
                if ev["type"] in ("done", "error", "cancelled"):
                    break
        assert "started" in types
        assert "done" in types
        assert (out / "doc.txt.out.txt").is_file()
