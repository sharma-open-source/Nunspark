from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .jobs import JobManager
from .models import ModelRegistry
from .runner import run_generation
from .schemas import BatchRequest

_STATIC = Path(__file__).parent / "static"


def create_app(*, packed_root: Path | str, hf_cache: Path | str | None = None,
               runner: Callable = run_generation) -> FastAPI:
    app = FastAPI(title="NunSpark")
    registry = ModelRegistry(packed_root=packed_root, hf_cache=hf_cache)
    manager = JobManager(runner=runner)
    upload_dir = tempfile.TemporaryDirectory(prefix="nunspark_uploads_")

    app.state.manager = manager
    app.state.registry = registry
    app.state.upload_dir = upload_dir

    @app.on_event("shutdown")
    def _shutdown() -> None:
        manager.shutdown()
        upload_dir.cleanup()

    @app.get("/api/models")
    def list_models() -> dict:
        return {"models": registry.list_models()}

    @app.post("/api/files")
    async def upload_files(files: list[UploadFile] = File(...)) -> dict:
        saved = []
        for f in files:
            fid = uuid.uuid4().hex
            safe_name = Path(f.filename).name  # strip any directory components
            sub = Path(upload_dir.name) / fid
            sub.mkdir(parents=True, exist_ok=True)
            dest = sub / safe_name
            dest.write_bytes(await f.read())
            manager.register_files({fid: str(dest)})
            saved.append({"id": fid, "name": safe_name})
        return {"files": saved}

    @app.post("/api/batch")
    def submit_batch(req: BatchRequest) -> dict:
        out = Path(req.output_dir)
        if not out.is_absolute():
            raise HTTPException(400, "output_dir must be an absolute path")
        if not registry.is_packed(req.model):
            raise HTTPException(400, "Model is not packed. Pack this model first.")
        out.mkdir(parents=True, exist_ok=True)
        jobs = manager.submit_batch(req)
        return {"jobs": [j.public() for j in jobs]}

    @app.post("/api/pack")
    def pack_model(body: dict) -> dict:
        source = body.get("source")
        if not source:
            raise HTTPException(400, "source is required")
        out_dir = body.get("out_dir") or str(Path(packed_root) / Path(source).name)
        from ..packer import pack
        try:
            manifest = pack(source, out_dir)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"pack failed: {exc}")
        return {"out_dir": out_dir, "num_layers": manifest.num_layers}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": manager.list_jobs()}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        if manager.get(job_id) is None:
            raise HTTPException(404, "unknown job")
        manager.cancel(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str):
        if manager.get(job_id) is None:
            raise HTTPException(404, "unknown job")
        q = manager.subscribe(job_id)

        def stream():
            terminal = {"done", "error", "cancelled"}
            while True:
                ev = q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") in terminal:
                    break
        return StreamingResponse(stream(), media_type="text/event-stream")

    if _STATIC.is_dir():
        app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")

    return app
