from __future__ import annotations

import os
from pathlib import Path


def _default_hf_cache() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


class ModelRegistry:
    """Discovers usable models: packed NunSpark folders (ready to stream) and
    raw Hugging Face cache models (must be packed first)."""

    def __init__(self, packed_root: Path | str, hf_cache: Path | str | None = None):
        self.packed_root = Path(packed_root)
        self.hf_cache = Path(hf_cache) if hf_cache is not None else _default_hf_cache()

    def list_models(self) -> list[dict]:
        return self._scan_packed() + self._scan_hf_cache()

    def is_packed(self, path: str) -> bool:
        return (Path(path) / "manifest.json").is_file()

    def _scan_packed(self) -> list[dict]:
        out: list[dict] = []
        if not self.packed_root.is_dir():
            return out
        for child in sorted(self.packed_root.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                out.append({
                    "kind": "packed",
                    "name": child.name,
                    "path": str(child.resolve()),
                    "size": _dir_size(child),
                })
        print(out)
        return out

    def _scan_hf_cache(self) -> list[dict]:
        out: list[dict] = []
        if not self.hf_cache.is_dir():
            return out
        for repo in sorted(self.hf_cache.glob("models--*")):
            snaps = repo / "snapshots"
            if not snaps.is_dir():
                continue
            revs = [d for d in snaps.iterdir() if d.is_dir()]
            if not revs:
                continue
            snap = max(revs, key=lambda d: d.stat().st_mtime)
            name = repo.name[len("models--"):].replace("--", "/")
            out.append({
                "kind": "unpacked",
                "name": name,
                "path": str(snap),
                "size": _dir_size(snap),
            })
        return out
