from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlx_lm.tokenizer_utils import load as load_tokenizer

from ..archspec import KVQuant
from ..engine import StreamingEngine
from ..manifest import Manifest


@dataclass
class EngineHandle:
    engine: StreamingEngine
    tokenizer: object
    draft_model: object | None
    key: tuple


def _kv_key(kv_quant: KVQuant | None):
    return None if kv_quant is None else (kv_quant.bits, kv_quant.group_size)


class EnginePool:
    """Holds at most one loaded engine (+ draft + tokenizer). Re-acquiring the
    same config returns the cached handle; a different config closes the old
    engine and loads the new one. The engine streams weights from disk, so
    'loading' it is cheap, but reusing it across a batch keeps the draft model
    and tokenizer resident and avoids re-reading the manifest per file."""

    def __init__(self) -> None:
        self._handle: EngineHandle | None = None

    def acquire(self, model: str, draft: str | None, *,
                budget_bytes: int, kv_quant: KVQuant | None) -> EngineHandle:
        key = (str(Path(model).resolve()), draft, budget_bytes, _kv_key(kv_quant))
        if self._handle is not None and self._handle.key == key:
            return self._handle
        self.close()

        packed = Path(model)
        manifest = Manifest.load(packed / "manifest.json")
        engine = StreamingEngine(packed, manifest, budget_bytes=budget_bytes, prefetch=True)
        tokenizer = load_tokenizer(packed)

        draft_model = None
        if draft:
            from mlx_lm import load as load_mlxlm
            draft_model, _ = load_mlxlm(draft)

        self._handle = EngineHandle(engine, tokenizer, draft_model, key)
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.engine.close()
            self._handle = None
