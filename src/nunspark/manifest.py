from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Piece:
    """One streamable unit of weights stored in a single file."""

    piece_id: str
    file: str
    keys: list[str]  # parameter names relative to the piece's module
    # quant is reserved for a later plan; null = fp16 weights as-is.
    quant: dict | None = None


@dataclass
class Manifest:
    """The contract between the Packer (writer) and the runtime (reader)."""

    model_type: str
    config: dict
    tie_word_embeddings: bool
    num_layers: int
    pieces: list[Piece] = field(default_factory=list)

    @staticmethod
    def layer_piece_id(index: int) -> str:
        return f"layer_{index:03d}"

    @staticmethod
    def layer_core_piece_id(index: int) -> str:
        return f"layer_{index:03d}_core"

    @staticmethod
    def layer_expert_piece_id(index: int, expert: int) -> str:
        return f"layer_{index:03d}_expert_{expert}"

    def __post_init__(self) -> None:
        # id -> Piece index for O(1) piece_for / has_piece in the hot forward loop
        # (a production MoE manifest has ~12K pieces). pieces are never mutated
        # after construction, so this stays consistent.
        self._by_id = {p.piece_id: p for p in self.pieces}

    def has_piece(self, piece_id: str) -> bool:
        return piece_id in self._by_id

    def piece_for(self, piece_id: str) -> Piece:
        try:
            return self._by_id[piece_id]
        except KeyError:
            raise KeyError(f"unknown piece_id: {piece_id}")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        raw = json.loads(Path(path).read_text())
        pieces = [Piece(**p) for p in raw.pop("pieces")]
        return cls(pieces=pieces, **raw)
