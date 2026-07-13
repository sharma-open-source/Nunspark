from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .manifest import Manifest


class PieceStore:
    """Reads a piece's weights from SSD into unified memory as MLX arrays.

    Piece-agnostic: it only knows piece_id -> file. This is the seam where
    MoE / sub-layer pieces plug in later with no change to callers.
    """

    def __init__(self, root: str | Path, manifest: Manifest):
        self.root = Path(root)
        self.manifest = manifest

    def load(self, piece_id: str) -> dict[str, mx.array]:
        piece = self.manifest.piece_for(piece_id)
        return mx.load(str(self.root / piece.file))

    def path_for(self, piece_id: str) -> Path:
        """Filesystem path of a piece's file — the seam the I/O warmer reads
        through (it raw-reads this file to warm the page cache before load())."""
        return self.root / self.manifest.piece_for(piece_id).file
