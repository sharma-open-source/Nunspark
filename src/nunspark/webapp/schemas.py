from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


Preset = Literal["lossless", "fast"]

# NOTE: keys must match the Preset Literal values above.
_PRESETS: dict[str, tuple[int, int]] = {
    # preset -> (accept_top_k, num_draft_tokens)
    "lossless": (1, 16),
    "fast": (3, 24),
}


def preset_params(
    preset: str,
    accept_top_k: int | None = None,
    num_draft_tokens: int | None = None,
) -> tuple[int, int]:
    """Resolve (accept_top_k, num_draft_tokens) from a preset name, with
    optional advanced overrides taking precedence over the preset value."""
    base_k, base_n = _PRESETS[preset]
    return (
        accept_top_k if accept_top_k is not None else base_k,
        num_draft_tokens if num_draft_tokens is not None else base_n,
    )


class Advanced(BaseModel):
    kv_bits: int | None = None
    kv_group_size: int = 64
    budget: str = "4GB"
    accept_top_k: int | None = None
    num_draft_tokens: int | None = None


class BatchRequest(BaseModel):
    model: str                              # absolute path to a packed dir
    draft: str | None = None                # path to a draft model, or None
    preset: Preset = "lossless"
    instruction: str = ""
    use_chat_template: bool = True
    max_tokens: int = 2048
    temperature: float = 0.0
    output_dir: str
    file_ids: list[str] = Field(default_factory=list)
    advanced: Advanced = Field(default_factory=Advanced)


@dataclass
class Job:
    id: str
    batch_id: str
    file_name: str
    file_path: str
    status: JobStatus = JobStatus.QUEUED
    tokens_done: int = 0
    error: str | None = None
    output_path: str | None = None
    metrics: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    model: str = ""
    draft: str | None = None
    preset: str = "lossless"
    instruction: str = ""
    use_chat_template: bool = True
    max_tokens: int = 2048
    temperature: float = 0.0
    output_dir: str = ""
    advanced: dict = field(default_factory=dict)

    def public(self) -> dict:
        """JSON-serializable snapshot for API responses / SSE."""
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "file_name": self.file_name,
            "status": self.status.value,
            "tokens_done": self.tokens_done,
            "error": self.error,
            "output_path": self.output_path,
            "metrics": self.metrics,
        }
