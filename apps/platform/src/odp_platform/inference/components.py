"""Reusable inference data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class InferenceArtifact:
    """One saved artifact produced during inference."""

    input_name: str
    output_path: Path
    kind: str


@dataclass(frozen=True)
class FramePrediction:
    """Summary of one processed frame."""

    frame_index: int
    input_name: str
    detections: int
    output_path: Path | None = None
    inference_ms: float | None = None


@dataclass(frozen=True)
class InferenceSummary:
    """Aggregate summary for one inference run."""

    frames_processed: int
    detections_total: int
    source: str
    artifacts: tuple[InferenceArtifact, ...] = field(default_factory=tuple)


__all__ = [
    "FramePrediction",
    "InferenceArtifact",
    "InferenceSummary",
]
