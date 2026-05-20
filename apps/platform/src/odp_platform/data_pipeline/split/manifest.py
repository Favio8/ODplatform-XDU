"""Shared manifest objects exchanged across the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PreparedSample:
    """One image/label pair ready for splitting or materialization."""

    stem: str
    image_path: Path
    label_path: Path
    class_names: tuple[str, ...]


@dataclass(slots=True)
class ConversionManifest:
    """Dataset-level conversion summary used by downstream pipeline steps."""

    dataset_name: str
    source_format: str
    task: str
    classes: list[str]
    samples: list[PreparedSample]
    source_root: Path
    labels_root: Path


__all__ = [
    "ConversionManifest",
    "PreparedSample",
]
