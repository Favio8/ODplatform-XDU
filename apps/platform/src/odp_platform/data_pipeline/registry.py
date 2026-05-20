"""Converter registry and shared option objects for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from odp_platform.common.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_RATE,
    DEFAULT_TRAIN_RATE,
    DEFAULT_VAL_RATE,
    FORMAT_COCO,
    FORMAT_PASCAL_VOC,
    FORMAT_YOLO,
    RATE_EPSILON,
    TASK_DETECT,
)


SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset(
    {
        FORMAT_PASCAL_VOC,
        FORMAT_COCO,
        FORMAT_YOLO,
    }
)

_REGISTRY: dict[str, Any] = {}
_CAPABILITIES: dict[str, tuple[str, ...]] = {}
_INITIALIZED = False


def _normalize_source_format(source_format: str) -> str:
    normalized = source_format.strip().lower()
    if not normalized:
        raise ValueError("source_format must not be empty")
    return normalized


def _normalize_capabilities(capabilities: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(task.strip().lower() for task in capabilities if task.strip())
    if not normalized:
        raise ValueError("capabilities must not be empty")
    return normalized


@dataclass(slots=True)
class ConvertOptions:
    """Common conversion options shared by all source formats."""

    dataset_name: str
    source_format: str
    task: str = TASK_DETECT
    classes: list[str] | None = None
    train_rate: float = DEFAULT_TRAIN_RATE
    val_rate: float = DEFAULT_VAL_RATE
    test_rate: float = DEFAULT_TEST_RATE
    random_state: int = DEFAULT_RANDOM_STATE
    source_root: Path | None = None
    output_root: Path | None = None

    def __post_init__(self) -> None:
        self.dataset_name = self.dataset_name.strip()
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")

        self.source_format = _normalize_source_format(self.source_format)
        if self.source_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported source format: {self.source_format!r}")

        self.task = self.task.strip().lower()
        if not self.task:
            raise ValueError("task must not be empty")

        total_rate = self.train_rate + self.val_rate + self.test_rate
        if abs(total_rate - 1.0) > RATE_EPSILON:
            raise ValueError("train_rate + val_rate + test_rate must equal 1.0")

        if self.classes is not None:
            normalized_classes = [item.strip() for item in self.classes if item.strip()]
            self.classes = normalized_classes or None

        if self.source_root is not None:
            self.source_root = Path(self.source_root)
        if self.output_root is not None:
            self.output_root = Path(self.output_root)


def register_converter(
    source_format: str,
    converter: Any,
    *,
    capabilities: tuple[str, ...] | list[str],
) -> None:
    """Register one converter module or object for a source format."""

    normalized_format = _normalize_source_format(source_format)
    normalized_capabilities = _normalize_capabilities(capabilities)

    if normalized_format in _REGISTRY:
        raise ValueError(f"Converter already registered for {normalized_format!r}")

    _REGISTRY[normalized_format] = converter
    _CAPABILITIES[normalized_format] = normalized_capabilities


def _lazy_init() -> None:
    """Import converter modules only when the registry is first used."""

    global _INITIALIZED
    if _INITIALIZED:
        return

    from odp_platform.data_pipeline.core import coco, pascal_voc, yolo

    register_converter(
        pascal_voc.SUPPORTED_SOURCE_FORMAT,
        pascal_voc,
        capabilities=pascal_voc.SUPPORTED_TASKS,
    )
    register_converter(
        coco.SUPPORTED_SOURCE_FORMAT,
        coco,
        capabilities=coco.SUPPORTED_TASKS,
    )
    register_converter(
        yolo.SUPPORTED_SOURCE_FORMAT,
        yolo,
        capabilities=yolo.SUPPORTED_TASKS,
    )
    _INITIALIZED = True


def get_converter(source_format: str) -> Any:
    """Return the converter registered for the requested source format."""

    _lazy_init()
    normalized_format = _normalize_source_format(source_format)
    try:
        return _REGISTRY[normalized_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported source format: {normalized_format!r}") from exc


def list_capabilities() -> dict[str, tuple[str, ...]]:
    """Return a snapshot of format-to-capabilities mappings."""

    _lazy_init()
    return dict(_CAPABILITIES)


__all__ = [
    "ConvertOptions",
    "get_converter",
    "list_capabilities",
    "register_converter",
]
