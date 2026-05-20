"""Shared vocabulary and defaults for the ODPlatform data pipeline."""

from __future__ import annotations

from typing import Final


TASK_DETECT: Final[str] = "detect"
TASK_SEGMENT: Final[str] = "segment"

FORMAT_PASCAL_VOC: Final[str] = "pascal_voc"
FORMAT_COCO: Final[str] = "coco"
FORMAT_YOLO: Final[str] = "yolo"

SPLIT_TRAIN: Final[str] = "train"
SPLIT_VAL: Final[str] = "val"
SPLIT_TEST: Final[str] = "test"

IMAGES_DIRNAME: Final[str] = "images"
LABELS_DIRNAME: Final[str] = "labels"
ANNOTATIONS_DIRNAME: Final[str] = "annotations"

ODP_META_KEY: Final[str] = "odp_meta"
SCHEMA_VERSION: Final[int] = 1

DEFAULT_TRAIN_RATE: Final[float] = 0.8
DEFAULT_VAL_RATE: Final[float] = 0.1
DEFAULT_TEST_RATE: Final[float] = 0.1
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_MIN_COVERAGE: Final[float] = 0.5
RATE_EPSILON: Final[float] = 1e-8

SUPPORTED_TASKS: Final[tuple[str, ...]] = (
    TASK_DETECT,
    TASK_SEGMENT,
)

SUPPORTED_DATASET_FORMATS: Final[tuple[str, ...]] = (
    FORMAT_PASCAL_VOC,
    FORMAT_COCO,
    FORMAT_YOLO,
)

SUPPORTED_SPLITS: Final[tuple[str, ...]] = (
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
)


__all__ = [
    "ANNOTATIONS_DIRNAME",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_TEST_RATE",
    "DEFAULT_TRAIN_RATE",
    "DEFAULT_VAL_RATE",
    "FORMAT_COCO",
    "FORMAT_PASCAL_VOC",
    "FORMAT_YOLO",
    "IMAGES_DIRNAME",
    "LABELS_DIRNAME",
    "ODP_META_KEY",
    "RATE_EPSILON",
    "SCHEMA_VERSION",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SUPPORTED_DATASET_FORMATS",
    "SUPPORTED_SPLITS",
    "SUPPORTED_TASKS",
    "TASK_DETECT",
    "TASK_SEGMENT",
]
