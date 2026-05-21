"""Immutable best-effort dataset snapshot used by validation checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from odp_platform.common.constants import (
    IMAGE_EXTENSIONS,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    ODP_META_KEY,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TASK_DETECT,
    TASK_SEGMENT,
)
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.performance_utils import time_it


logger = get_logger(__name__)
_SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


@dataclass(frozen=True)
class SplitStats:
    image_count: int
    annotated_count: int
    total_instances: int


@dataclass(frozen=True)
class DatasetSnapshot:
    yaml_path: Path
    yaml_data: dict[str, Any]
    yaml_load_error: str | None
    data_root: Path
    nc: int | None
    class_names: tuple[str, ...]
    task_type: str
    images_per_split: dict[str, tuple[Path, ...]]
    labels_per_split: dict[str, tuple[Path, ...]]
    stats_per_split: dict[str, SplitStats]
    scan_warnings: tuple[str, ...]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(split for split in _SPLIT_ORDER if split in self.stats_per_split)

    @property
    def total_images(self) -> int:
        return sum(stats.image_count for stats in self.stats_per_split.values())


def _parse_yaml(yaml_path: Path) -> tuple[dict[str, Any], str | None, list[str]]:
    warnings: list[str] = []
    if not yaml_path.exists():
        return {}, f"YAML file does not exist: {yaml_path}", warnings

    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, str(exc), warnings

    if payload is None:
        return {}, None, warnings
    if not isinstance(payload, dict):
        warnings.append(f"YAML top-level is {type(payload).__name__}, not dict; snapshot scan skipped.")
        return {}, None, warnings
    return payload, None, warnings


def _resolve_task_type(yaml_data: Mapping[str, Any], task_type: str | None, warnings: list[str]) -> str:
    if task_type is not None:
        candidate = task_type.strip()
    else:
        odp_meta = yaml_data.get(ODP_META_KEY, {})
        candidate = str(odp_meta.get("task", TASK_DETECT)).strip() if isinstance(odp_meta, dict) else TASK_DETECT

    if candidate not in {TASK_DETECT, TASK_SEGMENT}:
        warnings.append(f"Unsupported task type {candidate!r}; falling back to {TASK_DETECT!r}.")
        return TASK_DETECT
    return candidate


def _resolve_data_root(yaml_path: Path, yaml_data: Mapping[str, Any], warnings: list[str]) -> Path:
    raw_path = yaml_data.get("path")
    if raw_path is None:
        warnings.append("YAML path field is missing; falling back to the yaml parent directory.")
        return yaml_path.parent.resolve()

    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = (yaml_path.parent / candidate).resolve()
    return candidate


def _parse_nc(yaml_data: Mapping[str, Any]) -> int | None:
    raw_nc = yaml_data.get("nc")
    if isinstance(raw_nc, bool):
        return None
    if isinstance(raw_nc, int) and raw_nc > 0:
        return raw_nc
    return None


def _parse_class_names(yaml_data: Mapping[str, Any]) -> tuple[str, ...]:
    raw_names = yaml_data.get("names")
    if isinstance(raw_names, list):
        names = [str(item).strip() for item in raw_names if str(item).strip()]
        return tuple(names)
    if isinstance(raw_names, dict):
        pairs: list[tuple[int, str]] = []
        for raw_key, raw_value in raw_names.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            value = str(raw_value).strip()
            if value:
                pairs.append((key, value))
        pairs.sort(key=lambda item: item[0])
        return tuple(value for _, value in pairs)
    return ()


def _resolve_split_dirs(data_root: Path, split_value: object) -> tuple[Path, Path] | None:
    split_path = Path(str(split_value)).expanduser()
    if not split_path.is_absolute():
        split_path = (data_root / split_path).resolve()

    if split_path.name.lower() == IMAGES_DIRNAME:
        return split_path, split_path.parent / LABELS_DIRNAME
    return split_path / IMAGES_DIRNAME, split_path / LABELS_DIRNAME


def _list_images(images_dir: Path) -> tuple[Path, ...]:
    image_exts_lower = {extension.lower() for extension in IMAGE_EXTENSIONS}
    items = {
        candidate.resolve()
        for candidate in images_dir.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in image_exts_lower
    }
    return tuple(sorted(items))


def _list_labels(labels_dir: Path) -> tuple[Path, ...]:
    items = {
        candidate.resolve()
        for candidate in labels_dir.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".txt"
    }
    return tuple(sorted(items))


def _count_label_lines(label_path: Path, warnings: list[str]) -> int:
    try:
        return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError as exc:
        warnings.append(f"Failed to read label file {label_path}: {exc}")
        return 0


def _scan_split(
    *,
    split: str,
    data_root: Path,
    yaml_data: Mapping[str, Any],
    warnings: list[str],
) -> tuple[tuple[Path, ...], tuple[Path, ...], SplitStats] | None:
    raw_split_value = yaml_data.get(split)
    if raw_split_value is None:
        return None

    resolved_dirs = _resolve_split_dirs(data_root, raw_split_value)
    if resolved_dirs is None:
        return None
    images_dir, labels_dir = resolved_dirs

    if not images_dir.exists():
        warnings.append(f"{split} images directory does not exist: {images_dir}")
        images = ()
    else:
        try:
            images = _list_images(images_dir)
        except OSError as exc:
            warnings.append(f"Failed to scan {split} images directory {images_dir}: {exc}")
            images = ()

    if not labels_dir.exists():
        warnings.append(f"{split} labels directory does not exist: {labels_dir}")
        labels = ()
    else:
        try:
            labels = _list_labels(labels_dir)
        except OSError as exc:
            warnings.append(f"Failed to scan {split} labels directory {labels_dir}: {exc}")
            labels = ()

    image_stems = {path.stem for path in images}
    annotated_count = 0
    total_instances = 0
    for label_path in labels:
        line_count = _count_label_lines(label_path, warnings)
        if line_count > 0:
            total_instances += line_count
            if label_path.stem in image_stems:
                annotated_count += 1

    return images, labels, SplitStats(
        image_count=len(images),
        annotated_count=annotated_count,
        total_instances=total_instances,
    )


@time_it(name="build_snapshot", logger_instance=logger)
def build_snapshot(yaml_path: Path, task_type: str | None = None) -> DatasetSnapshot:
    """Build one immutable dataset snapshot without raising business exceptions."""

    yaml_path = Path(yaml_path)
    yaml_data, yaml_load_error, warnings = _parse_yaml(yaml_path)
    resolved_task_type = _resolve_task_type(yaml_data, task_type, warnings)
    data_root = _resolve_data_root(yaml_path, yaml_data, warnings)
    nc = _parse_nc(yaml_data)
    class_names = _parse_class_names(yaml_data)

    images_per_split: dict[str, tuple[Path, ...]] = {}
    labels_per_split: dict[str, tuple[Path, ...]] = {}
    stats_per_split: dict[str, SplitStats] = {}

    for split in _SPLIT_ORDER:
        scanned = _scan_split(split=split, data_root=data_root, yaml_data=yaml_data, warnings=warnings)
        if scanned is None:
            continue
        images, labels, stats = scanned
        images_per_split[split] = images
        labels_per_split[split] = labels
        stats_per_split[split] = stats

    return DatasetSnapshot(
        yaml_path=yaml_path,
        yaml_data=yaml_data,
        yaml_load_error=yaml_load_error,
        data_root=data_root,
        nc=nc,
        class_names=class_names,
        task_type=resolved_task_type,
        images_per_split=images_per_split,
        labels_per_split=labels_per_split,
        stats_per_split=stats_per_split,
        scan_warnings=tuple(warnings),
    )


__all__ = [
    "DatasetSnapshot",
    "SplitStats",
    "build_snapshot",
]
