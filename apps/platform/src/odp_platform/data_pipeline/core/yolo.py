"""YOLO-format dataset passthrough helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final
import shutil

import yaml

from odp_platform.common.constants import FORMAT_YOLO, IMAGE_EXTENSIONS, TASK_DETECT
from odp_platform.common.paths import YOLO_DATASETS_DIR, raw_dataset_root
from odp_platform.data_pipeline.registry import ConvertOptions
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample


SUPPORTED_SOURCE_FORMAT: Final[str] = FORMAT_YOLO
SUPPORTED_TASKS: Final[tuple[str, ...]] = (TASK_DETECT,)
logger = logging.getLogger(__name__)


def _resolve_source_root(options: ConvertOptions, source_root: Path | None) -> Path:
    if source_root is not None:
        return Path(source_root)
    if options.source_root is not None:
        return options.source_root
    return raw_dataset_root(options.dataset_name)


def _resolve_output_labels_dir(options: ConvertOptions, output_labels_dir: Path | None) -> Path:
    if output_labels_dir is not None:
        return Path(output_labels_dir)
    if options.output_root is not None:
        return options.output_root / "labels"
    return YOLO_DATASETS_DIR / options.dataset_name / "labels"


def _detect_images_dir(source_root: Path) -> Path:
    images_dir = source_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"YOLO images directory not found: {images_dir}")
    return images_dir


def _detect_labels_dir(source_root: Path) -> Path:
    labels_dir = source_root / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"YOLO labels directory not found: {labels_dir}")
    return labels_dir


def _load_classes(source_root: Path, user_classes: list[str] | None) -> list[str]:
    if user_classes is not None:
        return list(user_classes)

    for yaml_name in ("data.yaml", "dataset.yaml", f"{source_root.name}.yaml"):
        yaml_path = source_root / yaml_name
        if not yaml_path.exists():
            continue
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = payload.get("names")
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        if isinstance(names, list):
            return [str(item) for item in names]

    max_class_id = -1
    labels_dir = _detect_labels_dir(source_root)
    for label_path in labels_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except (TypeError, ValueError):
                logger.debug("%s has an invalid class id in one line while discovering classes", label_path.name)
                continue
            max_class_id = max(max_class_id, class_id)

    return [f"class_{index}" for index in range(max_class_id + 1)] if max_class_id >= 0 else []


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    image_exts_lower = {suffix.lower() for suffix in IMAGE_EXTENSIONS}
    image_index: dict[str, Path] = {}
    for candidate in sorted(images_dir.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in image_exts_lower:
            continue
        image_index.setdefault(candidate.stem, candidate)
    return image_index


def _find_image_path(image_index: dict[str, Path], images_dir: Path, stem: str) -> Path:
    if stem in image_index:
        return image_index[stem]
    raise FileNotFoundError(f"No YOLO image file found for {stem!r} under {images_dir}")


def _validate_yolo_content(content: str, n_classes: int, file_name: str) -> bool:
    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            logger.debug("%s:%s has %s columns instead of 5", file_name, line_no, len(parts))
            return False

        try:
            class_id = int(float(parts[0]))
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            logger.debug("%s:%s contains non-numeric YOLO values", file_name, line_no)
            return False

        if not 0 <= class_id < n_classes:
            logger.debug("%s:%s has class id %s outside [0, %s)", file_name, line_no, class_id, n_classes)
            return False
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            logger.debug("%s:%s contains coordinates outside [0, 1]", file_name, line_no)
            return False

    return True


def _supports_hardlink(src_dir: Path, dst_dir: Path) -> bool:
    try:
        return src_dir.stat().st_dev == dst_dir.stat().st_dev
    except OSError:
        return False


def _materialize_label(source_label_path: Path, target_label_path: Path, *, prefer_hardlink: bool) -> None:
    if target_label_path.exists():
        target_label_path.unlink()

    if prefer_hardlink:
        try:
            os.link(source_label_path, target_label_path)
            return
        except OSError:
            logger.debug("Hardlink failed for %s; falling back to copy", source_label_path.name)

    shutil.copy2(source_label_path, target_label_path)


def convert(
    *,
    options: ConvertOptions,
    source_root: Path | None = None,
    output_labels_dir: Path | None = None,
) -> ConversionManifest:
    """Normalize an already-YOLO dataset into the standard manifest shape."""

    source_root = _resolve_source_root(options, source_root)
    images_dir = _detect_images_dir(source_root)
    labels_dir = _detect_labels_dir(source_root)
    classes = _load_classes(source_root, options.classes)
    image_index = _build_image_index(images_dir)
    output_labels_dir = _resolve_output_labels_dir(options, output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    prefer_hardlink = _supports_hardlink(labels_dir, output_labels_dir)

    samples: list[PreparedSample] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem = label_file.stem
        try:
            image_path = _find_image_path(image_index, images_dir, stem)
        except FileNotFoundError as exc:
            logger.warning("%s skipped because its image is unavailable: %s", label_file.name, exc)
            continue

        content = label_file.read_text(encoding="utf-8")
        if not _validate_yolo_content(content, len(classes), label_file.name):
            logger.warning("YOLO label file is invalid and will be skipped: %s", label_file.name)
            continue

        final_label_path = output_labels_dir / label_file.name
        _materialize_label(label_file, final_label_path, prefer_hardlink=prefer_hardlink)

        sample_classes: list[str] = []
        for line in content.splitlines():
            if not line.strip():
                continue
            class_id = int(float(line.split()[0]))
            if 0 <= class_id < len(classes):
                sample_classes.append(classes[class_id])

        samples.append(
            PreparedSample(
                stem=stem,
                image_path=image_path,
                label_path=final_label_path,
                class_names=tuple(sample_classes),
            )
        )

    return ConversionManifest(
        dataset_name=options.dataset_name,
        source_format=SUPPORTED_SOURCE_FORMAT,
        task=options.task,
        classes=classes,
        samples=samples,
        source_root=source_root,
        labels_root=output_labels_dir,
    )


__all__ = [
    "SUPPORTED_SOURCE_FORMAT",
    "SUPPORTED_TASKS",
    "convert",
]
