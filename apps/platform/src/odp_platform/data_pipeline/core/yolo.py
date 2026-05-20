"""YOLO-format dataset passthrough helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from odp_platform.common.constants import FORMAT_YOLO, TASK_DETECT
from odp_platform.common.paths import RAW_DATASETS_DIR, YOLO_DATASETS_DIR
from odp_platform.data_pipeline.registry import ConvertOptions
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample


SUPPORTED_SOURCE_FORMAT: Final[str] = FORMAT_YOLO
SUPPORTED_TASKS: Final[tuple[str, ...]] = (TASK_DETECT,)
_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _resolve_source_root(options: ConvertOptions, source_root: Path | None) -> Path:
    if source_root is not None:
        return Path(source_root)
    if options.source_root is not None:
        return options.source_root
    return RAW_DATASETS_DIR / options.dataset_name


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
            class_id = int(float(line.split()[0]))
            max_class_id = max(max_class_id, class_id)

    return [f"class_{index}" for index in range(max_class_id + 1)] if max_class_id >= 0 else []


def _find_image_path(images_dir: Path, stem: str) -> Path:
    for suffix in _IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No YOLO image file found for {stem!r} under {images_dir}")


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
    output_labels_dir = _resolve_output_labels_dir(options, output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    samples: list[PreparedSample] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem = label_file.stem
        image_path = _find_image_path(images_dir, stem)
        final_label_path = output_labels_dir / label_file.name
        content = label_file.read_text(encoding="utf-8")
        final_label_path.write_text(content, encoding="utf-8")

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
