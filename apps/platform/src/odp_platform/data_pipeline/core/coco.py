"""COCO to YOLO conversion helpers."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Final

from odp_platform.common.constants import FORMAT_COCO, TASK_DETECT, TASK_SEGMENT
from odp_platform.common.paths import RAW_DATASETS_DIR, YOLO_DATASETS_DIR
from odp_platform.data_pipeline.registry import ConvertOptions
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample


SUPPORTED_SOURCE_FORMAT: Final[str] = FORMAT_COCO
SUPPORTED_TASKS: Final[tuple[str, ...]] = (TASK_DETECT, TASK_SEGMENT)


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
    if images_dir.exists():
        return images_dir
    return source_root


def _detect_annotation_file(source_root: Path, dataset_name: str) -> Path:
    candidates = [
        source_root / "annotations.json",
        source_root / f"{dataset_name}.json",
        source_root / "annotations" / "instances.json",
        source_root / "annotations" / f"{dataset_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    annotations_dir = source_root / "annotations"
    if annotations_dir.exists():
        matches = sorted(annotations_dir.glob("*.json"))
        if matches:
            return matches[0]

    matches = sorted(source_root.glob("*.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No COCO annotation file found under {source_root}")


def _normalize_polygon(points: list[float], image_width: int, image_height: int) -> list[float]:
    normalized: list[float] = []
    for index, value in enumerate(points):
        normalized.append(value / image_width if index % 2 == 0 else value / image_height)
    return normalized


def _bbox_to_yolo_line(class_id: int, bbox: list[float], image_width: int, image_height: int) -> str:
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return (
        f"{class_id} "
        f"{x_center / image_width:.6f} {y_center / image_height:.6f} "
        f"{width / image_width:.6f} {height / image_height:.6f}"
    )


def _segment_to_yolo_line(
    class_id: int,
    segmentation: object,
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> str:
    polygon: list[float]
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        polygon = [float(value) for value in segmentation[0]]
    elif isinstance(segmentation, list):
        polygon = [float(value) for value in segmentation]
    else:
        polygon = []

    if len(polygon) < 6:
        return _bbox_to_yolo_line(class_id, bbox, image_width, image_height)

    normalized = _normalize_polygon(polygon, image_width, image_height)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized)


def convert(
    *,
    options: ConvertOptions,
    source_root: Path | None = None,
    output_labels_dir: Path | None = None,
) -> ConversionManifest:
    """Convert a COCO dataset into YOLO labels."""

    source_root = _resolve_source_root(options, source_root)
    images_dir = _detect_images_dir(source_root)
    annotation_file = _detect_annotation_file(source_root, options.dataset_name)
    output_labels_dir = _resolve_output_labels_dir(options, output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])

    if options.classes is not None:
        classes = list(options.classes)
    else:
        classes = [str(category["name"]) for category in sorted(categories, key=lambda item: item["id"])]

    category_id_to_name = {int(category["id"]): str(category["name"]) for category in categories}
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}

    annotations_by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir)
        samples: list[PreparedSample] = []

        for image in sorted(images, key=lambda item: str(item["file_name"])):
            image_id = int(image["id"])
            image_path = images_dir / str(image["file_name"])
            if not image_path.exists():
                raise FileNotFoundError(f"COCO image not found: {image_path}")

            image_width = int(image["width"])
            image_height = int(image["height"])
            stem = Path(str(image["file_name"])).stem
            lines: list[str] = []
            sample_classes: list[str] = []

            for annotation in annotations_by_image.get(image_id, []):
                class_name = category_id_to_name[int(annotation["category_id"])]
                if class_name not in class_to_id:
                    continue

                bbox = [float(value) for value in annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])]
                if options.task == TASK_SEGMENT:
                    line = _segment_to_yolo_line(
                        class_to_id[class_name],
                        annotation.get("segmentation", []),
                        bbox,
                        image_width,
                        image_height,
                    )
                else:
                    line = _bbox_to_yolo_line(class_to_id[class_name], bbox, image_width, image_height)
                lines.append(line)
                sample_classes.append(class_name)

            staging_label_path = staging_dir / f"{stem}.txt"
            staging_label_path.write_text("\n".join(lines), encoding="utf-8")
            final_label_path = output_labels_dir / staging_label_path.name
            shutil.copy2(staging_label_path, final_label_path)
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
