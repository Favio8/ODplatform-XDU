"""COCO to YOLO conversion helpers."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import os
from pathlib import Path
from typing import Final

from odp_platform.common.constants import FORMAT_COCO, TASK_DETECT, TASK_SEGMENT
from odp_platform.common.paths import YOLO_DATASETS_DIR, raw_dataset_root
from odp_platform.data_pipeline.registry import ConvertOptions
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample


SUPPORTED_SOURCE_FORMAT: Final[str] = FORMAT_COCO
SUPPORTED_TASKS: Final[tuple[str, ...]] = (TASK_DETECT, TASK_SEGMENT)
logger = logging.getLogger(__name__)
_ROBOFLOW_SPLITS: Final[tuple[str, ...]] = ("train", "valid", "val", "test")


def _windows_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _path_exists(path: Path) -> bool:
    if path.exists():
        return True
    return os.path.exists(_windows_long_path(path))


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


def _detect_images_dir(source_root: Path, annotation_file: Path | None = None) -> Path:
    if annotation_file is not None and annotation_file.parent != source_root:
        return annotation_file.parent

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

    split_matches: list[Path] = []
    for split_name in _ROBOFLOW_SPLITS:
        split_dir = source_root / split_name
        if split_dir.exists():
            split_matches.extend(sorted(split_dir.glob("*.json")))
    if split_matches:
        return split_matches[0]

    raise FileNotFoundError(f"No COCO annotation file found under {source_root}")


def _detect_annotation_files(source_root: Path, dataset_name: str) -> list[Path]:
    try:
        primary = _detect_annotation_file(source_root, dataset_name)
    except FileNotFoundError:
        primary = None

    split_matches: list[Path] = []
    for split_name in _ROBOFLOW_SPLITS:
        split_dir = source_root / split_name
        if split_dir.exists():
            split_matches.extend(sorted(split_dir.glob("*.json")))

    if split_matches:
        return split_matches
    if primary is not None:
        return [primary]
    raise FileNotFoundError(f"No COCO annotation file found under {source_root}")


def _normalize_polygon(points: list[float], image_width: int, image_height: int) -> list[float]:
    normalized: list[float] = []
    for index, value in enumerate(points):
        ratio = value / image_width if index % 2 == 0 else value / image_height
        normalized.append(_clamp_unit_interval(ratio))
    return normalized


def _bbox_to_yolo_line(class_id: int, bbox: list[float], image_width: int, image_height: int) -> str:
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return (
        f"{class_id} "
        f"{_clamp_unit_interval(x_center / image_width):.6f} "
        f"{_clamp_unit_interval(y_center / image_height):.6f} "
        f"{_clamp_unit_interval(width / image_width):.6f} "
        f"{_clamp_unit_interval(height / image_height):.6f}"
    )


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _min_index(segment_a: list[tuple[float, float]], segment_b: list[tuple[float, float]]) -> tuple[int, int]:
    best_i = 0
    best_j = 0
    best_distance = float("inf")

    for index_a, point_a in enumerate(segment_a):
        for index_b, point_b in enumerate(segment_b):
            dx = point_a[0] - point_b[0]
            dy = point_a[1] - point_b[1]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_i = index_a
                best_j = index_b

    return best_i, best_j


def _to_points(flat_polygon: list[float]) -> list[tuple[float, float]]:
    return [
        (flat_polygon[index], flat_polygon[index + 1])
        for index in range(0, len(flat_polygon), 2)
    ]


def _rotate_points(points: list[tuple[float, float]], start_index: int) -> list[tuple[float, float]]:
    return points[start_index:] + points[:start_index]


def _merge_multi_segment(segments: list[list[float]]) -> list[list[tuple[float, float]]]:
    merged: list[list[tuple[float, float]]] = []
    segment_points = [_to_points(segment) for segment in segments]
    index_links: list[list[int]] = [[] for _ in segment_points]

    for index in range(1, len(segment_points)):
        prev_index, curr_index = _min_index(segment_points[index - 1], segment_points[index])
        index_links[index - 1].append(prev_index)
        index_links[index].append(curr_index)

    for round_index in range(2):
        if round_index == 0:
            for segment_index, links in enumerate(index_links):
                points = segment_points[segment_index]
                if len(links) == 2 and links[0] > links[1]:
                    links = links[::-1]
                    points = list(reversed(points))

                points = _rotate_points(points, links[0])
                points = points + points[:1]
                segment_points[segment_index] = points

                if segment_index in {0, len(index_links) - 1}:
                    merged.append(points)
                else:
                    merged.append(points[: links[1] - links[0] + 1])
        else:
            for segment_index in range(len(index_links) - 1, -1, -1):
                if segment_index in {0, len(index_links) - 1}:
                    continue

                links = index_links[segment_index]
                points = segment_points[segment_index]
                merged.append(points[abs(links[1] - links[0]) :])

    return merged


def _extract_polygons(segmentation: object) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []

    raw_polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
    polygons: list[list[float]] = []
    for raw_polygon in raw_polygons:
        try:
            polygon = [float(value) for value in raw_polygon]
        except (TypeError, ValueError):
            continue

        if len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        polygons.append(polygon)

    return polygons


def _segment_to_yolo_line(
    class_id: int,
    segmentation: object,
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> str:
    polygons = _extract_polygons(segmentation)
    if not polygons:
        return _bbox_to_yolo_line(class_id, bbox, image_width, image_height)

    if len(polygons) == 1:
        merged_points = polygons[0]
    else:
        merged_segments = _merge_multi_segment(polygons)
        merged_points = [
            coordinate
            for segment in merged_segments
            for point in segment
            for coordinate in point
        ]
        logger.debug("Merged %s polygons into one YOLO segment line", len(polygons))

    normalized = _normalize_polygon(merged_points, image_width, image_height)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized)


def convert(
    *,
    options: ConvertOptions,
    source_root: Path | None = None,
    output_labels_dir: Path | None = None,
) -> ConversionManifest:
    """Convert a COCO dataset into YOLO labels."""

    source_root = _resolve_source_root(options, source_root)
    annotation_files = _detect_annotation_files(source_root, options.dataset_name)
    output_labels_dir = _resolve_output_labels_dir(options, output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    payloads = [json.loads(annotation_file.read_text(encoding="utf-8")) for annotation_file in annotation_files]
    categories = [
        category
        for payload in payloads
        for category in payload.get("categories", [])
    ]

    if options.classes is not None:
        classes = list(options.classes)
    else:
        classes = []
        for category in sorted(categories, key=lambda item: (str(item["name"]).lower(), int(item["id"]))):
            class_name = str(category["name"])
            if class_name not in classes:
                classes.append(class_name)

    class_to_id = {class_name: index for index, class_name in enumerate(classes)}

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir)
        samples: list[PreparedSample] = []

        for annotation_file, payload in zip(annotation_files, payloads):
            images_dir = _detect_images_dir(source_root, annotation_file)
            images = payload.get("images", [])
            annotations = payload.get("annotations", [])
            category_id_to_name = {
                int(category["id"]): str(category["name"])
                for category in payload.get("categories", [])
            }

            annotations_by_image: dict[int, list[dict[str, object]]] = {}
            for annotation in annotations:
                annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

            for image in sorted(images, key=lambda item: str(item["file_name"])):
                image_id = int(image["id"])
                image_path = images_dir / str(image["file_name"])
                if not _path_exists(image_path):
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

                split_prefix = annotation_file.parent.name if annotation_file.parent != source_root else ""
                label_stem = f"{split_prefix}_{stem}" if split_prefix else stem
                staging_label_path = staging_dir / f"{label_stem}.txt"
                staging_label_path.write_text("\n".join(lines), encoding="utf-8")
                final_label_path = output_labels_dir / staging_label_path.name
                shutil.copy2(staging_label_path, final_label_path)
                samples.append(
                    PreparedSample(
                        stem=label_stem,
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
