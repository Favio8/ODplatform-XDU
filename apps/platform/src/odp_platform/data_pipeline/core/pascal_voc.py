"""Pascal VOC to YOLO conversion helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from odp_platform.common.constants import FORMAT_PASCAL_VOC, IMAGE_EXTENSIONS, TASK_DETECT
from odp_platform.common.paths import YOLO_DATASETS_DIR, raw_dataset_root
from odp_platform.data_pipeline.registry import ConvertOptions
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample


SUPPORTED_SOURCE_FORMAT: Final[str] = FORMAT_PASCAL_VOC
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
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return images_dir


def _detect_annotations_dir(source_root: Path) -> Path:
    annotations_dir = source_root / "annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    return annotations_dir


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
    raise FileNotFoundError(f"No image file found for {stem!r} under {images_dir}")


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _parse_objects(annotation_path: Path) -> list[tuple[str, tuple[float, float, float, float]]] | None:
    try:
        root = ElementTree.parse(annotation_path).getroot()
    except ElementTree.ParseError as exc:
        logger.warning("%s XML is malformed: %s; skipped", annotation_path.name, exc)
        return None

    objects: list[tuple[str, tuple[float, float, float, float]]] = []

    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if not class_name:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
        except (TypeError, ValueError):
            logger.debug("%s has one object with invalid bbox values; skipped object", annotation_path.name)
            continue
        objects.append((class_name, (xmin, ymin, xmax, ymax)))

    return objects


def _collect_classes(annotation_paths: list[Path], user_classes: list[str] | None) -> list[str]:
    if user_classes:
        return list(user_classes)

    discovered: set[str] = set()
    for annotation_path in annotation_paths:
        objects = _parse_objects(annotation_path)
        if objects is None:
            continue
        for class_name, _ in objects:
            discovered.add(class_name)
    return sorted(discovered)


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clip_bbox_to_image(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox
    clipped_xmin = min(max(xmin, 0.0), float(image_width))
    clipped_ymin = min(max(ymin, 0.0), float(image_height))
    clipped_xmax = min(max(xmax, 0.0), float(image_width))
    clipped_ymax = min(max(ymax, 0.0), float(image_height))

    if clipped_xmax <= clipped_xmin or clipped_ymax <= clipped_ymin:
        return None
    return clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax


def _to_yolo_bbox(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    clipped_bbox = _clip_bbox_to_image(
        bbox,
        image_width=image_width,
        image_height=image_height,
    )
    if clipped_bbox is None:
        return None

    xmin, ymin, xmax, ymax = clipped_bbox
    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + width / 2.0
    y_center = ymin + height / 2.0
    return (
        _clamp_unit_interval(x_center / image_width),
        _clamp_unit_interval(y_center / image_height),
        _clamp_unit_interval(width / image_width),
        _clamp_unit_interval(height / image_height),
    )


def convert(
    *,
    options: ConvertOptions,
    source_root: Path | None = None,
    output_labels_dir: Path | None = None,
) -> ConversionManifest:
    """Convert one Pascal VOC dataset into flat YOLO label files."""

    source_root = _resolve_source_root(options, source_root)
    images_dir = _detect_images_dir(source_root)
    annotations_dir = _detect_annotations_dir(source_root)

    annotation_paths = sorted(annotations_dir.glob("*.xml"))
    if not annotation_paths:
        raise FileNotFoundError(f"No Pascal VOC xml annotations found under {annotations_dir}")

    classes = _collect_classes(annotation_paths, options.classes)
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    image_index = _build_image_index(images_dir)

    output_labels_dir = _resolve_output_labels_dir(options, output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    samples: list[PreparedSample] = []
    for annotation_path in annotation_paths:
        stem = annotation_path.stem
        objects = _parse_objects(annotation_path)
        if objects is None:
            continue

        try:
            image_path = _find_image_path(image_index, images_dir, stem)
            image_width, image_height = _load_image_size(image_path)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("%s skipped because its image is unavailable: %s", annotation_path.name, exc)
            continue

        if image_width <= 0 or image_height <= 0:
            logger.warning("%s skipped because image size is invalid: %sx%s", annotation_path.name, image_width, image_height)
            continue

        lines: list[str] = []
        sample_classes: list[str] = []
        for class_name, bbox in objects:
            if class_name not in class_to_id:
                logger.debug("%s contains class %r outside the allowed class list; skipped object", annotation_path.name, class_name)
                continue

            yolo_bbox = _to_yolo_bbox(
                bbox,
                image_width=image_width,
                image_height=image_height,
            )
            if yolo_bbox is None:
                logger.debug("%s has one bbox completely outside image bounds; skipped object", annotation_path.name)
                continue
            x_center, y_center, width, height = yolo_bbox
            lines.append(
                f"{class_to_id[class_name]} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            sample_classes.append(class_name)

        label_path = output_labels_dir / f"{stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        samples.append(
            PreparedSample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
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
