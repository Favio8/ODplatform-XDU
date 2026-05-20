"""Pascal VOC to YOLO conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from odp_platform.common.constants import FORMAT_PASCAL_VOC, TASK_DETECT
from odp_platform.common.paths import RAW_DATASETS_DIR, RAW_DATA_DIR, YOLO_DATASETS_DIR
from odp_platform.data_pipeline.registry import ConvertOptions


SUPPORTED_SOURCE_FORMAT: Final[str] = FORMAT_PASCAL_VOC
SUPPORTED_TASKS: Final[tuple[str, ...]] = (TASK_DETECT,)
_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(slots=True)
class ConvertedSample:
    """One converted image/label pair."""

    image_path: Path
    label_path: Path
    class_names: tuple[str, ...]


@dataclass(slots=True)
class ConversionResult:
    """Output summary from a VOC conversion run."""

    dataset_name: str
    source_format: str
    classes: list[str]
    samples: list[ConvertedSample]


def _resolve_source_root(options: ConvertOptions, source_root: Path | None) -> Path:
    if source_root is not None:
        return Path(source_root)
    if options.source_root is not None:
        return options.source_root

    generic_root = RAW_DATASETS_DIR / options.dataset_name
    if generic_root.exists():
        return generic_root

    if options.dataset_name.lower() == "rsod" and RAW_DATA_DIR.exists():
        return RAW_DATA_DIR

    return generic_root


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


def _find_image_path(images_dir: Path, stem: str) -> Path:
    for suffix in _IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image file found for {stem!r} under {images_dir}")


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _parse_objects(annotation_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ElementTree.parse(annotation_path).getroot()
    objects: list[tuple[str, tuple[float, float, float, float]]] = []

    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if not class_name:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        objects.append((class_name, (xmin, ymin, xmax, ymax)))

    return objects


def _collect_classes(annotation_paths: list[Path], user_classes: list[str] | None) -> list[str]:
    if user_classes:
        return list(user_classes)

    discovered: set[str] = set()
    for annotation_path in annotation_paths:
        for class_name, _ in _parse_objects(annotation_path):
            discovered.add(class_name)
    return sorted(discovered)


def _to_yolo_bbox(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + width / 2.0
    y_center = ymin + height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def convert(
    *,
    options: ConvertOptions,
    source_root: Path | None = None,
    output_labels_dir: Path | None = None,
) -> ConversionResult:
    """Convert one Pascal VOC dataset into flat YOLO label files."""

    source_root = _resolve_source_root(options, source_root)
    images_dir = _detect_images_dir(source_root)
    annotations_dir = _detect_annotations_dir(source_root)

    annotation_paths = sorted(annotations_dir.glob("*.xml"))
    if not annotation_paths:
        raise FileNotFoundError(f"No Pascal VOC xml annotations found under {annotations_dir}")

    classes = _collect_classes(annotation_paths, options.classes)
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}

    output_labels_dir = _resolve_output_labels_dir(options, output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    samples: list[ConvertedSample] = []
    for annotation_path in annotation_paths:
        stem = annotation_path.stem
        image_path = _find_image_path(images_dir, stem)
        image_width, image_height = _load_image_size(image_path)

        lines: list[str] = []
        sample_classes: list[str] = []
        for class_name, bbox in _parse_objects(annotation_path):
            if class_name not in class_to_id:
                raise ValueError(
                    f"Class {class_name!r} from {annotation_path.name} is not present in the final class list"
                )

            x_center, y_center, width, height = _to_yolo_bbox(
                bbox,
                image_width=image_width,
                image_height=image_height,
            )
            lines.append(
                f"{class_to_id[class_name]} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            sample_classes.append(class_name)

        label_path = output_labels_dir / f"{stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        samples.append(
            ConvertedSample(
                image_path=image_path,
                label_path=label_path,
                class_names=tuple(sample_classes),
            )
        )

    return ConversionResult(
        dataset_name=options.dataset_name,
        source_format=SUPPORTED_SOURCE_FORMAT,
        classes=classes,
        samples=samples,
    )


__all__ = [
    "ConversionResult",
    "ConvertedSample",
    "SUPPORTED_SOURCE_FORMAT",
    "SUPPORTED_TASKS",
    "convert",
]
