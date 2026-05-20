"""End-to-end orchestration for dataset preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from odp_platform.common.constants import (
    ANNOTATIONS_DIRNAME,
    DEFAULT_MIN_COVERAGE,
    IMAGES_DIRNAME,
    LABELS_DIRNAME,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from odp_platform.common.paths import (
    DATASET_CONFIGS_DIR,
    DATA_DIR,
    RAW_DATASETS_DIR,
    RAW_DATA_DIR,
    YOLO_DATA_DIR,
)
from odp_platform.data_pipeline.registry import ConvertOptions, get_converter
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample
from odp_platform.data_pipeline.split.materializer import materialize_splits
from odp_platform.data_pipeline.split.splitter import split_pairs
from odp_platform.data_pipeline.split.yaml_writer import write_dataset_yaml


@dataclass(slots=True)
class CoverageReport:
    """Coverage summary computed before conversion starts."""

    image_count: int
    annotation_count: int
    matched_count: int
    coverage: float


@dataclass(slots=True)
class OrchestrationResult:
    """Final output from the dataset preparation pipeline."""

    manifest: ConversionManifest
    split_map: dict[str, list[PreparedSample]]
    yaml_path: Path
    coverage: CoverageReport


_SPLITS: Final[tuple[str, ...]] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


def _resolve_source_root(options: ConvertOptions) -> Path:
    if options.source_root is not None:
        return options.source_root

    if options.source_format == "pascal_voc" and options.dataset_name.lower() == "rsod" and RAW_DATA_DIR.exists():
        return RAW_DATA_DIR
    if options.source_format == "yolo" and options.dataset_name.lower() == "rsod" and YOLO_DATA_DIR.exists():
        return YOLO_DATA_DIR
    return RAW_DATASETS_DIR / options.dataset_name


def _resolve_data_root(data_root: Path | None) -> Path:
    return Path(data_root) if data_root is not None else DATA_DIR


def _resolve_yaml_path(dataset_name: str, yaml_path: Path | None) -> Path:
    if yaml_path is not None:
        return Path(yaml_path)
    return DATASET_CONFIGS_DIR / f"{dataset_name}.yaml"


def _scan_voc_coverage(source_root: Path) -> CoverageReport:
    images_dir = source_root / IMAGES_DIRNAME
    annotations_dir = source_root / ANNOTATIONS_DIRNAME
    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(f"Pascal VOC dataset requires {images_dir} and {annotations_dir}")

    image_stems = {path.stem for path in images_dir.iterdir() if path.is_file()}
    annotation_stems = {path.stem for path in annotations_dir.glob("*.xml")}
    matched = image_stems & annotation_stems
    denominator = max(len(image_stems), len(annotation_stems), 1)
    return CoverageReport(
        image_count=len(image_stems),
        annotation_count=len(annotation_stems),
        matched_count=len(matched),
        coverage=len(matched) / denominator,
    )


def _scan_yolo_coverage(source_root: Path) -> CoverageReport:
    images_dir = source_root / IMAGES_DIRNAME
    labels_dir = source_root / LABELS_DIRNAME
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"YOLO dataset requires {images_dir} and {labels_dir}")

    image_stems = {path.stem for path in images_dir.iterdir() if path.is_file()}
    label_stems = {path.stem for path in labels_dir.glob("*.txt")}
    matched = image_stems & label_stems
    denominator = max(len(image_stems), len(label_stems), 1)
    return CoverageReport(
        image_count=len(image_stems),
        annotation_count=len(label_stems),
        matched_count=len(matched),
        coverage=len(matched) / denominator,
    )


def _detect_coco_annotation_file(source_root: Path, dataset_name: str) -> Path:
    candidates = [
        source_root / "annotations.json",
        source_root / f"{dataset_name}.json",
        source_root / ANNOTATIONS_DIRNAME / "instances.json",
        source_root / ANNOTATIONS_DIRNAME / f"{dataset_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    annotations_dir = source_root / ANNOTATIONS_DIRNAME
    if annotations_dir.exists():
        matches = sorted(annotations_dir.glob("*.json"))
        if matches:
            return matches[0]
    matches = sorted(source_root.glob("*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No COCO annotation file found under {source_root}")


def _scan_coco_coverage(source_root: Path, dataset_name: str) -> CoverageReport:
    images_dir = source_root / IMAGES_DIRNAME if (source_root / IMAGES_DIRNAME).exists() else source_root
    annotation_file = _detect_coco_annotation_file(source_root, dataset_name)
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    image_ids = {int(item["id"]) for item in images}
    annotated_ids = {int(item["image_id"]) for item in annotations}
    image_files = [path for path in images_dir.iterdir() if path.is_file()]
    denominator = max(len(image_ids), len(image_files), 1)
    matched = len(image_ids & annotated_ids)
    return CoverageReport(
        image_count=max(len(image_ids), len(image_files)),
        annotation_count=len(annotated_ids),
        matched_count=matched,
        coverage=matched / denominator,
    )


def _check_raw(options: ConvertOptions, source_root: Path, min_coverage: float) -> CoverageReport:
    if options.source_format == "pascal_voc":
        report = _scan_voc_coverage(source_root)
    elif options.source_format == "coco":
        report = _scan_coco_coverage(source_root, options.dataset_name)
    elif options.source_format == "yolo":
        report = _scan_yolo_coverage(source_root)
    else:
        raise ValueError(f"Unsupported source format: {options.source_format!r}")

    if report.coverage < min_coverage:
        raise ValueError(
            f"Raw dataset coverage is below threshold: {report.coverage:.2%} < {min_coverage:.2%}"
        )
    return report


def _split_target_dirs(data_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_dirs = {split: data_root / split / IMAGES_DIRNAME for split in _SPLITS}
    label_dirs = {split: data_root / split / LABELS_DIRNAME for split in _SPLITS}
    return image_dirs, label_dirs


def prepare_dataset(
    options: ConvertOptions,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    data_root: Path | None = None,
    yaml_path: Path | None = None,
) -> OrchestrationResult:
    """Run coverage checks, conversion, split materialization, and yaml generation."""

    source_root = _resolve_source_root(options)
    coverage = _check_raw(options, source_root, min_coverage)
    target_data_root = _resolve_data_root(data_root)
    final_yaml_target = _resolve_yaml_path(options.dataset_name, yaml_path)
    _user_classes = list(options.classes) if options.classes is not None else None
    convert_options = ConvertOptions(
        dataset_name=options.dataset_name,
        source_format=options.source_format,
        task=options.task,
        classes=_user_classes,
        train_rate=options.train_rate,
        val_rate=options.val_rate,
        test_rate=options.test_rate,
        random_state=options.random_state,
        source_root=source_root,
    )

    converter = get_converter(convert_options.source_format)
    with TemporaryDirectory() as tmp_dir:
        temp_labels_dir = Path(tmp_dir) / LABELS_DIRNAME
        manifest = converter.convert(
            options=convert_options,
            source_root=source_root,
            output_labels_dir=temp_labels_dir,
        )
        _final_classes = list(manifest.classes)

        split_map = split_pairs(
            manifest.samples,
            train_rate=convert_options.train_rate,
            val_rate=convert_options.val_rate,
            test_rate=convert_options.test_rate,
            random_state=convert_options.random_state,
        )

        image_dirs, label_dirs = _split_target_dirs(target_data_root)
        materialized_split_map = materialize_splits(
            split_map,
            image_dir_by_split=image_dirs,
            label_dir_by_split=label_dirs,
        )

    final_manifest = ConversionManifest(
        dataset_name=manifest.dataset_name,
        source_format=manifest.source_format,
        task=manifest.task,
        classes=_final_classes,
        samples=[sample for split in _SPLITS for sample in materialized_split_map.get(split, [])],
        source_root=manifest.source_root,
        labels_root=label_dirs[SPLIT_TRAIN].parent,
    )

    final_yaml_path = write_dataset_yaml(
        yaml_path=final_yaml_target,
        data_root=target_data_root,
        manifest=final_manifest,
        split_map=materialized_split_map,
        train_rate=convert_options.train_rate,
        val_rate=convert_options.val_rate,
        test_rate=convert_options.test_rate,
        random_state=convert_options.random_state,
    )

    return OrchestrationResult(
        manifest=final_manifest,
        split_map=materialized_split_map,
        yaml_path=final_yaml_path,
        coverage=coverage,
    )


__all__ = [
    "CoverageReport",
    "OrchestrationResult",
    "prepare_dataset",
]
