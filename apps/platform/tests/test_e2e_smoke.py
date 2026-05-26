from __future__ import annotations

import logging
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import yaml
import pytest
from PIL import Image

from odp_platform.cli.transform_data import build_parser, main
from odp_platform.common.constants import FORMAT_PASCAL_VOC, ODP_META_KEY
from odp_platform.data_pipeline.orchestrator import prepare_dataset
from odp_platform.data_pipeline.registry import ConvertOptions


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(path)


def _write_voc_xml(path: Path, class_name: str = "aircraft") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = class_name
    bbox = SubElement(obj, "bndbox")
    SubElement(bbox, "xmin").text = "8"
    SubElement(bbox, "ymin").text = "8"
    SubElement(bbox, "xmax").text = "40"
    SubElement(bbox, "ymax").text = "40"
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _build_voc_dataset(root: Path, *, images: int, annotations: int) -> Path:
    source_root = root / "demo_voc"
    images_dir = source_root / "images"
    annotations_dir = source_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for index in range(images):
        stem = f"sample_{index:03d}"
        _write_image(images_dir / f"{stem}.jpg")
        if index < annotations:
            _write_voc_xml(annotations_dir / f"{stem}.xml")

    return source_root


def test_transform_cli_happy_path(tmp_path: Path) -> None:
    source_root = _build_voc_dataset(tmp_path, images=6, annotations=6)
    yaml_path = tmp_path / "configs" / "demo.yaml"
    data_root = tmp_path / "prepared"

    exit_code = main(
        [
            "--dataset",
            "demo_voc",
            "--format",
            FORMAT_PASCAL_VOC,
            "--source-root",
            str(source_root),
            "--data-root",
            str(data_root),
            "--config-path",
            str(yaml_path),
        ]
    )

    assert exit_code == 0
    assert yaml_path.exists()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert ODP_META_KEY in payload
    assert payload[ODP_META_KEY]["split"]["counts"]["total"] == 6
    assert (data_root / "train" / "images").exists()
    assert (data_root / "train" / "labels").exists()
    assert (data_root / "val" / "images").exists()
    assert (data_root / "test" / "images").exists()


def test_prepare_dataset_fails_fast_when_coverage_is_too_low(tmp_path: Path) -> None:
    source_root = _build_voc_dataset(tmp_path, images=4, annotations=1)
    options = ConvertOptions(
        dataset_name="broken_voc",
        source_format=FORMAT_PASCAL_VOC,
        source_root=source_root,
    )

    try:
        prepare_dataset(options, data_root=tmp_path / "prepared", yaml_path=tmp_path / "broken.yaml")
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("prepare_dataset should fail fast when coverage is below threshold")


def test_transform_help_includes_capability_matrix() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "Capability Matrix" in help_text
    assert "pascal_voc" in help_text


def test_prepare_dataset_warns_below_soft_threshold(tmp_path: Path) -> None:
    source_root = _build_voc_dataset(tmp_path, images=10, annotations=7)
    options = ConvertOptions(
        dataset_name="warning_voc",
        source_format=FORMAT_PASCAL_VOC,
        source_root=source_root,
    )
    logger = logging.getLogger("odp_platform.data_pipeline.orchestrator")
    handler = _ListHandler()
    logger.addHandler(handler)

    try:
        result = prepare_dataset(
            options,
            data_root=tmp_path / "prepared",
            yaml_path=tmp_path / "warning.yaml",
        )
    finally:
        logger.removeHandler(handler)

    assert result.coverage.coverage == pytest.approx(0.7)
    assert any("soft threshold" in message for message in handler.messages)


def test_transform_cli_rejects_unsupported_task_for_format() -> None:
    exit_code = main(
        [
            "--dataset",
            "demo_yolo",
            "--format",
            "yolo",
            "--task",
            "segment",
        ]
    )

    assert exit_code == 1
