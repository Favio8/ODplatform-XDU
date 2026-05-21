from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import pytest
import yaml
from PIL import Image

from odp_platform.common.constants import FORMAT_COCO, FORMAT_PASCAL_VOC, FORMAT_YOLO, TASK_DETECT, TASK_SEGMENT
from odp_platform.data_pipeline.core import coco, pascal_voc, yolo
from odp_platform.data_pipeline.registry import ConvertOptions


def _write_image(path: Path, *, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_voc_xml(path: Path, *, class_name: str, bbox: tuple[str, str, str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Element("annotation")
    obj = SubElement(root, "object")
    SubElement(obj, "name").text = class_name
    box = SubElement(obj, "bndbox")
    SubElement(box, "xmin").text = bbox[0]
    SubElement(box, "ymin").text = bbox[1]
    SubElement(box, "xmax").text = bbox[2]
    SubElement(box, "ymax").text = bbox[3]
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_pascal_voc_converter_skips_bad_xml_and_clips_bbox(tmp_path: Path) -> None:
    source_root = tmp_path / "demo_voc"
    images_dir = source_root / "images"
    annotations_dir = source_root / "annotations"

    _write_image(images_dir / "sample_001.JPG")
    _write_voc_xml(
        annotations_dir / "sample_001.xml",
        class_name="aircraft",
        bbox=("-10", "10", "40", "30"),
    )
    _write_image(images_dir / "sample_002.JPG")
    (annotations_dir / "sample_002.xml").write_text("<annotation><object>", encoding="utf-8")

    options = ConvertOptions(
        dataset_name="demo_voc",
        source_format=FORMAT_PASCAL_VOC,
        task=TASK_DETECT,
        classes=["aircraft"],
    )
    manifest = pascal_voc.convert(
        options=options,
        source_root=source_root,
        output_labels_dir=tmp_path / "labels",
    )

    assert manifest.classes == ["aircraft"]
    assert len(manifest.samples) == 1
    label_lines = manifest.samples[0].label_path.read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 1
    assert label_lines[0] == "0 0.200000 0.400000 0.400000 0.400000"


def test_coco_segment_converter_merges_multi_polygon_annotations(tmp_path: Path) -> None:
    source_root = tmp_path / "demo_coco"
    images_dir = source_root / "images"
    _write_image(images_dir / "sample_001.jpg", size=(200, 100))

    payload = {
        "images": [
            {"id": 1, "file_name": "sample_001.jpg", "width": 200, "height": 100},
        ],
        "categories": [
            {"id": 7, "name": "ship"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 10, 100, 60],
                "segmentation": [
                    [10, 10, 60, 10, 60, 40],
                    [80, 20, 110, 20, 110, 60],
                ],
            }
        ],
    }
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "annotations.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    options = ConvertOptions(
        dataset_name="demo_coco",
        source_format=FORMAT_COCO,
        task=TASK_SEGMENT,
    )
    manifest = coco.convert(
        options=options,
        source_root=source_root,
        output_labels_dir=tmp_path / "coco_labels",
    )

    assert manifest.classes == ["ship"]
    assert len(manifest.samples) == 1
    tokens = manifest.samples[0].label_path.read_text(encoding="utf-8").split()
    assert tokens[0] == "0"
    assert len(tokens) > 7
    assert all(0.0 <= float(value) <= 1.0 for value in tokens[1:])


def test_yolo_converter_skips_invalid_txt_and_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "demo_yolo"
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    output_dir = tmp_path / "prepared_labels"

    _write_image(images_dir / "sample_001.JPG")
    _write_image(images_dir / "sample_002.JPG")
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "sample_001.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (labels_dir / "sample_002.txt").write_text("0 1.5 0.5 0.4 0.4\n", encoding="utf-8")
    (source_root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["aircraft"]}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(yolo, "_supports_hardlink", lambda src_dir, dst_dir: True)
    monkeypatch.setattr(yolo.os, "link", lambda src, dst: (_ for _ in ()).throw(OSError("no hardlink")))

    options = ConvertOptions(
        dataset_name="demo_yolo",
        source_format=FORMAT_YOLO,
        task=TASK_DETECT,
    )
    manifest = yolo.convert(
        options=options,
        source_root=source_root,
        output_labels_dir=output_dir,
    )

    assert len(manifest.samples) == 1
    assert (output_dir / "sample_001.txt").exists()
    assert not (output_dir / "sample_002.txt").exists()
    assert (output_dir / "sample_001.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.4 0.4\n"
