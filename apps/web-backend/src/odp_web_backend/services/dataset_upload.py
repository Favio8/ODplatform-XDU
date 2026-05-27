from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .workspace import RAW_DATA_DIR


DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
SPLIT_DIRS = ("train", "valid", "val", "test")


def upload_dataset_archive(*, dataset_name: str, archive_bytes: bytes, force: bool = False) -> dict[str, Any]:
    clean_name = dataset_name.strip()
    if not DATASET_NAME_PATTERN.match(clean_name):
        raise ValueError("数据集名称只能包含字母、数字、下划线和短横线，长度不超过 80。")

    if not archive_bytes:
        raise ValueError("上传文件为空。")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if not members:
                raise ValueError("ZIP 中没有可用文件。")
            _validate_zip_members(archive.infolist())

            target_dir = RAW_DATA_DIR / clean_name
            created_target = not target_dir.exists()
            if target_dir.exists() and any(target_dir.iterdir()) and not force:
                raise FileExistsError(f"目标目录已存在且非空: {target_dir}")

            if target_dir.exists() and force:
                shutil.rmtree(target_dir)
                created_target = True
            target_dir.mkdir(parents=True, exist_ok=True)

            file_count = 0
            total_bytes = 0
            try:
                for member in archive.infolist():
                    _extract_member(archive, member, target_dir)
                    if not member.is_dir():
                        file_count += 1
                        total_bytes += int(member.file_size)
            except Exception:
                if created_target and target_dir.exists():
                    shutil.rmtree(target_dir)
                raise
    except zipfile.BadZipFile as exc:
        raise ValueError("上传文件不是有效 ZIP。") from exc

    _flatten_single_wrapper_dir(target_dir)
    detected = detect_dataset_format(target_dir)
    dir_count = sum(1 for item in target_dir.rglob("*") if item.is_dir())

    return {
        "dataset_name": clean_name,
        "raw_path": str(target_dir),
        "file_count": file_count,
        "dir_count": dir_count,
        "total_bytes": total_bytes,
        "detected_format": detected,
        "next_step": "请确认格式后点击转换数据；上传阶段不会自动执行 D3/D4。",
    }


def detect_dataset_format(root: Path) -> dict[str, Any]:
    checks = [_detect_coco(root), _detect_yolo(root), _detect_voc(root)]
    detected = [item for item in checks if item["format"] != "unknown"]
    if not detected:
        reasons = [reason for item in checks for reason in item["reasons"]]
        return {
            "format": "unknown",
            "confidence": "low",
            "reasons": reasons or ["未发现 COCO / YOLO / Pascal VOC 的典型文件结构。"],
            "candidates": [],
        }
    priority = {"coco": 3, "yolo": 2, "pascal_voc": 1}
    best = sorted(detected, key=lambda item: priority.get(item["format"], 0), reverse=True)[0]
    best["candidates"] = [{"format": item["format"], "reasons": item["reasons"]} for item in detected]
    return best


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    for member in members:
        raw_name = member.filename.replace("\\", "/")
        parts = Path(raw_name).parts
        if raw_name.startswith("/") or ".." in parts or re.match(r"^[A-Za-z]:", raw_name):
            raise ValueError(f"ZIP 包含不安全路径: {member.filename}")


def _extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, target_dir: Path) -> None:
    raw_name = member.filename.replace("\\", "/")
    destination = (target_dir / raw_name).resolve()
    root = target_dir.resolve()
    if root != destination and root not in destination.parents:
        raise ValueError(f"ZIP 包含越界路径: {member.filename}")
    if member.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _flatten_single_wrapper_dir(target_dir: Path) -> None:
    children = [item for item in target_dir.iterdir() if not item.name.startswith("__MACOSX")]
    if len(children) != 1 or not children[0].is_dir():
        return
    wrapper = children[0]
    for child in wrapper.iterdir():
        shutil.move(str(child), str(target_dir / child.name))
    shutil.rmtree(wrapper)


def _detect_coco(root: Path) -> dict[str, Any]:
    json_files = sorted(root.rglob("*.json"))
    reasons: list[str] = []
    for path in json_files[:20]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and all(key in payload for key in ("images", "annotations", "categories")):
            reasons.append(f"发现 COCO 标注文件: {path.relative_to(root)}")
            return {"format": "coco", "confidence": "high", "reasons": reasons}
    return {"format": "unknown", "confidence": "low", "reasons": ["未发现包含 images/annotations/categories 的 COCO JSON。"]}


def _detect_yolo(root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    candidate_roots = [root] + [root / split for split in SPLIT_DIRS]
    for candidate in candidate_roots:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            reasons.append(f"发现 YOLO 目录结构: {candidate.relative_to(root) if candidate != root else '.'}/images + labels")
            return {"format": "yolo", "confidence": "high", "reasons": reasons}
    if (root / "data.yaml").exists() or (root / "dataset.yaml").exists():
        reasons.append("发现 data.yaml/dataset.yaml，可能是 YOLO 数据集。")
        return {"format": "yolo", "confidence": "medium", "reasons": reasons}
    return {"format": "unknown", "confidence": "low", "reasons": ["未发现 YOLO images/labels 结构。"]}


def _detect_voc(root: Path) -> dict[str, Any]:
    pairs = [
        (root / "JPEGImages", root / "Annotations"),
        (root / "images", root / "annotations"),
    ]
    for images_dir, annotations_dir in pairs:
        if images_dir.exists() and annotations_dir.exists() and list(annotations_dir.glob("*.xml")):
            return {
                "format": "pascal_voc",
                "confidence": "high",
                "reasons": [f"发现 Pascal VOC 目录结构: {images_dir.name} + {annotations_dir.name}/*.xml"],
            }
    return {"format": "unknown", "confidence": "low", "reasons": ["未发现 Pascal VOC XML 标注结构。"]}
