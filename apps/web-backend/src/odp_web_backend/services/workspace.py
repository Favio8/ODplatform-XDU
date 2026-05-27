from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[5]
PLATFORM_DIR = ROOT / "apps" / "platform"
DATASETS_DIR = PLATFORM_DIR / "configs" / "datasets"
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
RUNTIME_CONFIGS_DIR = PLATFORM_DIR / "configs" / "runtime"


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    task: str
    yaml_path: Path
    data_root: Path
    splits: dict[str, int]
    class_names: list[str]
    coverage: float
    status: str


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _split_counts_from_yaml(data: dict[str, Any]) -> dict[str, int]:
    counts = data.get("odp_meta", {}).get("split", {}).get("counts", {})
    return {
        "train": int(counts.get("train", 0)),
        "val": int(counts.get("val", 0)),
        "test": int(counts.get("test", 0)),
    }


def _class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", {})
    if isinstance(names, dict):
        return [str(v) for _, v in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, list):
        return [str(v) for v in names]
    return []


def list_dataset_profiles() -> list[DatasetProfile]:
    profiles: list[DatasetProfile] = []
    for yaml_path in sorted(DATASETS_DIR.glob("*.yaml")):
        data = _safe_read_yaml(yaml_path)
        dataset_name = yaml_path.stem
        data_root = Path(data.get("path", RAW_DATA_DIR / dataset_name))
        splits = _split_counts_from_yaml(data)
        class_names = _class_names(data)
        task = str(data.get("odp_meta", {}).get("task", "segment"))
        if task != "segment":
            continue
        total = sum(splits.values()) or 1
        coverage = round(min(0.999, max(0.1, total / max(total, 1))), 4)
        status = "ready" if data else "draft"
        profiles.append(
            DatasetProfile(
                name=dataset_name,
                task=task,
                yaml_path=yaml_path,
                data_root=data_root,
                splits=splits,
                class_names=class_names,
                coverage=coverage,
                status=status,
            )
        )
    return profiles


def create_dataset_profile(
    name: str,
    class_names: list[str],
    train: int = 0,
    val: int = 0,
    test: int = 0,
) -> DatasetProfile:
    yaml_path = DATASETS_DIR / f"{name}.yaml"
    if yaml_path.exists():
        raise ValueError(f"数据集 {name} 已存在")

    total = train + val + test or 1
    data = {
        "path": str(RAW_DATA_DIR / name),
        "train": f"train/images",
        "val": f"val/images",
        "test": f"test/images",
        "nc": len(class_names),
        "names": {str(i): v for i, v in enumerate(class_names)},
        "odp_meta": {
            "dataset": name,
            "source_format": "manual",
            "task": "segment",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "split": {
                "train_rate": round(train / total, 3) if total else 0.8,
                "val_rate": round(val / total, 3) if total else 0.1,
                "test_rate": round(test / total, 3) if total else 0.1,
                "random_state": 42,
                "counts": {"train": train, "val": val, "test": test, "total": total},
            },
            "schema_version": 1,
        },
    }

    yaml_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return DatasetProfile(
        name=name,
        task="segment",
        yaml_path=yaml_path,
        data_root=RAW_DATA_DIR / name,
        splits={"train": train, "val": val, "test": test},
        class_names=class_names,
        coverage=round(min(0.999, max(0.1, total / max(total, 1))), 4),
        status="draft",
    )


def update_dataset_profile(
    name: str,
    class_names: list[str] | None = None,
    train: int | None = None,
    val: int | None = None,
    test: int | None = None,
) -> DatasetProfile:
    yaml_path = DATASETS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"数据集 {name} 不存在")

    data = _safe_read_yaml(yaml_path)
    meta = data.get("odp_meta", {})
    split = meta.get("split", {})
    counts = split.get("counts", {})

    if class_names is not None:
        data["nc"] = len(class_names)
        data["names"] = {str(i): v for i, v in enumerate(class_names)}

    if train is not None or val is not None or test is not None:
        t = train if train is not None else counts.get("train", 0)
        v = val if val is not None else counts.get("val", 0)
        te = test if test is not None else counts.get("test", 0)
        total = t + v + te or 1
        split["train_rate"] = round(t / total, 3)
        split["val_rate"] = round(v / total, 3)
        split["test_rate"] = round(te / total, 3)
        split["counts"] = {"train": t, "val": v, "test": te, "total": total}

    data["odp_meta"] = meta
    yaml_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    updated_data = _safe_read_yaml(yaml_path)
    splits = _split_counts_from_yaml(updated_data)
    new_class_names = _class_names(updated_data)
    total = sum(splits.values()) or 1

    return DatasetProfile(
        name=name,
        task="segment",
        yaml_path=yaml_path,
        data_root=Path(data.get("path", RAW_DATA_DIR / name)),
        splits=splits,
        class_names=new_class_names,
        coverage=round(min(0.999, max(0.1, total / max(total, 1))), 4),
        status="ready",
    )


def delete_dataset_profile(name: str) -> bool:
    yaml_path = DATASETS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"数据集 {name} 不存在")
    yaml_path.unlink()
    return True


def latest_validation_report() -> dict[str, Any]:
    reports = sorted((RUNS_DIR / "data_validation").glob("*/report.json"))
    if not reports:
        return {}
    return _safe_read_json(reports[-1])


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_checkpoints() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pt in sorted(CHECKPOINTS_DIR.glob("*.pt")):
        name = pt.stem
        parts = name.split("-")
        kind = "best" if name.endswith("best") else "last"
        dataset = parts[0] if parts else "unknown"
        items.append(
            {
                "path": str(pt),
                "name": pt.name,
                "dataset": dataset,
                "task": "segment" if "seg" in name else "detect",
                "created_at": datetime.fromtimestamp(pt.stat().st_mtime).isoformat(timespec="seconds"),
                "kind": kind,
            }
        )
    return items


def _latest_run_dir() -> Path | None:
    results_files = sorted((RUNS_DIR).glob("**/results.csv"), key=lambda p: p.stat().st_mtime)
    if not results_files:
        return None
    return results_files[-1].parent


def parse_results_csv(path: Path) -> tuple[list[dict[str, float]], float | None]:
    if not path.exists():
        return [], None
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": float(row.get("epoch", 0) or 0),
                    "miou": float(row.get("metrics/mAP50(M)", row.get("metrics/mAP50(B)", 0)) or 0),
                    "loss": float(row.get("train/seg_loss", row.get("train/box_loss", 0)) or 0),
                }
            )
    metric = rows[-1]["miou"] if rows else None
    return rows, metric


def list_training_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    latest_dir = latest_run_dir()
    if latest_dir:
        rows, metric = parse_results_csv(latest_dir / "results.csv")
        runs.append(
            {
                "run_id": latest_dir.name,
                "dataset": "room_separation_3",
                "task": "segment",
                "model": "yolov8n-seg.pt",
                "epochs": int(rows[-1]["epoch"]) if rows else 0,
                "status": "completed" if (latest_dir / "weights" / "best.pt").exists() else "running",
                "project_dir": str(latest_dir.parent),
                "best_checkpoint": str(latest_dir / "weights" / "best.pt") if (latest_dir / "weights" / "best.pt").exists() else None,
                "last_checkpoint": str(latest_dir / "weights" / "last.pt") if (latest_dir / "weights" / "last.pt").exists() else None,
                "metric": metric,
                "started_at": datetime.fromtimestamp(latest_dir.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return runs


def latest_run_dir() -> Path | None:
    results_files = sorted((RUNS_DIR).glob("**/results.csv"), key=lambda p: p.stat().st_mtime)
    if not results_files:
        return None
    return results_files[-1].parent


def latest_training_curve() -> list[dict[str, float]]:
    latest_dir = latest_run_dir()
    if latest_dir is None:
        return []
    rows, _metric = parse_results_csv(latest_dir / "results.csv")
    return rows


def latest_agent_report() -> dict[str, Any]:
    report_files = sorted((RUNS_DIR / "agent").glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    if report_files:
        data = _safe_read_json(report_files[-1])
        if data:
            return data
        md_files = sorted((RUNS_DIR / "agent").glob("*.md"), key=lambda p: p.stat().st_mtime)
        if md_files:
            return {
                "report_id": report_files[-1].parent.name if report_files else "unknown",
                "dataset": "room_separation_3",
                "scene_type": "户型图空间理解",
                "spaces": [],
                "circulation": "",
                "summary": md_files[-1].read_text(encoding="utf-8")[:500],
                "export_path": str(md_files[-1]),
                "advice": [],
            }
    return {
        "report_id": "agent-room-001",
        "dataset": "room_separation_3",
        "scene_type": "户型图空间理解",
        "spaces": ["客厅", "主卧", "次卧", "厨房", "卫生间"],
        "circulation": "建议从玄关-客餐厅-厨房形成连续动线，减少交叉干扰。",
        "summary": "该户型适合做轻度开放式改造，并增强收纳和采光。",
        "export_path": str(ROOT / "runs" / "agent" / "agent-room-001.md"),
        "advice": [
            {"title": "优化动线", "description": "打通客餐厅联系，减少厨房与卧室动线交叉。", "priority": "high"},
            {"title": "提升收纳", "description": "在玄关、主卧和过道增加通顶柜体。", "priority": "medium"},
            {"title": "增强采光", "description": "尽量保持客厅开敞与窗边视野通透。", "priority": "medium"},
        ],
    }


def latest_inference_result() -> dict[str, Any]:
    infer_dirs = sorted((RUNS_DIR).glob("**/inference*/"), key=lambda p: p.stat().st_mtime)
    if infer_dirs:
        latest_dir = infer_dirs[-1]
        result_file = latest_dir / "result.json"
        if result_file.exists():
            data = _safe_read_json(result_file)
            if data:
                return data
        image_files = list(latest_dir.glob("*.jpg")) + list(latest_dir.glob("*.png"))
        if image_files:
            return {
                "dataset": "detected",
                "image_name": image_files[-1].name,
                "image_path": str(image_files[-1]),
                "mask_path": str(latest_dir / f"{image_files[-1].stem}_mask.png"),
                "confidence": 0.0,
                "regions": [],
                "summary": f"Inference output: {latest_dir.name}",
            }
    return {
        "dataset": "room_separation_3",
        "image_name": "demo_floorplan_001.jpg",
        "image_path": str(RAW_DATA_DIR / "room_separation_3" / "images" / "demo_floorplan_001.jpg"),
        "mask_path": str(RUNS_DIR / "segment" / "demo_floorplan_001_mask.png"),
        "confidence": 0.91,
        "regions": [
            {"name": "客厅", "color": "#2dd4bf", "area_ratio": 0.35, "note": "通透度高，适合开放式客餐厅。"},
            {"name": "主卧", "color": "#f59e0b", "area_ratio": 0.24, "note": "建议增加衣柜与床头收纳。"},
            {"name": "厨房", "color": "#fb7185", "area_ratio": 0.16, "note": "建议优化动线并增加操作台长度。"},
            {"name": "卫生间", "color": "#a78bfa", "area_ratio": 0.09, "note": "建议做干湿分离。"},
        ],
        "summary": "分割结果可用于后续空间理解与装修建议生成。",
    }


def pipeline_stages() -> list[dict[str, Any]]:
    return [
        {"key": "raw", "title": "原始数据", "status": "done", "note": "data/raw/<dataset>"},
        {"key": "convert", "title": "数据转换", "status": "done", "note": "输出到 data/processed/<dataset>"},
        {"key": "qc", "title": "数据质检", "status": "done", "note": "PASS/INFO/WARNING/ERROR"},
        {"key": "config", "title": "运行配置", "status": "done", "note": "生成 runtime config"},
        {"key": "train", "title": "分割训练", "status": "running", "note": "可回放 runs/segment_train"},
        {"key": "eval", "title": "分割评估", "status": "ready", "note": "接收 checkpoint 后执行"},
        {"key": "infer", "title": "分割推理", "status": "ready", "note": "支持单图/批量"},
        {"key": "agent", "title": "Agent分析", "status": "ready", "note": "输出装修建议"},
    ]


def project_status() -> dict[str, Any]:
    def dir_items(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            {
                "name": item.name,
                "path": str(item),
                "kind": "dir" if item.is_dir() else "file",
                "updated_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat(timespec="seconds"),
            }
            for item in sorted(path.iterdir(), key=lambda p: p.name)
            if not item.name.startswith(".")
        ]

    def file_items(path: Path, pattern: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            {
                "name": item.name,
                "path": str(item),
                "updated_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat(timespec="seconds"),
            }
            for item in sorted(path.glob(pattern), key=lambda p: p.name)
        ]

    return {
        "root": str(ROOT),
        "raw_datasets": dir_items(RAW_DATA_DIR),
        "processed_datasets": dir_items(PROCESSED_DATA_DIR),
        "dataset_configs": file_items(DATASETS_DIR, "*.yaml"),
        "runtime_configs": file_items(RUNTIME_CONFIGS_DIR, "*.yaml"),
        "checkpoints": list_checkpoints(),
    }


def overview_payload() -> dict[str, Any]:
    datasets = [
        {
            "name": profile.name,
            "task": profile.task,
            "yaml_path": str(profile.yaml_path),
            "data_root": str(profile.data_root),
            "splits": profile.splits,
            "class_names": profile.class_names,
            "coverage": profile.coverage,
            "status": profile.status,
        }
        for profile in list_dataset_profiles()
    ]
    runs = list_training_runs()
    checkpoints = list_checkpoints()
    validation = latest_validation_report()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_name": "ODPlatform - 户型图语义分割与Agent分析",
        "pipeline": pipeline_stages(),
        "datasets": datasets,
        "runs": runs,
        "checkpoints": checkpoints,
        "inference": latest_inference_result(),
        "agent": latest_agent_report(),
        "metrics": {
            "dataset_count": len(datasets),
            "checkpoint_count": len([item for item in checkpoints if item["task"] == "segment"]),
            "run_count": len(runs),
            "validated_dataset": "room_separation_3",
            "training_curve": latest_training_curve(),
            "validation_summary": {
                "overall_severity": validation.get("overall_severity", "PASS"),
                "counts_by_severity": validation.get(
                    "counts_by_severity",
                    {"PASS": 0, "INFO": 0, "WARNING": 0, "ERROR": 0},
                ),
                "exit_code": validation.get("exit_code", 0),
            },
            "config_trace": [
                {"source": "DEFAULT", "value": "系统默认分割训练参数"},
                {"source": "YAML", "value": "apps/platform/configs/runtime/train.yaml"},
                {"source": "CLI", "value": "--data room_separation_3 --task segment"},
            ],
        },
    }
