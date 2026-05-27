from __future__ import annotations

import base64
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .workspace import ROOT, RUNS_DIR


ASSET_NAMES = {
    "confusion": ("confusion_matrix.png", "confusion_matrix_normalized.png"),
    "curves": (
        "MaskPR_curve.png",
        "MaskP_curve.png",
        "MaskR_curve.png",
        "MaskF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "results.png",
    ),
    "samples": (
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_pred.jpg",
        "val_batch0_labels.jpg",
        "val_batch1_labels.jpg",
        "val_batch2_labels.jpg",
    ),
}


def list_evaluation_reports() -> list[dict[str, Any]]:
    reports = [_report_summary(path) for path in _candidate_report_dirs()]
    return sorted(reports, key=lambda item: item["updated_at"], reverse=True)


def get_evaluation_report(report_id: str) -> dict[str, Any]:
    report_dir = _decode_report_id(report_id)
    if not _is_safe_path(report_dir) or not report_dir.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_id}")
    return _report_detail(report_dir)


def safe_file_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if not _is_safe_path(path) or not path.is_file():
        raise FileNotFoundError(relative_path)
    return path


def _candidate_report_dirs() -> list[Path]:
    dirs: list[Path] = []
    for path in RUNS_DIR.rglob("*"):
        if not path.is_dir():
            continue
        if _has_eval_assets(path):
            dirs.append(path)
    return dirs


def _has_eval_assets(path: Path) -> bool:
    names = {item.name for item in path.iterdir() if item.is_file()}
    expected = set().union(*ASSET_NAMES.values())
    return bool(names & expected) or (path / "predictions.json").exists()


def _report_summary(report_dir: Path) -> dict[str, Any]:
    detail = _report_detail(report_dir, include_assets=False)
    return {
        "report_id": _encode_report_id(report_dir),
        "name": report_dir.name,
        "path": str(report_dir),
        "updated_at": datetime.fromtimestamp(report_dir.stat().st_mtime).isoformat(timespec="seconds"),
        "metrics": detail["metrics"],
        "asset_count": detail["asset_count"],
    }


def _report_detail(report_dir: Path, include_assets: bool = True) -> dict[str, Any]:
    metrics = _parse_metrics(report_dir)
    grouped_assets = _group_assets(report_dir) if include_assets else {}
    asset_count = sum(len(items) for items in grouped_assets.values()) if include_assets else len(_asset_paths(report_dir))
    return {
        "report_id": _encode_report_id(report_dir),
        "name": report_dir.name,
        "path": str(report_dir),
        "updated_at": datetime.fromtimestamp(report_dir.stat().st_mtime).isoformat(timespec="seconds"),
        "metrics": metrics,
        "assets": grouped_assets,
        "asset_count": asset_count,
    }


def _parse_metrics(report_dir: Path) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "fitness": None,
    }
    results_csv = report_dir / "results.csv"
    if not results_csv.exists():
        return metrics
    try:
        rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    except Exception:
        return metrics
    if not rows:
        return metrics
    row = rows[-1]
    aliases = {
        "map50": ("metrics/mAP50(M)", "metrics/mAP50(B)", "mAP50"),
        "map50_95": ("metrics/mAP50-95(M)", "metrics/mAP50-95(B)", "mAP50-95"),
        "precision": ("metrics/precision(M)", "metrics/precision(B)", "precision"),
        "recall": ("metrics/recall(M)", "metrics/recall(B)", "recall"),
        "fitness": ("fitness",),
    }
    for key, names in aliases.items():
        metrics[key] = _first_float(row, names)
    return metrics


def _first_float(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    normalized = {key.strip(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _group_assets(report_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "confusion": [_asset_payload(path) for name in ASSET_NAMES["confusion"] if (path := report_dir / name).exists()],
        "curves": [_asset_payload(path) for name in ASSET_NAMES["curves"] if (path := report_dir / name).exists()],
        "samples": [_asset_payload(path) for name in ASSET_NAMES["samples"] if (path := report_dir / name).exists()],
    }


def _asset_paths(report_dir: Path) -> list[Path]:
    names = set().union(*ASSET_NAMES.values())
    return [path for path in report_dir.iterdir() if path.is_file() and path.name in names]


def _asset_payload(path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return {
        "name": path.name,
        "path": str(path),
        "url": f"/api/files?path={relative}",
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def _encode_report_id(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_report_id(report_id: str) -> Path:
    padding = "=" * (-len(report_id) % 4)
    relative = base64.urlsafe_b64decode((report_id + padding).encode("ascii")).decode("utf-8")
    return (ROOT / relative).resolve()


def _is_safe_path(path: Path) -> bool:
    root = ROOT.resolve()
    resolved = path.resolve()
    return resolved == root or root in resolved.parents
