from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from ..services.workspace import (
    ROOT,
    list_training_runs,
    parse_results_csv,
)

router = APIRouter(prefix="/api", tags=["uploads"])

RAW_DATA_DIR = ROOT / "data" / "raw"
UPLOAD_DIR = RAW_DATA_DIR / "user_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class UploadResponse(BaseModel):
    success: bool
    filename: str
    path: str
    message: str


class TrainingResultDetail(BaseModel):
    run_id: str
    dataset: str
    task: str
    model: str
    epochs: int
    status: str
    project_dir: str
    best_checkpoint: str | None
    last_checkpoint: str | None
    metric: float | None
    started_at: str
    finished_at: str | None
    curve: list[dict[str, float]]


@router.post("/upload/floorplan", response_model=UploadResponse)
async def upload_floorplan(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    ext = file.filename.split(".")[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "pdf"):
        raise HTTPException(400, f"Unsupported file type: .{ext}. Supported: jpg, png, pdf")

    import time
    timestamp = int(time.time())
    safe_name = file.filename.replace(f".{ext}", "").replace(" ", "_").replace("/", "_")
    saved_name = f"{timestamp}_{safe_name}.{ext}"
    save_path = UPLOAD_DIR / saved_name

    try:
        with file.file.stream("rb") as src:
            content = src.read()
        save_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(500, f"Failed to save file: {exc}")

    return UploadResponse(
        success=True,
        filename=file.filename,
        path=str(save_path),
        message=f"文件已保存，可用于后续分析",
    )


@router.get("/training/curve/{run_id}", response_model=TrainingResultDetail)
def get_training_curve(run_id: str) -> TrainingResultDetail:
    runs = list_training_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    if run is None:
        raise HTTPException(404, f"Training run {run_id} not found")

    curve_rows, metric = [], None
    run_dir = Path(run["project_dir"])
    if run_dir and (run_dir / "results.csv").exists():
        rows, metric = parse_results_csv(run_dir / "results.csv")
        curve_rows = rows
        if metric is not None:
            run["metric"] = metric

    project_dir = Path(run["project_dir"])
    best = project_dir / "weights" / "best.pt"
    last = project_dir / "weights" / "last.pt"

    return TrainingResultDetail(
        run_id=run["run_id"],
        dataset=run["dataset"],
        task=run["task"],
        model=run["model"],
        epochs=run["epochs"],
        status=run["status"],
        project_dir=run["project_dir"],
        best_checkpoint=str(best) if best.exists() else None,
        last_checkpoint=str(last) if last.exists() else None,
        metric=run["metric"],
        started_at=run["started_at"],
        finished_at=None,
        curve=curve_rows,
    )


@router.get("/training/curves/latest", response_model=list[TrainingResultDetail])
def get_latest_curves() -> list[TrainingResultDetail]:
    runs = list_training_runs()
    results: list[TrainingResultDetail] = []

    for run in runs:
        run_dir = Path(run["project_dir"])
        curve_rows = []
        metric = run.get("metric")
        if run_dir and (run_dir / "results.csv").exists():
            rows, _metric = parse_results_csv(run_dir / "results.csv")
            curve_rows = rows
            if _metric is not None:
                metric = _metric

        project_dir = Path(run["project_dir"])
        best = project_dir / "weights" / "best.pt"
        last = project_dir / "weights" / "last.pt"

        results.append(TrainingResultDetail(
            run_id=run["run_id"],
            dataset=run["dataset"],
            task=run["task"],
            model=run["model"],
            epochs=run["epochs"],
            status=run["status"],
            project_dir=run["project_dir"],
            best_checkpoint=str(best) if best.exists() else None,
            last_checkpoint=str(last) if last.exists() else None,
            metric=metric,
            started_at=run["started_at"],
            finished_at=None,
            curve=curve_rows,
        ))

    return results
