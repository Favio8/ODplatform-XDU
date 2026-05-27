from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthPayload(BaseModel):
    status: str
    timestamp: str


class DatasetItem(BaseModel):
    name: str
    task: str
    yaml_path: str
    data_root: str
    splits: dict[str, int]
    class_names: list[str]
    coverage: float
    status: str


class PipelineStage(BaseModel):
    key: str
    title: str
    status: Literal["done", "running", "ready", "warning"]
    note: str


class RunSummary(BaseModel):
    run_id: str
    dataset: str
    task: str
    model: str
    epochs: int
    status: Literal["completed", "running", "failed"]
    project_dir: str
    best_checkpoint: str | None = None
    last_checkpoint: str | None = None
    metric: float | None = None
    started_at: str


class CheckpointItem(BaseModel):
    path: str
    name: str
    dataset: str
    task: str
    created_at: str
    kind: Literal["best", "last"]


class SegmentationRegion(BaseModel):
    name: str
    color: str
    area_ratio: float
    note: str


class InferenceResult(BaseModel):
    dataset: str
    image_name: str
    image_path: str
    mask_path: str
    confidence: float
    regions: list[SegmentationRegion]
    summary: str


class AgentAdviceItem(BaseModel):
    title: str
    description: str
    priority: Literal["high", "medium", "low"]


class AgentReport(BaseModel):
    report_id: str
    dataset: str
    scene_type: str
    spaces: list[str]
    advice: list[AgentAdviceItem]
    circulation: str
    summary: str
    export_path: str


JobStatusLiteral = Literal["pending", "running", "completed", "failed", "cancelled"]


class JobResponse(BaseModel):
    job_id: str
    task: str
    command: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    progress_percent: int = 0
    pid: int | None = None
    log_path: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""
    result: dict[str, Any] | None = None


class TrainJobCreate(BaseModel):
    config: str | None = None
    model: str | None = None
    data: str | None = None
    epochs: int | None = None
    batch: int | None = None
    device: str | None = None
    project: str | None = None
    name: str | None = None
    task_type: str = "segment"
    resume: bool | None = None


class TransformJobCreate(BaseModel):
    dataset: str
    format: Literal["pascal_voc", "coco", "yolo"] = "pascal_voc"
    task_type: str = "segment"


class ValidateJobCreate(BaseModel):
    dataset: str
    task_type: str = "segment"


class ConfigJobCreate(BaseModel):
    task: Literal["train", "val", "infer"] = "train"
    force: bool = False


class EvalJobCreate(BaseModel):
    model: str
    data: str
    device: str | None = None
    batch: int | None = None
    imgsz: int | None = None
    split: str | None = None
    task_type: str = "segment"


class InferJobCreate(BaseModel):
    model: str
    data: str
    source: str | None = None
    task_type: str = "segment"


class ProjectStatusResponse(BaseModel):
    root: str
    raw_datasets: list[dict[str, Any]]
    processed_datasets: list[dict[str, Any]]
    dataset_configs: list[dict[str, Any]]
    runtime_configs: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]


class OverviewPayload(BaseModel):
    generated_at: str
    project_name: str
    pipeline: list[PipelineStage]
    datasets: list[DatasetItem]
    runs: list[RunSummary]
    checkpoints: list[CheckpointItem]
    inference: InferenceResult
    agent: AgentReport
    metrics: dict[str, Any]
