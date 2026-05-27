from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..schemas import (
    ConfigJobCreate,
    EvalJobCreate,
    InferJobCreate,
    JobResponse,
    TrainJobCreate,
    TransformJobCreate,
    ValidateJobCreate,
)
from ..services.jobs import get_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

ROOT = Path(__file__).resolve().parents[5]
PLATFORM_DIR = ROOT / "apps" / "platform"
PYTHON = os.environ.get("PYTHON", "python")


def _job_to_response(job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        task=job.task,
        command=job.command,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        return_code=job.return_code,
        progress_percent=job.progress_percent,
        pid=job.pid,
        log_path=job.log_path,
        stdout_tail=job.stdout[-4000:],
        stderr_tail=job.stderr[-4000:],
        error=job.error,
        result=job.result,
    )


def _submit_job(task: str, command: list[str]) -> JobResponse:
    manager = get_manager()
    job_id = manager.submit(task, command)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(500, "Failed to create job")
    return _job_to_response(job)


def _build_train_command(body: TrainJobCreate) -> list[str]:
    cmd = [str(PYTHON), "-m", "odp_platform.cli.train_model"]
    if body.config:
        cmd += ["--config", body.config]
    if body.model:
        cmd += ["--model", body.model]
    if body.data:
        cmd += ["--data", body.data]
    if body.epochs:
        cmd += ["--epochs", str(body.epochs)]
    if body.batch:
        cmd += ["--batch", str(body.batch)]
    if body.device:
        cmd += ["--device", body.device]
    if body.project:
        cmd += ["--project", body.project]
    if body.name:
        cmd += ["--name", body.name]
    if body.task_type:
        cmd += ["--task", body.task_type]
    if body.resume is not None:
        cmd.append("--resume" if body.resume else "--no-resume")
    return cmd


def _build_transform_command(body: TransformJobCreate) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "odp_platform.cli.transform_data",
        "--dataset",
        body.dataset,
        "--format",
        body.format,
        "--task",
        body.task_type,
    ]


def _build_validate_command(body: ValidateJobCreate) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "odp_platform.cli.validate_data",
        "--dataset",
        body.dataset,
        "--task",
        body.task_type,
    ]


def _build_init_command() -> list[str]:
    return [str(PYTHON), "-m", "odp_platform.cli.init_project"]


def _build_config_command(body: ConfigJobCreate) -> list[str]:
    cmd = [str(PYTHON), "-m", "odp_platform.runtime_config.generator", body.task]
    if body.force:
        cmd.append("--overwrite")
    return cmd


def _build_evaluate_command(body: EvalJobCreate) -> list[str]:
    cmd = [
        str(PYTHON),
        "-m",
        "odp_platform.cli.val_model",
        "--model",
        body.model,
        "--data",
        body.data,
        "--task",
        body.task_type,
    ]
    if body.device:
        cmd += ["--device", body.device]
    if body.batch is not None:
        cmd += ["--batch", str(body.batch)]
    if body.imgsz is not None:
        cmd += ["--imgsz", str(body.imgsz)]
    if body.split:
        cmd += ["--split", body.split]
    return cmd


def _build_infer_command(body: InferJobCreate) -> list[str]:
    cmd = [
        str(PYTHON),
        "-m",
        "odp_platform.cli.infer",
        "--model",
        body.model,
        "--data",
        body.data,
    ]
    if body.source:
        cmd += ["--source", body.source]
    if body.task_type:
        cmd += ["--task", body.task_type]
    return cmd


@router.post("/init", response_model=JobResponse)
def submit_init() -> JobResponse:
    return _submit_job("init", _build_init_command())


@router.post("/config", response_model=JobResponse)
def submit_config(body: ConfigJobCreate) -> JobResponse:
    return _submit_job("config", _build_config_command(body))


@router.post("/evaluate", response_model=JobResponse)
def submit_evaluate(body: EvalJobCreate) -> JobResponse:
    return _submit_job("evaluate", _build_evaluate_command(body))


@router.post("/train", response_model=JobResponse)
def submit_train(body: TrainJobCreate) -> JobResponse:
    return _submit_job("train", _build_train_command(body))


@router.post("/transform", response_model=JobResponse)
def submit_transform(body: TransformJobCreate) -> JobResponse:
    return _submit_job("transform", _build_transform_command(body))


@router.post("/validate", response_model=JobResponse)
def submit_validate(body: ValidateJobCreate) -> JobResponse:
    return _submit_job("validate", _build_validate_command(body))


@router.post("/infer", response_model=JobResponse)
def submit_infer(body: InferJobCreate) -> JobResponse:
    return _submit_job("infer", _build_infer_command(body))


@router.get("", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    manager = get_manager()
    return [_job_to_response(j) for j in manager.list_all()]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    manager = get_manager()
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_to_response(job)


@router.delete("/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str) -> JobResponse:
    manager = get_manager()
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    success = manager.cancel(job_id)
    if not success:
        raise HTTPException(409, f"Job {job_id} cannot be cancelled in current state")
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(500, "Job disappeared after cancellation")
    return _job_to_response(job)
