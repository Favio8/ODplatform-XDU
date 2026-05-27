from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
PLATFORM_CLI_DIR = ROOT / "apps" / "platform"
JOB_RUNS_DIR = ROOT / "runs" / "web_jobs"
PYTHON = os.environ.get("PYTHON", "python")
TAIL_LIMIT = 12000


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    task: str
    command: list[str]
    status: JobStatus = JobStatus.PENDING
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    progress_percent: int = 0
    pid: int | None = None
    log_path: str | None = None
    _process: subprocess.Popen | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")
        if not self.log_path:
            self.log_path = str(JOB_RUNS_DIR / self.job_id)

    @property
    def job_dir(self) -> Path:
        return Path(self.log_path or (JOB_RUNS_DIR / self.job_id))

    @property
    def json_path(self) -> Path:
        return self.job_dir / "job.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task": self.task,
            "command": self.command,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "error": self.error,
            "progress_percent": self.progress_percent,
            "pid": self.pid,
            "log_path": str(self.job_dir),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        status = JobStatus(data.get("status", JobStatus.PENDING.value))
        if status == JobStatus.RUNNING:
            status = JobStatus.FAILED
            data["error"] = data.get("error") or "后端服务重启，运行中任务状态已丢失。请重新提交任务。"
            data["finished_at"] = data.get("finished_at") or datetime.now().isoformat(timespec="seconds")
        return cls(
            job_id=str(data["job_id"]),
            task=str(data["task"]),
            command=list(data.get("command", [])),
            status=status,
            created_at=str(data.get("created_at") or ""),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            return_code=data.get("return_code"),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            result=data.get("result"),
            error=str(data.get("error") or ""),
            progress_percent=int(data.get("progress_percent") or 0),
            pid=data.get("pid"),
            log_path=data.get("log_path"),
        )


@dataclass
class JobManager:
    _jobs: dict[str, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _executor: threading.ThreadPoolExecutor | None = None
    _loaded: bool = False

    def _ensure_executor(self) -> threading.ThreadPoolExecutor:
        if self._executor is None or self._executor._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job_worker")
        return self._executor

    def submit(self, task: str, command: list[str]) -> str:
        self._load_from_disk()
        job_id = f"{task}-{uuid.uuid4().hex[:8]}"
        job = Job(job_id=job_id, task=task, command=command)
        job.job_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = job
            self._save_job(job)
        self._ensure_executor().submit(self._run, job_id)
        return job_id

    def _run(self, job_id: str) -> None:
        job: Job | None = None
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING
        job.progress_percent = 5
        job.started_at = datetime.now().isoformat(timespec="seconds")
        self._save_job(job)
        try:
            proc = subprocess.Popen(
                job.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(ROOT),
            )
            job._process = proc
            job.pid = proc.pid
            self._save_job(job)

            threads = [
                threading.Thread(target=self._read_stream, args=(job, proc.stdout, "stdout"), daemon=True),
                threading.Thread(target=self._read_stream, args=(job, proc.stderr, "stderr"), daemon=True),
            ]
            for thread in threads:
                thread.start()
            while proc.poll() is None:
                self._update_progress_from_logs(job)
                self._save_job(job)
                time.sleep(1)
            for thread in threads:
                thread.join(timeout=2)
            job.return_code = proc.returncode
            job.finished_at = datetime.now().isoformat(timespec="seconds")
            if proc.returncode == 0:
                job.status = JobStatus.COMPLETED
                job.progress_percent = 100
                job.result = self._collect_result(job)
            elif job.status == JobStatus.CANCELLED:
                job.error = job.error or "任务已取消。"
                job.progress_percent = min(job.progress_percent, 99)
            else:
                job.status = JobStatus.FAILED
                job.error = job.stderr or job.stdout
                job.progress_percent = min(job.progress_percent, 99)
            job._process = None
            job.pid = None
            self._save_job(job)
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.finished_at = datetime.now().isoformat(timespec="seconds")
            job._process = None
            job.pid = None
            self._save_job(job)

    def _collect_result(self, job: Job) -> dict[str, Any]:
        if job.task == "train":
            return _collect_train_result()
        if job.task == "init":
            return _collect_init_result()
        if job.task == "config":
            return _collect_config_result()
        if job.task == "evaluate":
            return _collect_evaluate_result()
        if job.task == "validate":
            return _collect_validation_result()
        if job.task == "transform":
            return _collect_transform_result()
        if job.task == "infer":
            return _collect_infer_result()
        return {}

    def get(self, job_id: str) -> Job | None:
        self._load_from_disk()
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        self._load_from_disk()
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        self._load_from_disk()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status == JobStatus.RUNNING and job._process is not None:
                job.status = JobStatus.CANCELLED
                job.error = "用户取消任务。"
                try:
                    job._process.terminate()
                except Exception:
                    pass
                self._save_job(job)
                return True
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                self._save_job(job)
                return True
        return False

    def _load_from_disk(self) -> None:
        if self._loaded:
            return
        JOB_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        loaded: dict[str, Job] = {}
        for path in JOB_RUNS_DIR.glob("*/job.json"):
            try:
                job = Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
                loaded[job.job_id] = job
            except Exception:
                continue
        with self._lock:
            self._jobs.update(loaded)
            for job in loaded.values():
                self._save_job(job)
            self._loaded = True

    def _save_job(self, job: Job) -> None:
        job.job_dir.mkdir(parents=True, exist_ok=True)
        job.json_path.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_stream(self, job: Job, stream, stream_name: str) -> None:
        if stream is None:
            return
        log_file = job.job_dir / f"{stream_name}.log"
        with log_file.open("a", encoding="utf-8", errors="replace") as file:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                file.write(line)
                file.flush()
                with self._lock:
                    if stream_name == "stdout":
                        job.stdout = (job.stdout + line)[-TAIL_LIMIT:]
                    else:
                        job.stderr = (job.stderr + line)[-TAIL_LIMIT:]

    def _update_progress_from_logs(self, job: Job) -> None:
        if job.status != JobStatus.RUNNING:
            return
        text = f"{job.stdout}\n{job.stderr}"
        if job.task == "train":
            matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
            if matches:
                current, total = matches[-1]
                total_int = max(int(total), 1)
                current_int = min(int(current), total_int)
                job.progress_percent = max(job.progress_percent, min(95, int(current_int / total_int * 95)))
                return
        elapsed_hint = min(85, job.progress_percent + 2)
        job.progress_percent = max(job.progress_percent, elapsed_hint)


def _collect_train_result() -> dict[str, Any]:
    runs_dir = ROOT / "runs"
    results_files = list(runs_dir.glob("**/results.csv"))
    if not results_files:
        return {"message": "Training completed, no results.csv found yet."}
    latest = sorted(results_files, key=lambda p: p.stat().st_mtime)[-1]
    run_dir = latest.parent
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best) if best.exists() else None,
        "last_checkpoint": str(last) if last.exists() else None,
        "results_csv": str(latest),
    }


def _collect_validation_result() -> dict[str, Any]:
    reports_dir = ROOT / "runs" / "data_validation"
    reports = sorted(reports_dir.glob("*/report.json"))
    if not reports:
        return {"message": "Validation completed, no report found yet."}
    latest = reports[-1]
    return {"report_path": str(latest), "report": json.loads(latest.read_text(encoding="utf-8"))}


def _collect_transform_result() -> dict[str, Any]:
    return {"message": "Transform completed successfully."}


def _collect_init_result() -> dict[str, Any]:
    return {
        "root": str(ROOT),
        "raw_dir": str(ROOT / "data" / "raw"),
        "processed_dir": str(ROOT / "data" / "processed"),
        "message": "Project initialized successfully.",
    }


def _collect_config_result() -> dict[str, Any]:
    runtime_dir = PLATFORM_CLI_DIR / "configs" / "runtime"
    configs = sorted(runtime_dir.glob("*.yaml"))
    return {
        "runtime_dir": str(runtime_dir),
        "configs": [str(path) for path in configs],
    }


def _collect_evaluate_result() -> dict[str, Any]:
    runs_dir = ROOT / "runs"
    candidates = sorted(
        [path for path in runs_dir.glob("**/*") if path.is_dir() and ("val" in path.name.lower())],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return {"message": "Evaluation completed, output dir not detected."}
    latest = candidates[-1]
    return {"output_dir": str(latest)}


def _collect_infer_result() -> dict[str, Any]:
    infer_dirs = sorted((ROOT / "runs").glob("**/inference*"))
    if infer_dirs:
        return {"output_dir": str(infer_dirs[-1])}
    return {"message": "Inference completed, output dir not detected."}


_manager: JobManager | None = None


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
