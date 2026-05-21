"""Orchestration for data-validation checks and report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.performance_utils import time_it
from odp_platform.common.system_utils import log_device_info
from odp_platform.common.paths import validation_run_dir
from odp_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    autodiscover_checks,
    get_registered_checks,
)
from odp_platform.data_validation.report import ValidationReport
from odp_platform.data_validation.snapshot import build_snapshot


logger = get_logger(__name__)


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _safe_run_one(name: str, ctx: CheckContext) -> CheckResult:
    try:
        entry = get_registered_checks()[name]
        return entry.func(ctx)
    except Exception as exc:
        return CheckResult(
            name=name,
            severity=CheckSeverity.ERROR,
            summary=f"Check crashed: {exc}",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )


@time_it(name="run_all_checks", logger_instance=logger)
def run_all_checks(ctx: CheckContext) -> list[CheckResult]:
    """Run every registered validation check and never stop at the first failure."""

    autodiscover_checks()
    results: list[CheckResult] = []
    for name in get_registered_checks():
        results.append(_safe_run_one(name, ctx))
    return results


def validate_dataset(
    yaml_path: Path,
    task_type: str | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
    write_report: bool = True,
) -> ValidationReport:
    """Build a snapshot, run all checks, and optionally persist a JSON report."""

    yaml_path = Path(yaml_path)
    resolved_run_id = run_id or _default_run_id()
    resolved_run_dir = Path(run_dir) if run_dir is not None else validation_run_dir(resolved_run_id)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_type="data_validation", model_name=yaml_path.stem)
    log_device_info(logger)

    started_at = datetime.now()
    started_at_iso = started_at.isoformat(timespec="seconds")
    start = perf_counter()

    snapshot = build_snapshot(yaml_path, task_type=task_type)
    ctx = CheckContext(yaml_path=yaml_path, snapshot=snapshot)
    results = run_all_checks(ctx)

    report = ValidationReport(
        run_id=resolved_run_id,
        yaml_path=yaml_path,
        snapshot=snapshot,
        results=results,
        duration_seconds=perf_counter() - start,
        started_at_iso=started_at_iso,
        run_dir=resolved_run_dir,
    )

    if write_report and report.report_path is not None:
        report.report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return report


__all__ = [
    "run_all_checks",
    "validate_dataset",
]
