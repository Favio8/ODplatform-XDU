"""Structured validation reports decoupled from rendering concerns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from odp_platform.data_validation.registry import CheckResult, CheckSeverity
from odp_platform.data_validation.snapshot import DatasetSnapshot, SplitStats


def _serialize_split_stats(stats: SplitStats) -> dict[str, int]:
    return {
        "image_count": stats.image_count,
        "annotated_count": stats.annotated_count,
        "total_instances": stats.total_instances,
    }


def _serialize_snapshot(snapshot: DatasetSnapshot) -> dict[str, Any]:
    return {
        "yaml_path": str(snapshot.yaml_path),
        "yaml_data": snapshot.yaml_data,
        "yaml_load_error": snapshot.yaml_load_error,
        "data_root": str(snapshot.data_root),
        "nc": snapshot.nc,
        "class_names": list(snapshot.class_names),
        "task_type": snapshot.task_type,
        "images_per_split": {
            split: [str(path) for path in paths]
            for split, paths in snapshot.images_per_split.items()
        },
        "labels_per_split": {
            split: [str(path) for path in paths]
            for split, paths in snapshot.labels_per_split.items()
        },
        "stats_per_split": {
            split: _serialize_split_stats(stats)
            for split, stats in snapshot.stats_per_split.items()
        },
        "scan_warnings": list(snapshot.scan_warnings),
        "splits": list(snapshot.splits),
        "total_images": snapshot.total_images,
    }


@dataclass
class ValidationReport:
    run_id: str
    yaml_path: Path
    snapshot: DatasetSnapshot
    results: list[CheckResult]
    duration_seconds: float
    started_at_iso: str
    run_dir: Path | None = None

    @property
    def overall_severity(self) -> str:
        if not self.results:
            return CheckSeverity.PASS
        return max(self.results, key=lambda result: CheckSeverity.rank(result.severity)).severity

    @property
    def counts_by_severity(self) -> dict[str, int]:
        counts = {
            CheckSeverity.PASS: 0,
            CheckSeverity.INFO: 0,
            CheckSeverity.WARNING: 0,
            CheckSeverity.ERROR: 0,
        }
        for result in self.results:
            counts[result.severity] = counts.get(result.severity, 0) + 1
        return counts

    @property
    def exit_code(self) -> int:
        if self.overall_severity == CheckSeverity.ERROR:
            return 2
        if self.overall_severity == CheckSeverity.WARNING:
            return 1
        return 0

    @property
    def failed_results(self) -> list[CheckResult]:
        return [result for result in self.results if result.severity not in (CheckSeverity.PASS, CheckSeverity.INFO)]

    @property
    def report_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "report.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "yaml_path": str(self.yaml_path),
            "run_dir": str(self.run_dir) if self.run_dir is not None else None,
            "report_path": str(self.report_path) if self.report_path is not None else None,
            "started_at_iso": self.started_at_iso,
            "duration_seconds": self.duration_seconds,
            "overall_severity": self.overall_severity,
            "counts_by_severity": self.counts_by_severity,
            "exit_code": self.exit_code,
            "snapshot": _serialize_snapshot(self.snapshot),
            "results": [
                {
                    "name": result.name,
                    "severity": result.severity,
                    "passed": result.passed,
                    "summary": result.summary,
                    "details": result.details,
                }
                for result in self.results
            ],
        }


__all__ = [
    "ValidationReport",
]
