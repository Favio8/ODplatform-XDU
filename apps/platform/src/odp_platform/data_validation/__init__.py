"""Dataset validation subsystem public API."""

from odp_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    autodiscover_checks,
    list_checks,
)
from odp_platform.data_validation.render import render_to_logger
from odp_platform.data_validation.report import ValidationReport
from odp_platform.data_validation.service import run_all_checks, validate_dataset
from odp_platform.data_validation.snapshot import DatasetSnapshot, SplitStats, build_snapshot


__all__ = [
    "CheckContext",
    "CheckResult",
    "CheckSeverity",
    "DatasetSnapshot",
    "SplitStats",
    "ValidationReport",
    "autodiscover_checks",
    "build_snapshot",
    "list_checks",
    "render_to_logger",
    "run_all_checks",
    "validate_dataset",
]
