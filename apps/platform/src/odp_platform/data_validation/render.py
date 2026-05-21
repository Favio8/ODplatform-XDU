"""Render structured validation reports to human-readable logs."""

from __future__ import annotations

import json
import logging

from odp_platform.data_validation.registry import CheckResult, CheckSeverity
from odp_platform.data_validation.report import ValidationReport


def _log_result_line(logger: logging.Logger, result: CheckResult) -> None:
    line = f"[{result.severity}] {result.name}: {result.summary}"
    if result.severity == CheckSeverity.ERROR:
        logger.error(line)
    elif result.severity == CheckSeverity.WARNING:
        logger.warning(line)
    elif result.severity == CheckSeverity.INFO:
        logger.info(line)
    else:
        logger.debug(line)


def _render_summary(logger: logging.Logger, report: ValidationReport) -> None:
    counts = report.counts_by_severity
    logger.info("Validation Summary".center(72, "="))
    logger.info("YAML: %s", report.yaml_path)
    logger.info("Task: %s | Splits: %s | Images: %s", report.snapshot.task_type, report.snapshot.splits, report.snapshot.total_images)
    logger.info("Overall Severity: %s | Exit Code: %s", report.overall_severity, report.exit_code)
    logger.info(
        "Counts: PASS=%s INFO=%s WARNING=%s ERROR=%s | Duration: %.3fs",
        counts[CheckSeverity.PASS],
        counts[CheckSeverity.INFO],
        counts[CheckSeverity.WARNING],
        counts[CheckSeverity.ERROR],
        report.duration_seconds,
    )
    if report.report_path is not None and report.report_path.exists():
        logger.info("JSON Report: %s", report.report_path)
    else:
        logger.info("JSON Report: not written")


def _render_failures(logger: logging.Logger, report: ValidationReport) -> None:
    logger.info("Failure Details".center(72, "="))
    if not report.failed_results:
        logger.info("No blocking validation issues found.")
        return

    for result in report.failed_results:
        logger.info("%s (%s)", result.name, result.severity)
        logger.info("%s", json.dumps(result.details, ensure_ascii=False, indent=2))


def render_to_logger(report: ValidationReport, logger: logging.Logger) -> None:
    """Render a validation report as a three-part logging summary."""

    _render_summary(logger, report)
    logger.info("Checks".center(72, "="))
    for result in report.results:
        _log_result_line(logger, result)
    _render_failures(logger, report)


__all__ = [
    "render_to_logger",
]
