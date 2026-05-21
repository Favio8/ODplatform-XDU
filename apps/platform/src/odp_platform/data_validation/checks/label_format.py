"""YOLO label-line structural validation check."""

from __future__ import annotations

from typing import Final

from odp_platform.common.constants import TASK_DETECT, TASK_SEGMENT
from odp_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check


DETAILS_PREVIEW_LIMIT: Final[int] = 20
FIELD_COUNT_MISMATCH: Final[str] = "field_count_mismatch"
PARSE_ERROR: Final[str] = "parse_error"
CLASS_ID_OUT_OF_RANGE: Final[str] = "class_id_out_of_range"
COORD_OUT_OF_RANGE: Final[str] = "coord_out_of_range"
POLYGON_TOO_FEW_POINTS: Final[str] = "polygon_too_few_points"


def _record_error(
    *,
    errors_preview: list[dict[str, object]],
    error_kinds: dict[str, int],
    split: str,
    label_name: str,
    line_no: int,
    error_kind: str,
    preview_limit: int,
) -> None:
    error_kinds[error_kind] = error_kinds.get(error_kind, 0) + 1
    if len(errors_preview) < preview_limit:
        errors_preview.append(
            {
                "split": split,
                "label_file": label_name,
                "line_no": line_no,
                "error_kind": error_kind,
            }
        )


def _validate_detect_line(parts: list[str], nc: int) -> str | None:
    if len(parts) != 5:
        return FIELD_COUNT_MISMATCH
    try:
        class_id = int(float(parts[0]))
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return PARSE_ERROR
    if not 0 <= class_id < nc:
        return CLASS_ID_OUT_OF_RANGE
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        return COORD_OUT_OF_RANGE
    return None


def _validate_segment_line(parts: list[str], nc: int) -> str | None:
    if len(parts) < 3:
        return FIELD_COUNT_MISMATCH
    if (len(parts) - 1) % 2 != 0:
        return FIELD_COUNT_MISMATCH
    point_count = (len(parts) - 1) // 2
    if point_count < 3:
        return POLYGON_TOO_FEW_POINTS
    try:
        class_id = int(float(parts[0]))
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return PARSE_ERROR
    if not 0 <= class_id < nc:
        return CLASS_ID_OUT_OF_RANGE
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        return COORD_OUT_OF_RANGE
    return None


@check("label_format")
def validate_label_format(ctx: CheckContext) -> CheckResult:
    snapshot = ctx.snapshot
    if snapshot.nc is None or snapshot.nc <= 0:
        return CheckResult(
            name="label_format",
            severity=CheckSeverity.INFO,
            summary="Label format check skipped because nc is invalid.",
            details={"reason": "snapshot.nc is missing or invalid"},
        )

    validator = _validate_detect_line if snapshot.task_type == TASK_DETECT else _validate_segment_line
    if snapshot.task_type not in {TASK_DETECT, TASK_SEGMENT}:
        return CheckResult(
            name="label_format",
            severity=CheckSeverity.INFO,
            summary=f"Label format check skipped because task {snapshot.task_type!r} is unsupported.",
            details={"reason": "unsupported task type", "task_type": snapshot.task_type},
        )

    total_lines = 0
    total_errors = 0
    error_kinds: dict[str, int] = {}
    errors_preview: list[dict[str, object]] = []

    for split in snapshot.splits:
        for label_path in snapshot.labels_per_split.get(split, ()):
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                total_errors += 1
                _record_error(
                    errors_preview=errors_preview,
                    error_kinds=error_kinds,
                    split=split,
                    label_name=label_path.name,
                    line_no=0,
                    error_kind=PARSE_ERROR,
                    preview_limit=DETAILS_PREVIEW_LIMIT,
                )
                continue

            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                total_lines += 1
                error_kind = validator(stripped.split(), snapshot.nc)
                if error_kind is None:
                    continue
                total_errors += 1
                _record_error(
                    errors_preview=errors_preview,
                    error_kinds=error_kinds,
                    split=split,
                    label_name=label_path.name,
                    line_no=line_no,
                    error_kind=error_kind,
                    preview_limit=DETAILS_PREVIEW_LIMIT,
                )

    if total_errors == 0:
        return CheckResult(
            name="label_format",
            severity=CheckSeverity.PASS,
            summary=f"Validated {total_lines} label lines with no structural errors.",
            details={
                "task_type": snapshot.task_type,
                "total_lines": total_lines,
                "total_errors": 0,
                "error_kinds": {},
                "errors_preview": [],
            },
        )

    return CheckResult(
        name="label_format",
        severity=CheckSeverity.ERROR,
        summary=f"Found {total_errors} label format error(s) across {total_lines} lines.",
        details={
            "task_type": snapshot.task_type,
            "total_lines": total_lines,
            "total_errors": total_errors,
            "error_kinds": error_kinds,
            "errors_preview": errors_preview,
        },
    )
