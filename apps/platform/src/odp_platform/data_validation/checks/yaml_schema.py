"""YAML schema validation check."""

from __future__ import annotations

from typing import Any

import yaml

from odp_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check


def _normalize_names(raw_names: object) -> list[str] | None:
    if isinstance(raw_names, list):
        names = [str(item).strip() for item in raw_names]
        if all(name for name in names):
            return names
        return None

    if isinstance(raw_names, dict):
        ordered: list[tuple[int, str]] = []
        for raw_key, raw_value in raw_names.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                return None
            value = str(raw_value).strip()
            if not value:
                return None
            ordered.append((key, value))
        ordered.sort(key=lambda item: item[0])
        return [value for _, value in ordered]

    return None


@check("yaml_schema")
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    yaml_path = ctx.yaml_path
    if not yaml_path.exists():
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary="YAML file is missing.",
            details={"problems": [f"yaml file does not exist: {yaml_path}"]},
        )

    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary="YAML file cannot be parsed.",
            details={"problems": [str(exc)]},
        )

    if not isinstance(payload, dict):
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary="YAML top-level must be a mapping.",
            details={"problems": [f"yaml top-level must be dict, got {type(payload).__name__}"]},
        )

    problems: list[str] = []
    raw_nc = payload.get("nc")
    nc: int | None = None
    if isinstance(raw_nc, bool) or not isinstance(raw_nc, int) or raw_nc <= 0:
        problems.append("nc is required and must be a positive integer")
    else:
        nc = raw_nc

    raw_names = payload.get("names")
    if raw_names is None:
        problems.append("names is required")
        names: list[str] | None = None
    else:
        names = _normalize_names(raw_names)
        if names is None:
            problems.append("names must be list[str] or dict[int, str] with non-empty values")

    if nc is not None and names is not None and len(names) != nc:
        problems.append(f"nc ({nc}) does not match names length ({len(names)})")

    if problems:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary=f"YAML schema has {len(problems)} problem(s).",
            details={"problems": problems},
        )

    return CheckResult(
        name="yaml_schema",
        severity=CheckSeverity.PASS,
        summary="YAML schema is valid.",
        details={
            "nc": nc,
            "names_count": len(names or []),
        },
    )
