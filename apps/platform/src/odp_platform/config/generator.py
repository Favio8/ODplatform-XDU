"""Generate commented YAML templates from config metadata."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from odp_platform.common.constants import (
    RUNTIME_TASK_INFER,
    RUNTIME_TASK_TRAIN,
    RUNTIME_TASK_VAL,
)
from odp_platform.common.paths import CONFIGS_DIR
from odp_platform.config.base import RuntimeConfigBase, get_config_class


def _default_template_path(task_kind: str) -> Path:
    return CONFIGS_DIR / f"{task_kind}.example.yaml"


def _serialize_scalar(value: Any) -> str:
    dumped = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    return dumped.splitlines()[0] if dumped.splitlines() else "null"


def _build_template_text(config_cls: type[RuntimeConfigBase], task_kind: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# ODPlatform {task_kind} runtime config template",
        f"# Generated at: {now}",
        "# Edit the values below, then load the file through the runtime configuration subsystem.",
        "",
    ]

    for group_name, entries in config_cls.examples_by_group().items():
        lines.append(f"# [{group_name}]")
        for field_name, spec in entries:
            if spec.internal:
                continue
            output_name = config_cls.external_field_name(field_name)
            lines.append(f"# {spec.description}")
            if spec.examples:
                lines.append("# Examples: " + ", ".join(repr(item) for item in spec.examples))
            for tip in spec.tuning_tips:
                lines.append(f"# Tip: {tip}")
            lines.append(f"{output_name}: {_serialize_scalar(spec.default)}")
            lines.append("")

    lines.extend(
        [
            "# FAQ",
            "# 1. If a YAML file is missing, regenerate it with odp-generate-config.",
            "# 2. Use experiment_name through CLI or builder inputs when you only need a custom run label.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def generate_template(
    task_kind: str,
    output_path: Path | None = None,
    *,
    force: bool = False,
    backup: bool = True,
) -> Path:
    """Generate one task-specific YAML template from config metadata."""

    config_cls = get_config_class(task_kind)
    target_path = Path(output_path) if output_path is not None else _default_template_path(task_kind)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        return target_path

    if target_path.exists() and force and backup:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = target_path.with_suffix(target_path.suffix + f".bak.{timestamp}")
        shutil.copy2(target_path, backup_path)

    target_path.write_text(_build_template_text(config_cls, task_kind), encoding="utf-8")
    return target_path


__all__ = [
    "generate_template",
]
