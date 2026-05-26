"""Generate commented YAML templates from config metadata."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from odp_platform.common.constants import SUPPORTED_RUNTIME_TASKS
from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.paths import runtime_config_path
from odp_platform.runtime_config.base import RuntimeConfigBase, get_config_class


LOGGER_NAME = __name__


def _default_template_path(task_kind: str) -> Path:
    return runtime_config_path(task_kind)


def _serialize_scalar(value: Any) -> str:
    dumped = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    return dumped.splitlines()[0] if dumped.splitlines() else "null"


def _build_template_text(config_cls: type[RuntimeConfigBase], task_kind: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# " + "=" * 78,
        f"# ODPlatform {task_kind} runtime config",
        f"# 自动生成时间: {now}",
        "# " + "=" * 78,
        "",
    ]

    for group_name, entries in config_cls.examples_by_group().items():
        visible_entries = [(field_name, spec) for field_name, spec in entries if not spec.internal]
        if not visible_entries:
            continue
        lines.append(f"# [{group_name}]")
        for field_name, spec in visible_entries:
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
            "# 1. If a YAML file is missing, regenerate it with odp-gen-config <train|val|infer>.",
            "# 2. CLI values override YAML, and YAML overrides defaults.",
            "# 3. Use experiment_name through CLI or builder inputs when you only need a custom run label.",
            "# 4. To refresh a template safely, rerun odp-gen-config <name> --overwrite.",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="Generate a commented ODPlatform runtime config template.",
    )
    parser.add_argument(
        "name",
        nargs="?",
        choices=list(SUPPORTED_RUNTIME_TASKS),
        help="Template name: train, val, or infer.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        choices=list(SUPPORTED_RUNTIME_TASKS),
        help="Compatibility flag for selecting the template task kind.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Target yaml path. Defaults to apps/platform/configs/runtime/<name>.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        "--force",
        dest="overwrite",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup when overwriting an existing file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task_kind = args.name or args.task
    if task_kind is None:
        parser.error("one of NAME or --task is required")

    setup_logging(log_type="generate_config")
    logger = get_logger(LOGGER_NAME)

    output_path = args.output if args.output is not None else runtime_config_path(task_kind)
    existed_before = output_path.exists()
    path = generate_template(
        task_kind,
        output_path,
        force=args.overwrite,
        backup=not args.no_backup,
    )

    if existed_before and not args.overwrite:
        logger.warning("Template already exists, skipped: %s", path)
        return 0

    logger.info("Runtime config template ready: %s", path)
    return 0


__all__ = [
    "build_parser",
    "generate_template",
    "main",
]
