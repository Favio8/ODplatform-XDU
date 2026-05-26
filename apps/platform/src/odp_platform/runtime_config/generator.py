"""Teacher-style runtime config template generator compatibility layer."""

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
from odp_platform.config.base import RuntimeConfigBase
from odp_platform.runtime_config.infer import YOLOInferConfig
from odp_platform.runtime_config.train import YOLOTrainConfig
from odp_platform.runtime_config.val import YOLOValConfig


LOGGER_NAME = __name__

_CONFIG_CLASS_MAP: dict[str, tuple[type[RuntimeConfigBase], str]] = {
    "train": (YOLOTrainConfig, "YOLO 训练配置"),
    "val": (YOLOValConfig, "YOLO 验证配置"),
    "infer": (YOLOInferConfig, "YOLO 推理配置"),
}


def _serialize_scalar(value: Any) -> str:
    dumped = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    return dumped.splitlines()[0] if dumped.splitlines() else "null"


class ConfigGenerator:
    """Generate YAML templates from runtime config metadata."""

    def __init__(self, indent: int = 2):
        self.indent = indent

    def generate(
        self,
        config_class: type[RuntimeConfigBase],
        output_path: str | Path,
        *,
        overwrite: bool = False,
        backup: bool = True,
        title: str | None = None,
    ) -> bool:
        target = Path(output_path)
        resolved_title = title or self._default_title_for(config_class)
        if target.exists() and not overwrite:
            return False

        if target.exists() and overwrite and backup:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = target.with_name(f"{target.name}.bak.{stamp}")
            shutil.copy2(target, backup_path)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._generate_yaml(config_class, title=resolved_title), encoding="utf-8")
        return True

    def _generate_yaml(
        self,
        config_class: type[RuntimeConfigBase],
        *,
        title: str | None = None,
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# " + "=" * 78,
            f"# {title or config_class.__name__}",
            f"# 自动生成时间: {now}",
            "# " + "=" * 78,
            "",
        ]

        for group_name, entries in config_class.examples_by_group().items():
            visible_entries = [(field_name, spec) for field_name, spec in entries if not spec.internal]
            if not visible_entries:
                continue
            lines.append(f"# [{group_name}]")
            for field_name, spec in visible_entries:
                output_name = config_class.external_field_name(field_name)
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

    @staticmethod
    def _default_title_for(config_class: type[RuntimeConfigBase]) -> str:
        task_kind = config_class.task_kind_name()
        if task_kind in _CONFIG_CLASS_MAP:
            return _CONFIG_CLASS_MAP[task_kind][1]
        return config_class.__name__


def generate_template(
    task_kind: str,
    output_path: Path | None = None,
    *,
    force: bool = False,
    backup: bool = True,
) -> Path:
    config_class, title = _CONFIG_CLASS_MAP[task_kind]
    target = Path(output_path) if output_path is not None else runtime_config_path(task_kind)
    generator = ConfigGenerator()
    generator.generate(
        config_class,
        target,
        overwrite=force,
        backup=backup,
        title=title,
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="从 Pydantic 配置类反射生成 YOLO 运行配置 YAML 模板",
    )
    parser.add_argument("name", nargs="?", choices=list(SUPPORTED_RUNTIME_TASKS), help="要生成的配置名")
    parser.add_argument("--task", dest="task", choices=list(SUPPORTED_RUNTIME_TASKS), help="兼容旧调用的任务名参数")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出路径 (默认: configs/runtime/<name>.yaml)")
    parser.add_argument("--overwrite", "--force", dest="overwrite", action="store_true", help="覆盖已有文件")
    parser.add_argument("--no-backup", action="store_true", help="覆盖时不备份原文件")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task_kind = args.name or args.task
    if task_kind is None:
        parser.error("one of NAME or --task is required")

    setup_logging(log_type="generate_config")
    logger = get_logger(LOGGER_NAME)

    config_class, title = _CONFIG_CLASS_MAP[task_kind]
    output_path = Path(args.output) if args.output is not None else runtime_config_path(task_kind)
    generator = ConfigGenerator()
    generated = generator.generate(
        config_class,
        output_path,
        overwrite=args.overwrite,
        backup=not args.no_backup,
        title=title,
    )

    if generated:
        logger.info("Runtime config template ready: %s", output_path)
    else:
        logger.warning("Template already exists, skipped: %s", output_path)
    return 0


__all__ = ["ConfigGenerator", "build_parser", "generate_template", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
