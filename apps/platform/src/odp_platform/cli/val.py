"""Validation CLI configuration preflight."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from odp_platform.common.constants import RUNTIME_TASK_VAL, SUPPORTED_TASKS
from odp_platform.common.logging_utils import ROOT_LOGGER_NAME, get_logger, setup_logging
from odp_platform.common.system_utils import log_device_info
from odp_platform.runtime_config import ConfigBuildError, ConfigLoadError, build_config
from odp_platform.runtime_config.base import ConfigTrace, RuntimeConfigBase


LOGGER_NAME = f"{ROOT_LOGGER_NAME}.cli.val"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and audit an ODPlatform validation configuration without starting evaluation."
    )
    parser.add_argument("--config", "--yaml", dest="config", type=Path, help="Validation YAML config path.")
    parser.add_argument("--model", help="Model weights or model identifier.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--task", dest="task_type", choices=list(SUPPORTED_TASKS), help="Validation task semantics.")
    parser.add_argument("--experiment-name", help="ODPlatform experiment label.")
    parser.add_argument("--project", help="Output project directory.")
    parser.add_argument("--name", help="Ultralytics run name.")
    parser.add_argument("--device", help="Device expression, for example cpu, 0, or 0,1.")
    parser.add_argument("--imgsz", type=int, help="Validation image size.")
    parser.add_argument("--batch", type=int, help="Validation batch size.")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Maximum detections per image.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-json", dest="save_json", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-hybrid", dest="save_hybrid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dnn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level for the preflight, for example INFO or DEBUG.",
    )
    return parser


def _log_config_info(logger: logging.Logger, config: RuntimeConfigBase, trace: ConfigTrace) -> None:
    logger.info("开始记录验证参数信息".center(72, "="))
    for field_name in config.to_runtime_dict():
        field_trace = trace.get(field_name)
        display_name = config.external_field_name(field_name)
        logger.info(
            "%-20s : %s  (来源: %s)",
            display_name,
            field_trace.final_value,
            field_trace.final_source_label,
        )
    logger.info(trace.get_source_report())
    logger.info(trace.get_conflict_report())


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(
        log_type="val",
        model_name=args.model or (args.config.stem if args.config else None),
        log_level=args.log_level,
    )
    logger = get_logger(LOGGER_NAME)
    log_device_info(logger)

    try:
        config, trace, warnings = build_config(
            task_kind=RUNTIME_TASK_VAL,
            yaml_path=str(args.config) if args.config else None,
            cli_args=args,
            ignored_cli_keys={"config", "log_level"},
        )
    except (ConfigLoadError, ConfigBuildError) as exc:
        logger.error("验证配置构建失败: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("验证配置构建被用户中断。")
        return 3
    except Exception:
        logger.exception("验证配置构建发生未预期异常。")
        return 3

    logger.info("验证配置预检开始".center(72, "="))
    logger.info("运行模式: 仅构建配置并记录参数溯源，不启动真实验证")
    for warning in warnings:
        logger.warning("%s: %s", warning.field_name, warning.message)

    _log_config_info(logger, config, trace)
    logger.info("Ultralytics kwargs: %s", config.to_ultralytics_kwargs())
    logger.info("验证配置预检完成".center(72, "="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
