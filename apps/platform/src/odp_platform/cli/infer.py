"""Inference CLI configuration preflight."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from odp_platform.common.constants import RUNTIME_TASK_INFER, SUPPORTED_TASKS
from odp_platform.common.logging_utils import ROOT_LOGGER_NAME, get_logger, setup_logging
from odp_platform.common.system_utils import log_device_info
from odp_platform.runtime_config import ConfigBuildError, ConfigLoadError, build_config
from odp_platform.runtime_config.base import ConfigTrace, RuntimeConfigBase


LOGGER_NAME = f"{ROOT_LOGGER_NAME}.cli.infer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and audit an ODPlatform inference configuration without starting prediction."
    )
    parser.add_argument("--config", "--yaml", dest="config", type=Path, help="Inference YAML config path.")
    parser.add_argument("--model", help="Model weights or model identifier.")
    parser.add_argument("--data", help="Dataset YAML path when needed by the model.")
    parser.add_argument("--source", help="Input source, for example image path, video path, URL, or camera index.")
    parser.add_argument("--task", dest="task_type", choices=list(SUPPORTED_TASKS), help="Inference task semantics.")
    parser.add_argument("--experiment-name", help="ODPlatform experiment label.")
    parser.add_argument("--project", help="Output project directory.")
    parser.add_argument("--name", help="Ultralytics run name.")
    parser.add_argument("--device", help="Device expression, for example cpu, 0, or 0,1.")
    parser.add_argument("--imgsz", type=int, help="Inference image size.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Maximum detections per image.")
    parser.add_argument("--vid-stride", dest="vid_stride", type=int, help="Frame stride for video inputs.")
    parser.add_argument("--line-width", dest="line_width", type=int, help="Explicit line width for rendering.")
    parser.add_argument("--save-txt", dest="save_txt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-conf", dest="save_conf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-crop", dest="save_crop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-frames", dest="save_frames", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-labels", dest="show_labels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-conf", dest="show_conf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-boxes", dest="show_boxes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--agnostic-nms", dest="agnostic_nms", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stream-buffer", dest="stream_buffer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--retina-masks", dest="retina_masks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level for the preflight, for example INFO or DEBUG.",
    )
    return parser


def _log_config_info(logger: logging.Logger, config: RuntimeConfigBase, trace: ConfigTrace) -> None:
    logger.info("开始记录推理参数信息".center(72, "="))
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
        log_type="infer",
        model_name=args.model or (args.config.stem if args.config else None),
        log_level=args.log_level,
    )
    logger = get_logger(LOGGER_NAME)
    log_device_info(logger)

    try:
        config, trace, warnings = build_config(
            task_kind=RUNTIME_TASK_INFER,
            yaml_path=str(args.config) if args.config else None,
            cli_args=args,
            ignored_cli_keys={"config", "log_level"},
        )
    except (ConfigLoadError, ConfigBuildError) as exc:
        logger.error("推理配置构建失败: %s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("推理配置构建被用户中断。")
        return 3
    except Exception:
        logger.exception("推理配置构建发生未预期异常。")
        return 3

    logger.info("推理配置预检开始".center(72, "="))
    logger.info("运行模式: 仅构建配置并记录参数溯源，不启动真实推理")
    for warning in warnings:
        logger.warning("%s: %s", warning.field_name, warning.message)

    _log_config_info(logger, config, trace)
    logger.info("Ultralytics kwargs: %s", config.to_ultralytics_kwargs())
    logger.info("推理配置预检完成".center(72, "="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
