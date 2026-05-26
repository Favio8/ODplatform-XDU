"""Real training CLI for the D6 training subsystem."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from odp_platform.common.constants import SUPPORTED_TASKS
from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.training import train_yolo


LOGGER_NAME = __name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ODPlatform YOLO training job.")
    parser.add_argument("--config", "--yaml", dest="config", type=Path, help="Training YAML config path.")
    parser.add_argument("--model", help="Model weights or model identifier.")
    parser.add_argument("--data", help="Dataset YAML path or dataset identifier.")
    parser.add_argument("--task", dest="task_type", choices=list(SUPPORTED_TASKS), help="Training task semantics.")
    parser.add_argument("--experiment-name", help="ODPlatform experiment label.")
    parser.add_argument("--project", help="Output project directory.")
    parser.add_argument("--name", help="Ultralytics run name.")
    parser.add_argument("--device", help="Device expression, for example cpu, 0, or 0,1.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument("--imgsz", type=int, help="Image size.")
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--lr0", type=float, help="Initial learning rate.")
    parser.add_argument("--exist-ok", dest="exist_ok", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--patience", type=int, help="Early stopping patience.")
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, help="Last epochs without mosaic.")
    parser.add_argument("--no-pre-validate", action="store_true", help="Skip D4 dataset validation before training.")
    parser.add_argument("--no-archive", action="store_true", help="Skip checkpoint archival to models/checkpoints.")
    parser.add_argument("--no-rename-log", action="store_true", help="Keep the original D2 log filename.")
    parser.add_argument("--academic-plots", action="store_true", help="Apply academic matplotlib plot styling.")
    parser.add_argument("--log-level", default=None, help="Logging level such as INFO or DEBUG.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(
        log_type="train",
        model_name=args.model or (args.config.stem if args.config else None),
        log_level=args.log_level,
    )
    logger = get_logger(LOGGER_NAME)

    try:
        result = train_yolo(
            yaml_path=args.config,
            cli_args=args,
            pre_validate=not args.no_pre_validate,
            archive=not args.no_archive,
            rename_log=not args.no_rename_log,
            academic_plots=args.academic_plots,
        )
    except KeyboardInterrupt:
        logger.warning("训练被用户中断。")
        return 3
    except Exception:
        logger.exception("训练入口发生未预期异常。")
        return 3

    if result.success:
        logger.info("训练成功，输出目录: %s", result.output_dir)
        return 0

    logger.error("训练失败: %s", result.error)
    return 2


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
