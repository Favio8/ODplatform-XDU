"""CLI entrypoint for dataset validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from odp_platform.common.constants import TASK_DETECT, TASK_SEGMENT
from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.paths import dataset_yaml_path
from odp_platform.data_validation import render_to_logger, validate_dataset


LOGGER_NAME = __name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a prepared YOLO dataset and emit a structured report.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--dataset", help="Dataset name resolved via apps/platform/configs/datasets/<name>.yaml.")
    target_group.add_argument("--yaml", type=Path, help="Direct path to a dataset yaml file.")
    parser.add_argument("--task", choices=[TASK_DETECT, TASK_SEGMENT], default=None, help="Override the dataset task type.")
    parser.add_argument("--no-report", action="store_true", help="Skip writing runs/data_validation/<run_id>/report.json.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose DEBUG logging.")
    return parser


def _resolve_yaml_path(args: argparse.Namespace) -> Path:
    if args.yaml is not None:
        return Path(args.yaml)
    return dataset_yaml_path(args.dataset)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(log_type="data_validation", log_level=logging.DEBUG if args.verbose else None)
    logger = get_logger(LOGGER_NAME)

    try:
        yaml_path = _resolve_yaml_path(args)
        report = validate_dataset(
            yaml_path=yaml_path,
            task_type=args.task,
            write_report=not args.no_report,
        )
        render_to_logger(report, logger)
        return report.exit_code
    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user.")
        return 3
    except Exception:
        logger.exception("Dataset validation crashed unexpectedly.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
