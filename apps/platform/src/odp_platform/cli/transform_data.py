"""CLI entrypoint for dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from odp_platform.common.constants import FORMAT_COCO, FORMAT_PASCAL_VOC, FORMAT_YOLO, SUPPORTED_TASKS
from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.paths import LOGGING_DIR
from odp_platform.data_pipeline import ConvertOptions, list_capabilities
from odp_platform.data_pipeline.orchestrator import prepare_dataset


LOGGER_NAME = __name__


def _build_epilog() -> str:
    capabilities = list_capabilities()
    lines = ["", "Capability Matrix:"]
    for source_format, tasks in capabilities.items():
        lines.append(f"  - {source_format}: {', '.join(tasks)}")
    return "\n".join(lines)


def _is_task_supported(source_format: str, task: str) -> bool:
    capabilities = list_capabilities()
    return task in capabilities.get(source_format, ())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform raw datasets into split YOLO training assets.",
        epilog=_build_epilog(),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name used for path resolution and yaml output.")
    parser.add_argument(
        "--format",
        required=True,
        choices=[FORMAT_PASCAL_VOC, FORMAT_COCO, FORMAT_YOLO],
        dest="source_format",
        help="Source dataset format.",
    )
    parser.add_argument("--task", default="detect", choices=list(SUPPORTED_TASKS), help="Target task type.")
    parser.add_argument("--classes", default=None, help="Comma-separated class names that override auto discovery.")
    parser.add_argument("--train-rate", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-rate", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-rate", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for deterministic splitting.")
    parser.add_argument("--source-root", type=Path, default=None, help="Override the raw dataset source root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override the final split data root.")
    parser.add_argument("--config-path", type=Path, default=None, help="Override the generated yaml path.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Minimum raw coverage required before conversion starts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(base_path=LOGGING_DIR, log_type="transform_data", temp_log=False)
    logger = get_logger(LOGGER_NAME)

    if not _is_task_supported(args.source_format, args.task):
        capabilities = list_capabilities()
        supported_tasks = ", ".join(capabilities.get(args.source_format, ()))
        logger.error(
            "Dataset transform failed: source format %s does not support task %s; supported tasks: %s",
            args.source_format,
            args.task,
            supported_tasks or "(none)",
        )
        return 1

    classes = None
    if args.classes:
        classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    options = ConvertOptions(
        dataset_name=args.dataset,
        source_format=args.source_format,
        task=args.task,
        classes=classes,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        test_rate=args.test_rate,
        random_state=args.random_state,
        source_root=args.source_root,
    )

    try:
        result = prepare_dataset(
            options,
            min_coverage=args.min_coverage,
            data_root=args.data_root,
            yaml_path=args.config_path,
        )
    except Exception as exc:
        logger.error("Dataset transform failed: %s", exc)
        return 1

    split_counts = {split: len(samples) for split, samples in result.split_map.items()}
    logger.info("Dataset transform finished: %s", options.dataset_name)
    logger.info("Coverage: %.2f%%", result.coverage.coverage * 100.0)
    logger.info("Split counts: %s", split_counts)
    logger.info("Generated yaml: %s", result.yaml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
