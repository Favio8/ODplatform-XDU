"""CLI entrypoint for runtime config template generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from odp_platform.common.constants import SUPPORTED_RUNTIME_TASKS
from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.config.generator import generate_template


LOGGER_NAME = __name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a commented ODPlatform runtime config template.")
    parser.add_argument("--task", required=True, choices=list(SUPPORTED_RUNTIME_TASKS), help="Runtime task kind.")
    parser.add_argument("--output", type=Path, required=True, help="Target yaml path.")
    parser.add_argument("--force", action="store_true", help="Overwrite the target file if it already exists.")
    parser.add_argument("--no-backup", action="store_true", help="Disable automatic backup when overwriting.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(log_type="generate_config")
    logger = get_logger(LOGGER_NAME)

    existed_before = args.output.exists()
    path = generate_template(
        args.task,
        args.output,
        force=args.force,
        backup=not args.no_backup,
    )

    if existed_before and not args.force:
        logger.warning("Template already exists, skipped: %s", path)
        return 0

    logger.info("Config template ready: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
