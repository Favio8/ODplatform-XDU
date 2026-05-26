"""CLI command for preparing the platform workspace directories."""

from __future__ import annotations

import logging
from pathlib import Path

from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.performance_utils import time_it
from odp_platform.common.paths import LOGGING_DIR, RAW_DATASETS_DIR, ROOT_DIR, get_dirs_to_initialize
from odp_platform.common.string_utils import format_table_row, format_table_separator


LINE_WIDTH = 60
LOGGER_NAME = "odp_platform.cli.init_project"
logger = logging.getLogger(LOGGER_NAME)


def _check_raw_data_status() -> list[str]:
    """Inspect the shared raw-dataset root and return status messages."""

    raw_status: list[str] = []
    rel_raw = RAW_DATASETS_DIR.relative_to(ROOT_DIR)

    if not RAW_DATASETS_DIR.exists():
        logger.warning(
            "原始数据集根目录不存在: %s\n请按 data/raw/<dataset_name> 的结构放置原始数据，例如 data/raw/rsod。",
            RAW_DATASETS_DIR,
        )
        raw_status.append(f"{rel_raw} 不存在 -> 请按 data/raw/<dataset_name> 放入原始数据集")
        return raw_status

    entries = sorted(
        (path for path in RAW_DATASETS_DIR.iterdir() if not path.name.startswith(".")),
        key=lambda path: path.name,
    )
    if not entries:
        logger.warning("原始数据集根目录为空: %s", RAW_DATASETS_DIR)
        raw_status.append(f"{rel_raw} 为空 -> 请创建至少一个 data/raw/<dataset_name> 子目录")
        return raw_status

    directories = [path for path in entries if path.is_dir()]
    files = [path for path in entries if path.is_file()]
    logger.info(
        "原始数据集根目录就绪: %s 个数据集目录, %s 个文件",
        len(directories),
        len(files),
    )
    raw_status.append(f"{rel_raw} 就绪 -> {len(directories)} 个数据集目录, {len(files)} 个文件")

    for path in directories:
        raw_status.append(f"数据集目录: {path.name}")
    for path in files:
        raw_status.append(f"文件: {path.name}")
    return raw_status


@time_it(iterations=1, name="项目初始化", logger_instance=logger)
def initialize_project() -> None:
    """Ensure the core platform directories exist and summarize the result."""

    setup_logging(base_path=LOGGING_DIR, log_type="init_project", temp_log=False)
    project_logger = get_logger(LOGGER_NAME)

    project_logger.info("%s", "开始初始化项目核心目录".center(LINE_WIDTH, "="))
    project_logger.info("项目根目录: %s", ROOT_DIR)

    created: list[Path] = []
    existed: list[Path] = []

    for directory in get_dirs_to_initialize():
        relative_directory = directory.relative_to(ROOT_DIR)
        if directory.exists():
            project_logger.info("目录已存在: %s", relative_directory)
            existed.append(directory)
            continue

        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            project_logger.error("创建目录失败: %s: %s", relative_directory, exc)
            raise SystemExit(1) from exc

        project_logger.info("成功创建目录: %s", relative_directory)
        created.append(directory)

    project_logger.info("%s", "检查原始数据目录状态".center(LINE_WIDTH, "="))
    raw_status = _check_raw_data_status()

    project_logger.info("%s", "项目核心目录初始化完成".center(LINE_WIDTH, "="))
    widths = [40, 8]
    aligns = ["left", "right"]
    project_logger.info("%s", format_table_row(["目录", "状态"], widths, aligns))
    project_logger.info("%s", format_table_separator(widths))

    for directory in created:
        project_logger.info(
            "%s",
            format_table_row([str(directory.relative_to(ROOT_DIR)), "新建"], widths, aligns),
        )
    for directory in existed:
        project_logger.info(
            "%s",
            format_table_row([str(directory.relative_to(ROOT_DIR)), "已存在"], widths, aligns),
        )

    project_logger.info("%s", "原始数据状态".center(LINE_WIDTH, "="))
    for status in raw_status:
        project_logger.info("%s", status)
    project_logger.info("%s", "=" * LINE_WIDTH)


def main() -> int:
    initialize_project()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
