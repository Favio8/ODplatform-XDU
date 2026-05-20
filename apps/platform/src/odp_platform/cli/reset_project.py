"""CLI command for resetting generated platform artifacts."""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.paths import (
    META_LOGGING_DIR,
    PRETRAINED_MODELS_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
    get_dirs_to_reset,
    is_protected,
)
from odp_platform.common.string_utils import format_table_row, format_table_separator


CONFIRM_KEYWORD = "RESET"
LINE_WIDTH = 70
LOGGER_NAME = __name__
logger = get_logger(LOGGER_NAME)


def _format_size(bytes_size: int) -> str:
    """Format a byte count using binary units."""

    if bytes_size >= 1024**3:
        return f"{bytes_size / (1024**3):.2f} GiB"
    if bytes_size >= 1024**2:
        return f"{bytes_size / (1024**2):.2f} MiB"
    if bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KiB"
    return f"{bytes_size} B"


def _on_rm_error(func, path, exc_info) -> None:
    """Retry deleting a read-only path after making it writable."""

    del exc_info
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def _audit_context() -> dict[str, Any]:
    """Collect lightweight audit metadata for reset operations."""

    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            encoding="utf-8",
            errors="ignore",
            stderr=subprocess.STDOUT,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_rev = "(not a git repo)"

    return {
        "user": getpass.getuser(),
        "pid": os.getpid(),
        "git_rev": git_rev,
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }


def _scan_targets() -> tuple[list[tuple[Path, int, int]], list[Path]]:
    """Return deletable targets plus skipped targets."""

    deletable: list[tuple[Path, int, int]] = []
    skipped: list[Path] = []

    for directory in get_dirs_to_reset():
        if is_protected(directory):
            logger.warning("拒绝处理受保护目录，已跳过: %s", directory)
            skipped.append(directory)
            continue

        if not directory.exists():
            skipped.append(directory)
            continue

        file_count = 0
        total_size = 0
        try:
            for entry in directory.rglob("*"):
                if entry.is_file():
                    file_count += 1
                    try:
                        total_size += entry.stat().st_size
                    except OSError:
                        pass
        except OSError as exc:
            logger.warning("扫描目录失败 %s: %s", directory, exc)

        deletable.append((directory, file_count, total_size))

    return deletable, skipped


def _print_plan(
    deletable: list[tuple[Path, int, int]],
    skipped: list[Path],
    *,
    will_actually_delete: bool,
) -> None:
    """Print a reset plan before any deletion occurs."""

    if will_actually_delete:
        logger.warning("%s", "即将删除以下目录".center(LINE_WIDTH, "="))
    else:
        logger.info("%s", "[DRY-RUN] 计划如下(未实际删除)".center(LINE_WIDTH, "="))

    if not deletable:
        logger.info("(没有可删除的目录，项目已经是干净状态)")
        return

    widths = [40, 12, 14]
    aligns = ["left", "right", "right"]
    logger.info("%s", format_table_row(["目录", "文件数", "大小"], widths, aligns))
    logger.info("%s", format_table_separator(widths))

    total_files = 0
    total_bytes = 0
    for path, count, size in deletable:
        logger.info(
            "%s",
            format_table_row([str(path.relative_to(ROOT_DIR)), str(count), _format_size(size)], widths, aligns),
        )
        total_files += count
        total_bytes += size

    logger.info("%s", format_table_separator(widths))
    logger.info(
        "%s",
        format_table_row(["【合计】", str(total_files), _format_size(total_bytes)], widths, aligns),
    )

    if skipped:
        logger.info("")
        logger.info("以下目录已跳过(不存在或受保护):")
        for path in skipped:
            logger.info("  - %s", path.relative_to(ROOT_DIR))

    logger.info("")
    logger.info("以下重要目录不会被动:")
    logger.info("  - 原始数据: %s/", RAW_DATA_DIR.relative_to(ROOT_DIR))
    logger.info("  - 预训练权重: %s/", PRETRAINED_MODELS_DIR.relative_to(ROOT_DIR))
    logger.info("  - 所有代码、文档、配置(进 git 的)")


def _confirm(deletable_count: int) -> bool:
    """Ask for explicit confirmation before deleting anything."""

    print()
    print("=" * LINE_WIDTH)
    print(f"⚠️  你正要删除 {deletable_count} 个目录的内容。这个操作不可撤销。")
    print(f"⚠️  如果确认，请精确输入大写的 '{CONFIRM_KEYWORD}'（其他任何输入都会取消）:")
    print("=" * LINE_WIDTH)
    try:
        user_input = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return user_input == CONFIRM_KEYWORD


def _delete_one(path: Path, idx: int, total: int, file_count: int, size: int) -> str | None:
    """Delete one target directory and return an error string on failure."""

    if is_protected(path):
        logger.error("[%s/%s] 删除前检查失败，跳过受保护目录: %s", idx, total, path)
        return "受保护目录"

    relative_path = path.relative_to(ROOT_DIR)
    size_text = _format_size(size)

    if size > 1024**3:
        logger.warning(
            "[%s/%s] 正在删除 %s (%s, %s 个文件)，这可能需要一会...",
            idx,
            total,
            relative_path,
            size_text,
            file_count,
        )
    else:
        logger.info(
            "[%s/%s] 删除 %s (%s, %s 个文件)",
            idx,
            total,
            relative_path,
            size_text,
            file_count,
        )

    try:
        shutil.rmtree(path, onerror=_on_rm_error)
        logger.info("[%s/%s] 已删除: %s", idx, total, relative_path)
        return None
    except OSError as exc:
        logger.error("[%s/%s] 删除失败 %s: %s", idx, total, relative_path, exc)
        return str(exc)


def _execute_delete(deletable: list[tuple[Path, int, int]]) -> None:
    """Delete all planned targets and summarize the result."""

    total = len(deletable)
    success: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for idx, (path, file_count, size) in enumerate(deletable, start=1):
        reason = _delete_one(path, idx, total, file_count, size)
        if reason is None:
            success.append(path)
        else:
            failed.append((path, reason))

    logger.info("%s", "=" * LINE_WIDTH)
    if failed:
        logger.warning("完成: 成功 %s 个，失败 %s 个", len(success), len(failed))
        for path, reason in failed:
            logger.warning("  - %s: %s", path.relative_to(ROOT_DIR), reason)
    else:
        logger.info("完成: 成功 %s 个，失败 0 个", len(success))


def reset_project(yes: bool = False, force: bool = False, dry_run: bool = False) -> int:
    """Preview or reset generated project artifacts."""

    setup_logging(base_path=META_LOGGING_DIR, log_type="reset_project", temp_log=False)

    logger.info("%s", "项目重置工具".center(LINE_WIDTH, "="))
    logger.info("项目根目录: %s", ROOT_DIR)

    context = _audit_context()
    logger.info(
        "审计上下文: user=%s pid=%s git=%s argv=%s cwd=%s",
        context["user"],
        context["pid"],
        str(context["git_rev"])[:8],
        context["argv"],
        context["cwd"],
    )

    if dry_run and yes:
        logger.warning("同时给了 --dry-run 和 --yes，将以 --dry-run 为准(只打印不删除)")
        yes = False

    deletable, skipped = _scan_targets()
    _print_plan(deletable, skipped, will_actually_delete=yes)

    if not deletable:
        return 0

    if not yes:
        logger.info("")
        if dry_run:
            logger.info("这是显式的 --dry-run。要真正执行删除，请加 --yes:")
        else:
            logger.info("这是 dry-run(默认行为)。要真正执行删除，请加 --yes:")
        logger.info("   python scripts/reset_project.py --yes")
        return 0

    if not force and not _confirm(len(deletable)):
        logger.warning("用户取消，未执行删除")
        return 1

    logger.info("")
    logger.info("%s", "开始删除...".center(LINE_WIDTH, "="))
    _execute_delete(deletable)
    return 0


def main() -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="重置项目工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="允许执行删除；默认只做 dry-run 预览",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="跳过交互式确认（需与 --yes 配合才会真的删除）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不删除任何目录",
    )
    args = parser.parse_args()
    return reset_project(
        yes=args.yes,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
