"""Rename the D2 project log file so it matches the final Ultralytics save_dir."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from odp_platform.common.logging_utils import ROOT_LOGGER_NAME


logger = logging.getLogger(__name__)


def _sanitize(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in value)
    return cleaned.strip("_") or "unknown"


def _build_target_path(old_path: Path, save_dir: Path, model_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_name = f"{_sanitize(save_dir.name)}_{timestamp}_{_sanitize(model_stem)}{old_path.suffix or '.jsonl'}"
    return old_path.with_name(new_name)


def _next_available_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def rename_log_to_save_dir(save_dir: str | Path, model_stem: str) -> Path | None:
    """Rename the project file log and swap in a fresh handler on the named root logger.

    This function only operates on the D2 named root logger instead of the unnamed
    Python root logger, which keeps the logging boundary aligned with the rest of
    the project.
    """

    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    file_handlers = [handler for handler in root_logger.handlers if isinstance(handler, logging.FileHandler)]
    if not file_handlers:
        logger.warning(
            "'%s' 根 logger 上没有 FileHandler，跳过日志改名。",
            ROOT_LOGGER_NAME,
        )
        return None

    old_handler = file_handlers[0]
    old_path = Path(old_handler.baseFilename)
    target_path = _next_available_path(_build_target_path(old_path, Path(save_dir), model_stem))

    formatter = old_handler.formatter
    filters = list(old_handler.filters)
    level = old_handler.level
    encoding = getattr(old_handler, "encoding", "utf-8") or "utf-8"

    try:
        old_handler.flush()
    except Exception:
        pass

    root_logger.removeHandler(old_handler)
    old_handler.close()

    try:
        if old_path.exists() and old_path != target_path:
            old_path.replace(target_path)
        replacement = logging.FileHandler(target_path, encoding=encoding)
        replacement.setLevel(level)
        if formatter is not None:
            replacement.setFormatter(formatter)
        for current_filter in filters:
            replacement.addFilter(current_filter)
        root_logger.addHandler(replacement)
        setattr(root_logger, "_odp_log_file", target_path)
        root_logger.info("日志文件已重命名: %s", target_path.name)
        return target_path
    except Exception as exc:
        fallback = logging.FileHandler(old_path, encoding=encoding)
        fallback.setLevel(level)
        if formatter is not None:
            fallback.setFormatter(formatter)
        for current_filter in filters:
            fallback.addFilter(current_filter)
        root_logger.addHandler(fallback)
        setattr(root_logger, "_odp_log_file", old_path)
        logger.warning("日志改名失败，已回退到原日志文件: %s", exc)
        return None


__all__ = ["rename_log_to_save_dir"]
