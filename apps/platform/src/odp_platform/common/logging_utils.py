"""Logging helpers for ODPlatform."""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from colorlog import ColoredFormatter

from odp_platform.common.paths import PLATFORM_LOGGING_DIR


ROOT_LOGGER_NAME: Final[str] = "odp_platform"
DEFAULT_LOG_LEVEL: Final[int] = logging.INFO
LOG_LEVEL_ENV_VAR: Final[str] = "ODP_LOG_LEVEL"
_LOGGER_CONFIGURED_ATTR: Final[str] = "_odp_logging_configured"

_STANDARD_LOG_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
    | {
        "message",
        "asctime",
        "log_color",
        "reset",
        "blue",
        "cyan",
        "green",
        "purple",
        "red",
        "white",
        "yellow",
    }
)


class _ContextFilter(logging.Filter):
    """Attach static logging context to each emitted record."""

    def __init__(self, context: dict[str, Any]) -> None:
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _JsonLineFormatter(logging.Formatter):
    """Serialize log records as JSON Lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "process": record.process,
            "thread": record.thread,
            "log_type": getattr(record, "log_type", None),
            "model_name": getattr(record, "model_name", None),
            "temp_log": getattr(record, "temp_log", None),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = self._normalize_value(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
        return value


def _resolve_log_level(log_level: int | str | None) -> int:
    if isinstance(log_level, int):
        return log_level

    candidate: str | None
    if isinstance(log_level, str):
        candidate = log_level
    else:
        candidate = os.getenv(LOG_LEVEL_ENV_VAR)

    if candidate is None:
        return DEFAULT_LOG_LEVEL

    normalized = candidate.strip()
    if not normalized:
        return DEFAULT_LOG_LEVEL

    if normalized.isdigit():
        return int(normalized)

    level_name = normalized.upper()
    if level_name in logging._nameToLevel:
        return logging._nameToLevel[level_name]

    raise ValueError(f"Unsupported log level: {candidate!r}")


def _sanitize_filename_component(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in "_-" else "_" for char in value)
    return sanitized.strip("_") or "unknown"


def _build_log_file_path(
    *,
    base_path: Path,
    log_type: str,
    model_name: str | None,
    temp_log: bool,
) -> Path:
    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    prefix = "temp" if temp_log else _sanitize_filename_component(log_type.replace(" ", "_"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    parts = [prefix, timestamp]
    if model_name:
        parts.append(_sanitize_filename_component(model_name))

    return log_dir / f"{'_'.join(parts)}.jsonl"


def _build_console_handler(log_level: int) -> logging.Handler:
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s%(reset)s | "
        "%(log_color)s%(levelname)-8s%(reset)s | "
        "%(cyan)s%(name)s%(reset)s | "
        "%(blue)s%(filename)s:%(lineno)d%(reset)s | "
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red,bg_white",
        },
    )
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(log_file: Path, log_level: int, encoding: str) -> logging.Handler:
    formatter = _JsonLineFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    handler = logging.FileHandler(log_file, encoding=encoding)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    return handler


def setup_logging(
    base_path: Path | None = None,
    log_type: str = "general",
    model_name: str | None = None,
    log_level: int | str | None = None,
    temp_log: bool = False,
    encoding: str = "utf-8",
    logger_name: str = ROOT_LOGGER_NAME,
) -> logging.Logger:
    """Configure the ODPlatform logger tree once per process."""

    logger = logging.getLogger(logger_name)
    if getattr(logger, _LOGGER_CONFIGURED_ATTR, False):
        return logger

    resolved_base_path = Path(base_path) if base_path is not None else PLATFORM_LOGGING_DIR
    resolved_level = _resolve_log_level(log_level)
    log_file = _build_log_file_path(
        base_path=resolved_base_path,
        log_type=log_type,
        model_name=model_name,
        temp_log=temp_log,
    )

    context_filter = _ContextFilter(
        {
            "log_type": log_type,
            "model_name": model_name,
            "temp_log": temp_log,
        }
    )

    console_handler = _build_console_handler(resolved_level)
    file_handler = _build_file_handler(log_file, resolved_level, encoding)
    console_handler.addFilter(context_filter)
    file_handler.addFilter(context_filter)

    logger.setLevel(resolved_level)
    logger.propagate = False
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    setattr(logger, _LOGGER_CONFIGURED_ATTR, True)
    setattr(logger, "_odp_log_file", log_file)

    initialization_record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        "Logging initialized.",
        args=(),
        exc_info=None,
        extra={
            "log_file": str(log_file),
            "resolved_level": logging.getLevelName(resolved_level),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
    )
    for handler in logger.handlers:
        handler.handle(initialization_record)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the requested logger or the project root logger by default."""

    return logging.getLogger(name or ROOT_LOGGER_NAME)


def reset_logging(logger_name: str = ROOT_LOGGER_NAME) -> None:
    """Remove and close handlers from a configured logger."""

    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.filters.clear()
    logger.propagate = True
    setattr(logger, _LOGGER_CONFIGURED_ATTR, False)
    if hasattr(logger, "_odp_log_file"):
        delattr(logger, "_odp_log_file")


__all__ = [
    "DEFAULT_LOG_LEVEL",
    "LOG_LEVEL_ENV_VAR",
    "ROOT_LOGGER_NAME",
    "get_logger",
    "reset_logging",
    "setup_logging",
]
