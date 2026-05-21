"""System inspection helpers."""

from __future__ import annotations

import logging
import platform
import sys


def python_runtime() -> str:
    return sys.version


def platform_summary() -> str:
    return platform.platform()


def log_device_info(logger: logging.Logger) -> None:
    """Emit lightweight runtime metadata for the current process."""

    logger.info("Runtime: Python %s", platform.python_version())
    logger.info("Platform: %s", platform_summary())


__all__ = [
    "log_device_info",
    "platform_summary",
    "python_runtime",
]
