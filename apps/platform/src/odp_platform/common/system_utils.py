"""System inspection helpers."""

from __future__ import annotations

import platform
import sys


def python_runtime() -> str:
    return sys.version


def platform_summary() -> str:
    return platform.platform()
