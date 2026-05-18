"""Configuration merge placeholders."""

from __future__ import annotations

from typing import Any


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        merged.update(config)
    return merged
