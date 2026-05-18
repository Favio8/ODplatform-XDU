"""Configuration loading placeholders."""

from __future__ import annotations

from pathlib import Path


def load_config(config_path: str | Path) -> Path:
    return Path(config_path)
