"""Validation configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseConfig


@dataclass(slots=True)
class ValConfig(BaseConfig):
    batch_size: int = 16
