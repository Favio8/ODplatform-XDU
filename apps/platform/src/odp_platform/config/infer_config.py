"""Inference configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseConfig


@dataclass(slots=True)
class InferConfig(BaseConfig):
    conf_threshold: float = 0.25
