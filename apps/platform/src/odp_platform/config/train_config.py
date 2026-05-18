"""Training configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseConfig


@dataclass(slots=True)
class TrainConfig(BaseConfig):
    epochs: int = 100
    image_size: int = 640
