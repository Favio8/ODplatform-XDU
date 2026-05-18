"""Base configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BaseConfig:
    name: str = "default"
    extras: dict[str, Any] = field(default_factory=dict)
