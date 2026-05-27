#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Core visualization data types."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass
class Detection:
    """Single detection value."""

    box: Tuple[int, int, int, int]
    confidence: float
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)


class LabelPosition(Enum):
    """Relative label position."""

    ABOVE = auto()
    INSIDE_TOP = auto()
    BELOW = auto()


@dataclass
class LabelLayout:
    """Calculated label layout."""

    box: Tuple[int, int, int, int]
    text_pos: Tuple[int, int]
    position: LabelPosition
    align_right: bool = False
    label_wider: bool = False


class DrawStyle(BaseModel):
    """Validated drawing style config."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=False,
    )

    font_path: Optional[str] = Field(default=None)
    font_size: int = Field(default=26, gt=0, le=500)
    line_width: int = Field(default=1, gt=0, le=50)
    padding_x: int = Field(default=6, ge=0, le=500)
    padding_y: int = Field(default=10, ge=0, le=500)
    radius: int = Field(default=3, ge=0, le=500)
    text_color: Tuple[int, int, int] = Field(default=(0, 0, 0))

    @field_validator("text_color")
    @classmethod
    def _validate_color(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        for channel in value:
            if not isinstance(channel, int) or not (0 <= channel <= 255):
                raise ValueError(
                    f"text_color 每个分量必须是 0-255 之间的整数,得到 {value}"
                )
        return value

    @classmethod
    def from_image_size(
        cls,
        height: int,
        width: int,
        ref_dim: int = 720,
        base_font_size: int = 26,
        base_line_width: int = 2,
        base_padding_x: int = 10,
        base_padding_y: int = 10,
        base_radius: int = 8,
        **kwargs,
    ) -> "DrawStyle":
        """Scale drawing style from image size."""
        scale = min(height, width) / max(ref_dim, 1)
        return cls(
            font_size=max(10, int(base_font_size * scale)),
            line_width=max(1, int(base_line_width * scale)),
            padding_x=max(5, int(base_padding_x * scale)),
            padding_y=max(5, int(base_padding_y * scale)),
            radius=max(3, int(base_radius * scale)),
            **kwargs,
        )
