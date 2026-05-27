#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Core data types for frame sources."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SourceType(str, Enum):
    """Supported input source categories."""

    CAMERA = "camera"
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_FOLDER = "image_folder"


IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
)


@dataclass(frozen=True)
class FrameInfo:
    """Frame metadata."""

    width: int
    height: int
    source_type: SourceType
    source_path: str
    frame_index: int = 0
    total_frames: Optional[int] = None
    timestamp: float = 0.0
    fps: Optional[float] = None
    filename: Optional[str] = None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class Frame:
    """Unified frame payload."""

    image: np.ndarray
    info: FrameInfo

    @property
    def resolution(self) -> tuple[int, int]:
        return self.info.resolution

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height
