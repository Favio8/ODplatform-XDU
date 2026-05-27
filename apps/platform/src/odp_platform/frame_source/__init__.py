#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""frame_source public API."""
from __future__ import annotations

from .core.base import FrameSource
from .core.config import CameraBackend, CameraCodec, CameraConfig
from .core.types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)
from .factory import create_async_source, create_frame_source, create_threaded_source
from .overlay import FPSCounter, Metrics, RateCounter, draw_hud, draw_pause
from .sources.camera import CameraSource
from .sources.image import ImageFolderSource, ImageSource
from .sources.video import VideoSource
from .wrappers.aio import AsyncSource
from .wrappers.threaded import BufferStrategy, ThreadedSource

__version__ = "2.0.0"
__author__ = "雨霓同学"

__all__ = [
    "__version__",
    "__author__",
    "SourceType",
    "FrameInfo",
    "Frame",
    "FrameSource",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "CameraConfig",
    "CameraBackend",
    "CameraCodec",
    "CameraSource",
    "VideoSource",
    "ImageSource",
    "ImageFolderSource",
    "ThreadedSource",
    "BufferStrategy",
    "AsyncSource",
    "create_frame_source",
    "create_threaded_source",
    "create_async_source",
    "FPSCounter",
    "RateCounter",
    "Metrics",
    "draw_hud",
    "draw_pause",
]
