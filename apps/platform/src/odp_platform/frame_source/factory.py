#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Factory helpers for frame sources."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .core.base import FrameSource
from .core.config import CameraConfig
from .core.types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .sources.camera import CameraSource
from .sources.image import ImageFolderSource, ImageSource
from .sources.video import VideoSource
from .wrappers.aio import AsyncSource
from .wrappers.threaded import BufferStrategy, ThreadedSource


def create_frame_source(
    source: str,
    camera_config: Optional[CameraConfig] = None,
) -> FrameSource:
    """Create the appropriate frame source from a string identifier."""
    if source.isdigit():
        camera_id = int(source)
        if camera_config is None:
            config = CameraConfig(camera_id=camera_id)
        else:
            config = camera_config.model_copy(update={"camera_id": camera_id})
        return CameraSource(config)

    path = Path(source)
    if not path.exists():
        raise ValueError(f"路径不存在: {source}")

    if path.is_dir():
        return ImageFolderSource(source)

    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return VideoSource(source)
    if ext in IMAGE_EXTENSIONS:
        return ImageSource(source)

    raise ValueError(
        f"不支持的文件格式: '{ext}'\n"
        f"  支持的视频格式: {sorted(VIDEO_EXTENSIONS)}\n"
        f"  支持的图片格式: {sorted(IMAGE_EXTENSIONS)}"
    )


def create_threaded_source(
    source: str,
    camera_config: Optional[CameraConfig] = None,
    *,
    buffer: BufferStrategy = "latest",
    buffer_size: int = 1,
    warmup_frames: int = 0,
    read_timeout: float = 5.0,
) -> ThreadedSource:
    """Create a threaded source wrapper."""
    inner = create_frame_source(source, camera_config=camera_config)
    return ThreadedSource(
        inner,
        buffer=buffer,
        buffer_size=buffer_size,
        warmup_frames=warmup_frames,
        read_timeout=read_timeout,
    )


def create_async_source(
    source: str,
    camera_config: Optional[CameraConfig] = None,
    *,
    threaded: bool = True,
    buffer: BufferStrategy = "latest",
    buffer_size: int = 1,
    warmup_frames: int = 0,
    read_timeout: float = 5.0,
) -> AsyncSource:
    """Create an async wrapper, optionally with threaded capture."""
    inner: FrameSource = create_frame_source(source, camera_config=camera_config)
    if threaded:
        inner = ThreadedSource(
            inner,
            buffer=buffer,
            buffer_size=buffer_size,
            warmup_frames=warmup_frames,
            read_timeout=read_timeout,
        )
    return AsyncSource(inner)
