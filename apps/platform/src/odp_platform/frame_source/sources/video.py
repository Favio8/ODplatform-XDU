#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Video file source implementation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from ..core.base import FrameSource
from ..core.types import Frame, FrameInfo, SourceType

logger = logging.getLogger(__name__)


class VideoSource(FrameSource):
    """Video file source with seek support."""

    def __init__(self, video_path: str):
        super().__init__(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._total_frames = 0
        self._filename = Path(video_path).name

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source_path)
        if not self._cap.isOpened():
            logger.error("无法打开视频: %s", self.source_path)
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not raw_fps or raw_fps <= 0:
            logger.warning(
                "视频 '%s' FPS 元数据缺失或为 0,已回退到默认值 30fps。seek(time_sec=...) 结果可能不准确。",
                self._filename,
            )
            self._fps = 30.0
        else:
            self._fps = raw_fps

        logger.info("视频已打开: %s", self._filename)
        logger.info("  分辨率: %sx%s @ %.1ffps", self._width, self._height, self._fps)
        logger.info("  总帧数: %s", self._total_frames)
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ret, image = self._cap.read()
        if not ret:
            return None

        info = FrameInfo(
            width=self._width,
            height=self._height,
            source_type=SourceType.VIDEO,
            source_path=self.source_path,
            frame_index=self._frame_index,
            total_frames=self._total_frames,
            timestamp=self._frame_index / self._fps if self._fps > 0 else 0.0,
            fps=self._fps,
            filename=self._filename,
        )
        self._frame_index += 1
        return Frame(image=image, info=info)

    def seek(
        self,
        frame: Optional[int] = None,
        time_sec: Optional[float] = None,
    ) -> bool:
        if self._cap is None:
            logger.error("视频未打开,无法 seek")
            return False
        if (frame is None) == (time_sec is None):
            logger.error("frame 和 time_sec 必须且只能指定一个")
            return False

        target = int(time_sec * self._fps) if time_sec is not None else int(frame)
        target = max(0, target)
        if self._total_frames > 0:
            target = min(target, self._total_frames - 1)

        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        if ok:
            self._frame_index = target
            logger.debug("视频跳转到帧 %s", target)
        else:
            logger.warning("视频跳转失败:目标帧 %s", target)
        return ok

    @property
    def seekable(self) -> bool:
        return True

    @property
    def duration(self) -> float:
        if self._fps > 0 and self._total_frames > 0:
            return self._total_frames / self._fps
        return 0.0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("视频已关闭")

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO
