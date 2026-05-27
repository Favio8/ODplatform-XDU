#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Camera source implementation."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2

from ..core.base import FrameSource
from ..core.config import CameraConfig
from ..core.types import Frame, FrameInfo, SourceType

logger = logging.getLogger(__name__)


class CameraSource(FrameSource):
    """Camera input source."""

    def __init__(self, config: CameraConfig):
        super().__init__(str(config.camera_id))
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = 0
        self._height = 0
        self._fps = 0.0

    def open(self) -> bool:
        if self.config.backend == "msmf":
            os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

        self._cap = cv2.VideoCapture(self.config.camera_id, self._get_backend())
        if not self._cap.isOpened():
            logger.error("无法打开摄像头 %s", self.config.camera_id)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.codec))
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        ret, _ = self._cap.read()
        if not ret:
            logger.warning("格式协商触发帧读取失败,实际参数可能不准确")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

        if self._width != self.config.width or self._height != self.config.height:
            logger.warning(
                "分辨率未完全生效:期望 %sx%s,实际 %sx%s",
                self.config.width,
                self.config.height,
                self._width,
                self._height,
            )
        if self._fps < self.config.fps * 0.9:
            logger.warning(
                "帧率未完全生效:期望 %sfps,实际标称 %.1ffps",
                self.config.fps,
                self._fps,
            )

        logger.info(
            "摄像头已打开 (backend=%s, codec=%s)",
            self.config.backend,
            self.config.codec,
        )
        logger.info(
            "  目标: %sx%s @ %sfps",
            self.config.width,
            self.config.height,
            self.config.fps,
        )
        logger.info("  实际: %sx%s @ %.1ffps", self._width, self._height, self._fps)
        return True

    def _get_backend(self) -> int:
        backends = {
            "auto": cv2.CAP_ANY,
            "msmf": cv2.CAP_MSMF,
            "dshow": cv2.CAP_DSHOW,
            "v4l2": cv2.CAP_V4L2,
        }
        return backends[self.config.backend]

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        ret, image = self._cap.read()
        if not ret:
            return None

        info = FrameInfo(
            width=self._width,
            height=self._height,
            source_type=SourceType.CAMERA,
            source_path=self.source_path,
            frame_index=self._frame_index,
            timestamp=time.time() - self._start_time,
            fps=self._fps,
            filename=f"camera:{self.config.camera_id}",
        )
        self._frame_index += 1
        return Frame(image=image, info=info)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("摄像头已关闭")

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA
