#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Single-image and image-folder sources."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..core.base import FrameSource
from ..core.types import IMAGE_EXTENSIONS, Frame, FrameInfo, SourceType

logger = logging.getLogger(__name__)


class ImageSource(FrameSource):
    """Single-image source."""

    def __init__(self, image_path: str):
        super().__init__(image_path)
        self._image: Optional[np.ndarray] = None
        self._read_count = 0
        self._filename = Path(image_path).name

    def open(self) -> bool:
        self._image = cv2.imread(self.source_path)
        if self._image is None:
            logger.error("无法读取图片: %s", self.source_path)
            return False
        height, width = self._image.shape[:2]
        logger.info("图片已加载: %s (%sx%s)", self._filename, width, height)
        return True

    def read(self) -> Optional[Frame]:
        if self._image is None or self._read_count > 0:
            return None

        height, width = self._image.shape[:2]
        info = FrameInfo(
            width=width,
            height=height,
            source_type=SourceType.IMAGE,
            source_path=self.source_path,
            frame_index=0,
            total_frames=1,
            filename=self._filename,
        )
        self._read_count += 1
        return Frame(image=self._image.copy(), info=info)

    def close(self) -> None:
        self._image = None

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE


class ImageFolderSource(FrameSource):
    """Image-folder source."""

    def __init__(self, folder_path: str):
        super().__init__(folder_path)
        self._image_files: List[Path] = []
        self._current_index = 0

    def open(self) -> bool:
        folder = Path(self.source_path)
        if not folder.is_dir():
            logger.error("不是有效文件夹: %s", self.source_path)
            return False

        self._image_files = sorted(
            [
                file_path
                for file_path in folder.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        if not self._image_files:
            logger.error("文件夹中没有支持的图片: %s", self.source_path)
            return False

        logger.info("文件夹已加载: %s (%s 张)", folder.name, len(self._image_files))
        return True

    def read(self) -> Optional[Frame]:
        while self._current_index < len(self._image_files):
            image_path = self._image_files[self._current_index]
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("无法读取,已跳过: %s", image_path.name)
                self._current_index += 1
                continue

            height, width = image.shape[:2]
            info = FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.IMAGE_FOLDER,
                source_path=self.source_path,
                frame_index=self._current_index,
                total_frames=len(self._image_files),
                filename=image_path.name,
            )
            self._current_index += 1
            return Frame(image=image, info=info)

        return None

    def seek(
        self,
        frame: Optional[int] = None,
        time_sec: Optional[float] = None,
    ) -> bool:
        if time_sec is not None:
            logger.warning("图片文件夹不支持按时间跳转,请使用 frame 参数")
            return False
        if frame is None:
            logger.error("必须指定 frame 参数")
            return False

        total = len(self._image_files)
        target = max(0, min(frame, total - 1)) if total > 0 else 0
        self._current_index = target
        logger.debug("图片文件夹跳转到索引 %s", target)
        return True

    @property
    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self._image_files = []
        logger.info("文件夹已关闭")

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE_FOLDER
