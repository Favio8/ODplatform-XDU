#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""FrameSource abstract base class."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .types import Frame, SourceType

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Shared protocol for cameras, videos, images, and wrappers."""

    def __init__(self, source_path: str):
        self.source_path = source_path
        self._frame_index = 0
        self._start_time = time.time()

    @abstractmethod
    def open(self) -> bool:
        """Open the source."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read one frame, or None when exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Close the source."""

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Return the source type."""

    def seek(
        self,
        frame: Optional[int] = None,
        time_sec: Optional[float] = None,
    ) -> bool:
        logger.warning("%s 不支持 seek 操作", self.__class__.__name__)
        return False

    @property
    def seekable(self) -> bool:
        return False

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
