#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Output sink abstractions for inference results."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from odp_platform.frame_source import SourceType


logger = logging.getLogger(__name__)


class OutputSink(ABC):
    """Destination adapter for rendered inference frames."""

    @abstractmethod
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        """Initialize sink resources."""

    @abstractmethod
    def write(self, frame, annotated: np.ndarray) -> None:
        """Persist or forward one rendered frame."""

    @abstractmethod
    def close(self) -> None:
        """Finalize sink resources. Must be idempotent."""


class LocalFileSink(OutputSink):
    """Save video streams as mp4 and image sources as jpg files."""

    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self._is_stream: bool = False
        self._video = None

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self.output_dir = output_dir
        self._is_stream = source_type in (SourceType.VIDEO, SourceType.CAMERA)

    def write(self, frame, annotated: np.ndarray) -> None:
        import cv2

        if self.output_dir is None:
            raise RuntimeError("LocalFileSink 尚未 open()")

        try:
            if self._is_stream:
                if self._video is None:
                    height, width = annotated.shape[:2]
                    fps = float(frame.info.fps) if getattr(frame.info, "fps", None) else 30.0
                    output_path = self.output_dir / "output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                self._video.write(annotated)
            else:
                filename = frame.info.filename or f"frame_{frame.info.frame_index:06d}"
                output_path = self.output_dir / f"{Path(filename).stem}.jpg"
                cv2.imwrite(str(output_path), annotated)
        except Exception as exc:  # pragma: no cover - defensive IO boundary
            logger.warning("LocalFileSink.write 失败, 跳过: %s", exc)

    def close(self) -> None:
        if self._video is None:
            return
        try:
            self._video.release()
        except Exception as exc:  # pragma: no cover - defensive IO boundary
            logger.warning("LocalFileSink.close release 失败 (已吞): %s", exc)
        finally:
            self._video = None


class NullSink(OutputSink):
    """No-op sink for streaming or dry-run scenarios."""

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        return None

    def write(self, frame, annotated: np.ndarray) -> None:
        return None

    def close(self) -> None:
        return None
