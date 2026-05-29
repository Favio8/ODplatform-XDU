#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Lifecycle hooks for inference execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    """One rendered frame emitted by the pipeline."""

    frame_idx: int
    image: np.ndarray
    annotated: np.ndarray
    n_detections: int
    detections: list[dict[str, Any]] | None = None


@dataclass
class ProgressEvent:
    """Periodic progress update emitted from the main pipeline loop."""

    frame_idx: int
    total_frames: int | None
    elapsed_sec: float
    fps_loop: float
    fps_infer: float
    detections_total: int


@dataclass
class InferHooks:
    """Optional callbacks around one inference run."""

    on_frame: Callable[[FrameEvent], None] | None = None
    on_progress: Callable[[ProgressEvent], None] | None = None
    on_complete: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    progress_interval_frames: int = 30

    def fire_frame(self, event: FrameEvent) -> None:
        if self.on_frame is None:
            return
        try:
            self.on_frame(event)
        except Exception as exc:  # pragma: no cover - defensive hook boundary
            logger.warning("on_frame 回调异常 (已吞): %s", exc)

    def fire_progress(self, event: ProgressEvent) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(event)
        except Exception as exc:  # pragma: no cover - defensive hook boundary
            logger.warning("on_progress 回调异常 (已吞): %s", exc)

    def fire_complete(self, result: Any) -> None:
        if self.on_complete is None:
            return
        try:
            self.on_complete(result)
        except Exception as exc:  # pragma: no cover - defensive hook boundary
            logger.warning("on_complete 回调异常 (已吞): %s", exc)

    def fire_error(self, exc: Exception) -> None:
        if self.on_error is None:
            return
        try:
            self.on_error(exc)
        except Exception as hook_exc:  # pragma: no cover - defensive hook boundary
            logger.warning("on_error 回调异常 (已吞): %s", hook_exc)
