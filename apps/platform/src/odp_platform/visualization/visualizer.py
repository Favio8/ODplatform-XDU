#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Beautified visualizer for YOLO-style detections."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.data_types import Detection, DrawStyle
from .core.draw_utils import LayoutCalculator, RoundedRect
from .core.renderers import PillowTextRenderer
from .core.text_cache import TextSizeCache


class BeautifyVisualizer:
    """Beautiful visualization for detection results."""

    def __init__(
        self,
        labels: List[str],
        label_mapping: Optional[Dict[str, str]] = None,
        color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (0, 255, 0),
        font_path: Optional[str] = None,
        font_sizes: Optional[Tuple[int, ...]] = None,
    ):
        self.label_mapping = label_mapping or {}
        self.color_mapping = color_mapping or {}
        self.default_color = default_color
        self._size_cache = TextSizeCache(
            labels=labels,
            label_mapping=label_mapping,
            font_path=font_path,
            font_sizes=font_sizes,
        )
        self._renderer = PillowTextRenderer(size_cache=self._size_cache)

    def draw(
        self,
        image: np.ndarray,
        detections: List[Detection],
        style: Optional[DrawStyle] = None,
        use_label_mapping: bool = False,
        *,
        draw_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
    ) -> np.ndarray:
        """Draw detections on a BGR image."""
        if not detections:
            return image.copy()

        height, width = image.shape[:2]
        style = style or DrawStyle.from_image_size(height, width)
        result = image.copy()
        texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self.color_mapping.get(det.label, det.color or self.default_color)
            display_label = (
                self.label_mapping.get(det.label, det.label)
                if use_label_mapping
                else det.label
            )
            label_parts: list[str] = []
            if show_labels:
                label_parts.append(display_label)
            if show_conf:
                label_parts.append(f"{det.confidence * 100:.1f}%")
            label_text = " ".join(label_parts)
            text_size = self._size_cache.get_size(display_label, style.font_size)
            layout = LayoutCalculator.compute(det.box, text_size, (height, width), style)
            det_corners = LayoutCalculator.get_corners(layout, for_detection=True)
            label_corners = LayoutCalculator.get_corners(layout, for_detection=False)

            if draw_boxes:
                RoundedRect.bordered(
                    result,
                    (x1, y1),
                    (x2, y2),
                    color,
                    style.line_width,
                    style.radius,
                    det_corners,
                )

            if label_text:
                lx1, ly1, lx2, ly2 = layout.box
                RoundedRect.filled(
                    result,
                    (lx1, ly1),
                    (lx2, ly2),
                    color,
                    style.radius,
                    label_corners,
                )
                texts.append((label_text, layout.text_pos, style.text_color))

        if texts:
            result = self._renderer.render_batch(result, texts, style)
        return result

    @staticmethod
    def from_yolo_results(
        boxes: np.ndarray,
        confidences: np.ndarray,
        labels: List[str],
        color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> List[Detection]:
        """Build Detection values from YOLO arrays."""
        color_mapping = color_mapping or {}
        return [
            Detection(
                box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                confidence=float(conf),
                label=label,
                color=color_mapping.get(label, (0, 255, 0)),
            )
            for box, conf, label in zip(boxes, confidences, labels)
        ]
