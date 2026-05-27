#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Drawing primitives and layout helpers."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .data_types import DrawStyle, LabelLayout, LabelPosition


class RoundedRect:
    """Rounded rectangle drawing helpers."""

    @staticmethod
    def filled(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int,
        corners: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> None:
        x1, y1 = pt1
        x2, y2 = pt2
        tl, tr, bl, br = corners

        cv2.rectangle(
            img,
            (x1 + (radius if tl else 0), y1),
            (x2 - (radius if tr else 0), y1 + radius),
            color,
            -1,
        )
        cv2.rectangle(
            img,
            (x1 + (radius if bl else 0), y2 - radius),
            (x2 - (radius if br else 0), y2),
            color,
            -1,
        )
        cv2.rectangle(
            img,
            (x1, y1 + (radius if tl or tr else 0)),
            (x2, y2 - (radius if bl or br else 0)),
            color,
            -1,
        )

        if tl:
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        if tr:
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        if bl:
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        if br:
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)

    @staticmethod
    def bordered(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        radius: int,
        corners: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> None:
        x1, y1 = pt1
        x2, y2 = pt2
        tl, tr, bl, br = corners

        cv2.line(img, (x1 + (radius if tl else 0), y1), (x2 - (radius if tr else 0), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + (radius if bl else 0), y2), (x2 - (radius if br else 0), y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + (radius if tl else 0)), (x1, y2 - (radius if bl else 0)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + (radius if tr else 0)), (x2, y2 - (radius if br else 0)), color, thickness, cv2.LINE_AA)

        if tl:
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        if tr:
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        if bl:
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        if br:
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


class LayoutCalculator:
    """Compute label positions and corner behavior."""

    @staticmethod
    def compute(
        det_box: Tuple[int, int, int, int],
        text_size: Tuple[int, int],
        img_size: Tuple[int, int],
        style: DrawStyle,
    ) -> LabelLayout:
        x1, y1, x2, y2 = det_box
        text_w, text_h = text_size
        img_h, img_w = img_size

        label_w = max(text_w + 2 * style.padding_x, 2 * style.radius)
        label_h = text_h + 2 * style.padding_y
        det_w = x2 - x1
        label_x1 = x1 - style.line_width // 2

        if y1 - label_h >= 0:
            position = LabelPosition.ABOVE
            label_y1 = y1 - label_h
            label_y2 = y1
        elif (y2 - y1) >= label_h + style.line_width * 2:
            position = LabelPosition.INSIDE_TOP
            label_y1 = y1 - style.line_width // 2
            label_y2 = y1 + label_h
        else:
            position = LabelPosition.BELOW
            label_y1 = y2 + style.line_width
            label_y2 = min(y2 + label_h + style.line_width, img_h)
            if label_y2 > img_h:
                label_y1 = img_h - label_h
                label_y2 = img_h

        label_x2 = label_x1 + label_w
        align_right = False
        if label_x2 > img_w:
            align_right = True
            label_x1 = x2 + style.line_width // 2 - label_w
            label_x1 = max(0, label_x1)
            label_x2 = label_x1 + label_w

        text_x = label_x1 + (label_w - text_w) // 2
        text_y = label_y1 + (label_h - text_h) // 2

        return LabelLayout(
            box=(label_x1, label_y1, label_x2, label_y2),
            text_pos=(text_x, text_y),
            position=position,
            align_right=align_right,
            label_wider=label_w > det_w,
        )

    @staticmethod
    def get_corners(
        layout: LabelLayout,
        for_detection: bool = False,
    ) -> Tuple[bool, bool, bool, bool]:
        pos = layout.position
        right = layout.align_right
        wider = layout.label_wider

        if for_detection:
            if pos == LabelPosition.ABOVE:
                return (not wider, False, True, True) if right else (False, not wider, True, True)
            if pos == LabelPosition.BELOW:
                return (True, True, not wider, False) if right else (True, True, False, not wider)
            return (False, False, True, True)

        if pos == LabelPosition.ABOVE:
            return (True, True, wider, False) if right else (True, True, False, wider)
        if pos == LabelPosition.BELOW:
            return (not wider, False, True, True) if right else (False, not wider, True, True)
        return (True, True, wider, False) if right else (True, True, False, True)
