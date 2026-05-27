#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Pillow-based text renderer."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data_types import DrawStyle
from .text_cache import TextSizeCache, _resolve_font_path

logger = logging.getLogger(__name__)


class PillowTextRenderer:
    """Render text onto BGR numpy images."""

    def __init__(self, size_cache: Optional[TextSizeCache] = None):
        self._size_cache = size_cache
        self._fallback_warned = False

    def set_cache(self, cache: TextSizeCache) -> None:
        self._size_cache = cache

    def render_batch(
        self,
        img: np.ndarray,
        texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
        style: DrawStyle,
    ) -> np.ndarray:
        if not texts:
            return img

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(style)

        for text, pos, color in texts:
            draw.text(pos, text, font=font, fill=color)

        return np.array(pil_img)

    def get_text_size(self, text: str, style: DrawStyle) -> Tuple[int, int]:
        if self._size_cache is not None:
            parts = text.rsplit(" ", 1)
            if len(parts) == 2:
                label = parts[0]
                return self._size_cache.get_size(label, style.font_size)

        font = self._get_font(style)
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    def _get_font(self, style: DrawStyle) -> ImageFont.FreeTypeFont:
        if self._size_cache is not None:
            return self._size_cache.get_font(style.font_size)

        font_path = _resolve_font_path(style.font_path)
        try:
            return ImageFont.truetype(font_path, style.font_size)
        except OSError as exc:
            if not self._fallback_warned:
                logger.warning(
                    "字体 '%s' 加载失败(%s),已回退到 PIL 默认 bitmap 字体。",
                    font_path,
                    exc,
                )
                self._fallback_warned = True
            return ImageFont.load_default()
