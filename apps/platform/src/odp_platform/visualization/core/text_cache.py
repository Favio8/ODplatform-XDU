#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Text size precomputation cache."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_DEFAULT_FONT_NAME = "LXGWWenKai-Bold"
_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


def _iter_system_font_dirs() -> List[Path]:
    dirs: List[Path] = []
    if sys.platform.startswith("win"):
        win = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(win) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
    else:
        dirs += [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local" / "share" / "fonts",
        ]
    return [directory for directory in dirs if directory.is_dir()]


def _match_font_in_dir(directory: Path, name: str, recursive: bool) -> Optional[str]:
    has_ext = Path(name).suffix.lower() in _FONT_EXTENSIONS
    if not recursive:
        if has_ext:
            candidate = directory / name
            return str(candidate) if candidate.is_file() else None
        for ext in _FONT_EXTENSIONS:
            candidate = directory / f"{name}{ext}"
            if candidate.is_file():
                return str(candidate)
        return None

    name_lower = name.lower()
    try:
        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() not in _FONT_EXTENSIONS:
                continue
            if file_path.name.lower() == name_lower or file_path.stem.lower() == name_lower:
                return str(file_path)
    except (OSError, PermissionError):
        pass
    return None


def _resolve_font_path(font: Optional[str]) -> str:
    name = font if font else _DEFAULT_FONT_NAME
    if Path(name).is_file():
        return str(name)

    hit = _match_font_in_dir(_ASSETS_DIR, name, recursive=False)
    if hit:
        return hit

    for directory in _iter_system_font_dirs():
        hit = _match_font_in_dir(directory, name, recursive=True)
        if hit:
            return hit

    return name


class TextSizeCache:
    """Precompute text sizes for label + confidence display."""

    def __init__(
        self,
        labels: List[str],
        label_mapping: Optional[Dict[str, str]] = None,
        font_path: Optional[str] = None,
        font_sizes: Optional[Tuple[int, ...]] = None,
        confidence_template: str = "99.0%",
    ):
        self.font_path = _resolve_font_path(font_path)
        self.label_mapping = label_mapping or {}
        self.font_sizes = font_sizes or tuple(range(12, 30, 1))
        self.confidence_template = confidence_template
        self._fallback_warned = False
        self._size_cache: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._precompute(labels)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(self.font_path, size)
        except OSError as exc:
            if not self._fallback_warned:
                logger.warning(
                    "字体 '%s' 加载失败(%s),已回退到 PIL 默认 bitmap 字体。",
                    self.font_path,
                    exc,
                )
                self._fallback_warned = True
            return ImageFont.load_default()

    def _load_fonts(self) -> None:
        for size in self.font_sizes:
            self._font_cache[size] = self._load_font(size)

    def _precompute(self, labels: List[str]) -> None:
        self._load_fonts()
        measure_img = Image.new("RGB", (1, 1))
        measure_draw = ImageDraw.Draw(measure_img)
        display_labels = set(labels)
        for label in labels:
            if label in self.label_mapping:
                display_labels.add(self.label_mapping[label])

        for display_label in display_labels:
            full_text = f"{display_label} {self.confidence_template}"
            for size in self.font_sizes:
                font = self._font_cache[size]
                bbox = measure_draw.textbbox((0, 0), full_text, font=font)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                self._size_cache[(display_label, size)] = (width, height)

    def get_size(self, display_label: str, font_size: int) -> Tuple[int, int]:
        key = (display_label, font_size)
        if key in self._size_cache:
            return self._size_cache[key]

        nearest_size = min(self.font_sizes, key=lambda size: abs(size - font_size))
        fallback_key = (display_label, nearest_size)
        if fallback_key in self._size_cache:
            width, height = self._size_cache[fallback_key]
            scale = font_size / nearest_size
            return int(width * scale), int(height * scale)
        return (100, 30)

    def get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        if font_size in self._font_cache:
            return self._font_cache[font_size]
        font = self._load_font(font_size)
        self._font_cache[font_size] = font
        return font
