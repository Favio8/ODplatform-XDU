#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Concrete frame source implementations."""
from __future__ import annotations

from .camera import CameraSource
from .image import ImageFolderSource, ImageSource
from .video import VideoSource

__all__ = [
    "CameraSource",
    "VideoSource",
    "ImageSource",
    "ImageFolderSource",
]
