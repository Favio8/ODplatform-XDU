#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Camera configuration models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CameraBackend = Literal["auto", "msmf", "dshow", "v4l2"]
CameraCodec = Literal["MJPG", "YUYV", "H264", "MP4V"]


class CameraConfig(BaseModel):
    """Validated camera configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    camera_id: int = Field(default=0, ge=0, description="OpenCV 设备 ID")
    width: int = Field(default=1280, gt=0, le=7680, description="请求分辨率宽")
    height: int = Field(default=720, gt=0, le=4320, description="请求分辨率高")
    fps: int = Field(default=30, gt=0, le=1000, description="请求帧率")
    backend: CameraBackend = Field(default="auto", description="摄像头后端")
    codec: CameraCodec = Field(default="MJPG", description="FOURCC 编码")

    def get_resolution(self) -> tuple[int, int, int]:
        return (self.width, self.height, self.fps)
