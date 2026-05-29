#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Pipeline-side config loader for frame source and visualization settings."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)


def _to_bgr_tuple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"颜色必须是 3 元素的 [B, G, R], 收到: {value!r}")


@dataclass
class PipelineConfig:
    """Parsed frame-source and visualization config."""

    camera_raw: dict[str, Any] = field(default_factory=dict)
    viz_enabled: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: str | None = None
    style_overrides: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def build_camera_config(self):
        if not self.camera_raw:
            return None
        from odp_platform.frame_source import CameraConfig

        return CameraConfig(**self.camera_raw)

    def to_audit(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "camera": self.camera_raw or None,
            "visualization": {
                "enabled": self.viz_enabled,
                "use_label_mapping": self.use_label_mapping,
                "label_mapping": self.label_mapping or None,
                "color_mapping": {key: list(value) for key, value in self.color_mapping.items()} or None,
                "default_color": list(self.default_color),
                "font_path": self.font_path,
                "style": self.style_overrides or None,
            },
        }


def load_pipeline_config(yaml_path: str | Path | None = None) -> PipelineConfig:
    from odp_platform.common.paths import RUNTIME_CONFIGS_DIR

    if yaml_path is None:
        path = RUNTIME_CONFIGS_DIR / "infer_pipeline.yaml"
    else:
        candidate = Path(yaml_path)
        path = candidate if candidate.is_absolute() or str(candidate.parent) != "." else RUNTIME_CONFIGS_DIR / candidate

    if not path.exists():
        logger.warning(
            "未找到帧源+美化配置 %s, 使用默认(美化开启、无中文映射、摄像头默认参数). 如需自定义请创建该文件或用 odp-infer --pipeline-yaml 指定.",
            path,
        )
        return PipelineConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    frame_source = raw.get("frame_source") or {}
    visualization = raw.get("visualization") or {}

    color_mapping = {
        str(key): _to_bgr_tuple(value)
        for key, value in (visualization.get("color_mapping") or {}).items()
    }
    default_color = _to_bgr_tuple(visualization["default_color"]) if visualization.get("default_color") else (0, 255, 0)

    return PipelineConfig(
        camera_raw=frame_source.get("camera") or {},
        viz_enabled=bool(visualization.get("enabled", True)),
        use_label_mapping=bool(visualization.get("use_label_mapping", True)),
        label_mapping={str(key): str(value) for key, value in (visualization.get("label_mapping") or {}).items()},
        color_mapping=color_mapping,
        default_color=default_color,
        font_path=visualization.get("font_path"),
        style_overrides=visualization.get("style") or {},
        source_path=path,
    )
