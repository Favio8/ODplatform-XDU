"""Compatibility exports for runtime-config base models."""

from odp_platform.config.base import (
    ConfigTrace,
    FieldSpec,
    FieldTrace,
    InferConfig as YOLOInferConfig,
    RuntimeConfigBase as BaseConfig,
    SourceOverride,
    TrainConfig as YOLOTrainConfig,
    ValConfig as YOLOValConfig,
)


__all__ = [
    "BaseConfig",
    "ConfigTrace",
    "FieldSpec",
    "FieldTrace",
    "SourceOverride",
    "YOLOInferConfig",
    "YOLOTrainConfig",
    "YOLOValConfig",
]
