"""Compatibility public API for the D5 runtime configuration subsystem."""

from typing import Any

from odp_platform.config import (
    ConfigBuildError,
    ConfigLoadError,
    ConfigSourcePayload,
    ConfigTrace,
    ConfigWarning,
    FieldSpec,
    FieldTrace,
    build_config as _build_config,
    build_infer_config as _build_infer_config,
    build_train_config as _build_train_config,
    build_val_config as _build_val_config,
    generate_template,
    preview_config as _preview_config,
    preview_infer_config as _preview_infer_config,
    preview_train_config as _preview_train_config,
    preview_val_config as _preview_val_config,
)
from odp_platform.config.base import InferConfig as YOLOInferConfig
from odp_platform.config.base import RuntimeConfigBase as BaseConfig
from odp_platform.config.base import TrainConfig as YOLOTrainConfig
from odp_platform.config.base import ValConfig as YOLOValConfig
from odp_platform.config.base import get_config_class
from odp_platform.runtime_config.generator import ConfigGenerator
from odp_platform.runtime_config.loaders import CLILoader, YAMLLoader, load_all_sources
from odp_platform.runtime_config.merger import ConfigMerger, ConfigMetadata, ConfigSource


def _wrap_trace(trace: ConfigTrace, *, config_class: type[BaseConfig] | None = None) -> ConfigMerger:
    return ConfigMerger.from_trace(trace, config_class=config_class)


def build_config(*args: Any, **kwargs: Any):
    config, trace, warnings = _build_config(*args, **kwargs)
    return config, _wrap_trace(trace, config_class=type(config)), warnings


def build_train_config(*args: Any, **kwargs: Any):
    config, trace = _build_train_config(*args, **kwargs)
    return config, _wrap_trace(trace, config_class=type(config))


def build_val_config(*args: Any, **kwargs: Any):
    config, trace = _build_val_config(*args, **kwargs)
    return config, _wrap_trace(trace, config_class=type(config))


def build_infer_config(*args: Any, **kwargs: Any):
    config, trace = _build_infer_config(*args, **kwargs)
    return config, _wrap_trace(trace, config_class=type(config))


def preview_config(*args: Any, **kwargs: Any):
    merged, trace = _preview_config(*args, **kwargs)
    task_kind = kwargs.get("task_kind")
    config_class = get_config_class(task_kind) if isinstance(task_kind, str) else None
    return merged, _wrap_trace(trace, config_class=config_class)


def preview_train_config(*args: Any, **kwargs: Any):
    merged, trace = _preview_train_config(*args, **kwargs)
    return merged, _wrap_trace(trace, config_class=YOLOTrainConfig)


def preview_val_config(*args: Any, **kwargs: Any):
    merged, trace = _preview_val_config(*args, **kwargs)
    return merged, _wrap_trace(trace, config_class=YOLOValConfig)


def preview_infer_config(*args: Any, **kwargs: Any):
    merged, trace = _preview_infer_config(*args, **kwargs)
    return merged, _wrap_trace(trace, config_class=YOLOInferConfig)


__all__ = [
    "BaseConfig",
    "CLILoader",
    "ConfigBuildError",
    "ConfigLoadError",
    "ConfigMerger",
    "ConfigMetadata",
    "ConfigGenerator",
    "ConfigSource",
    "ConfigSourcePayload",
    "ConfigTrace",
    "ConfigWarning",
    "FieldSpec",
    "FieldTrace",
    "YAMLLoader",
    "YOLOInferConfig",
    "YOLOTrainConfig",
    "YOLOValConfig",
    "build_config",
    "build_infer_config",
    "build_train_config",
    "build_val_config",
    "generate_template",
    "load_all_sources",
    "preview_config",
    "preview_infer_config",
    "preview_train_config",
    "preview_val_config",
]
