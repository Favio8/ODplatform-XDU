"""Validation helpers for runtime configuration objects."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from odp_platform.common.constants import SUPPORTED_TASKS, TASK_DETECT, TASK_SEGMENT
from odp_platform.runtime_config.base import ConfigTrace, InferConfig, RuntimeConfigBase, TrainConfig, ValConfig


@dataclass(frozen=True)
class ConfigWarning:
    field_name: str
    message: str


class ConfigBuildError(ValueError):
    """Raised when a runtime config cannot be validated safely."""


def _task_type_error_message(task_type: str) -> str:
    return (
        f"task_type={task_type!r} is not supported. "
        f"Expected one of {SUPPORTED_TASKS}. "
        "If you intended to label this run, use experiment_name instead."
    )


def _format_trace_suffix(field_name: str, trace: ConfigTrace) -> str:
    try:
        field_trace = trace.get(field_name)
    except KeyError:
        return ""
    return f" Source trace: {field_trace.to_human_readable()}"


def validate_config(
    config_cls: type[RuntimeConfigBase],
    values: dict[str, object],
    trace: ConfigTrace,
) -> tuple[RuntimeConfigBase, list[ConfigWarning]]:
    """Validate one merged config dict and return warnings separately."""

    task_type = str(values.get("task_type", TASK_DETECT))
    if task_type not in SUPPORTED_TASKS:
        raise ConfigBuildError(_task_type_error_message(task_type) + _format_trace_suffix("task_type", trace))

    try:
        config = config_cls.model_validate(values)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        location = first_error["loc"][0] if first_error.get("loc") else "unknown"
        message = first_error.get("msg", str(exc))
        raise ConfigBuildError(f"Invalid field {location!r}: {message}.{_format_trace_suffix(str(location), trace)}") from exc

    warnings: list[ConfigWarning] = []
    if isinstance(config, TrainConfig):
        if not config.save and config.save_period > 0:
            raise ConfigBuildError(
                "Configuration is contradictory: save=False but save_period requests periodic checkpoints."
                + _format_trace_suffix("save", trace)
                + _format_trace_suffix("save_period", trace)
            )
        if not config.cache and config.batch == 0:
            warnings.append(
                ConfigWarning(
                    field_name="batch",
                    message="batch=0 while cache=False is allowed, but auto-batch may be slow or unsupported downstream.",
                )
            )
    elif isinstance(config, ValConfig):
        if config.task_type != TASK_SEGMENT:
            if config.mask_ratio != 4 or config.overlap_mask is not True:
                warnings.append(
                    ConfigWarning(
                        field_name="mask_ratio",
                        message="mask_ratio / overlap_mask are only meaningful for segment validation and will be ignored for non-segment tasks.",
                    )
                )
    elif isinstance(config, InferConfig):
        if config.save_conf and not config.save_txt:
            warnings.append(
                ConfigWarning(
                    field_name="save_conf",
                    message="save_conf=True has no effect unless save_txt=True.",
                )
            )
        if config.stream_buffer and not config.stream:
            warnings.append(
                ConfigWarning(
                    field_name="stream_buffer",
                    message="stream_buffer=True has no effect unless stream=True.",
                )
            )
        if config.retina_masks and config.task_type != TASK_SEGMENT:
            warnings.append(
                ConfigWarning(
                    field_name="retina_masks",
                    message="retina_masks only applies to segment inference and will be ignored for non-segment tasks.",
                )
            )

    return config, warnings


__all__ = [
    "ConfigBuildError",
    "ConfigWarning",
    "validate_config",
]
