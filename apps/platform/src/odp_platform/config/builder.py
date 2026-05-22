"""One-stop config construction and preview helpers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from odp_platform.common.constants import RUNTIME_TASK_INFER, RUNTIME_TASK_TRAIN, RUNTIME_TASK_VAL
from odp_platform.config.base import ConfigTrace, InferConfig, RuntimeConfigBase, TrainConfig, ValConfig, get_config_class
from odp_platform.config.loaders import ConfigSourcePayload, load_cli_config, load_mapping_source, load_yaml_config
from odp_platform.config.merger import merge_sources
from odp_platform.config.validator import ConfigBuildError, ConfigWarning, validate_config


def _normalize_extra_sources(extra_sources: Sequence[ConfigSourcePayload] | None) -> list[ConfigSourcePayload]:
    return list(extra_sources or [])


def preview_config(
    *,
    task_kind: str,
    yaml_path: str | None = None,
    cli_args: argparse.Namespace | Mapping[str, Any] | None = None,
    extra_sources: Sequence[ConfigSourcePayload] | None = None,
    ignored_cli_keys: set[str] | None = None,
) -> tuple[dict[str, object], ConfigTrace]:
    """Load and merge config sources without validation."""

    config_cls = get_config_class(task_kind)
    ordered_sources: list[tuple[str, dict[str, Any]]] = []
    if yaml_path is not None:
        yaml_source = load_yaml_config(task_kind=task_kind, config_path=yaml_path)
        ordered_sources.append((yaml_source.source_name, yaml_source.values))

    for source in _normalize_extra_sources(extra_sources):
        ordered_sources.append((source.source_name, source.values))

    cli_source = load_cli_config(task_kind=task_kind, namespace=cli_args, ignored_keys=ignored_cli_keys)
    if cli_source.values:
        ordered_sources.append((cli_source.source_name, cli_source.values))

    merged, trace = merge_sources(config_cls=config_cls, ordered_sources=ordered_sources)
    return merged, trace


def build_config(
    *,
    task_kind: str,
    yaml_path: str | None = None,
    cli_args: argparse.Namespace | Mapping[str, Any] | None = None,
    extra_sources: Sequence[ConfigSourcePayload] | None = None,
    ignored_cli_keys: set[str] | None = None,
) -> tuple[RuntimeConfigBase, ConfigTrace, list[ConfigWarning]]:
    """Load, merge, and validate one runtime config."""

    config_cls = get_config_class(task_kind)
    merged, trace = preview_config(
        task_kind=task_kind,
        yaml_path=yaml_path,
        cli_args=cli_args,
        extra_sources=extra_sources,
        ignored_cli_keys=ignored_cli_keys,
    )
    config, warnings = validate_config(config_cls, merged, trace)
    return config, trace, warnings


def build_train_config(**kwargs: Any) -> tuple[TrainConfig, ConfigTrace]:
    """Build a validated train config plus full provenance trace."""
    config, trace, _warnings = build_config(task_kind=RUNTIME_TASK_TRAIN, **kwargs)
    return config, trace  # type: ignore[return-value]


def build_val_config(**kwargs: Any) -> tuple[ValConfig, ConfigTrace]:
    """Build a validated validation config plus full provenance trace."""
    config, trace, _warnings = build_config(task_kind=RUNTIME_TASK_VAL, **kwargs)
    return config, trace  # type: ignore[return-value]


def build_infer_config(**kwargs: Any) -> tuple[InferConfig, ConfigTrace]:
    """Build a validated inference config plus full provenance trace."""
    config, trace, _warnings = build_config(task_kind=RUNTIME_TASK_INFER, **kwargs)
    return config, trace  # type: ignore[return-value]


def preview_train_config(**kwargs: Any) -> tuple[dict[str, object], ConfigTrace]:
    return preview_config(task_kind=RUNTIME_TASK_TRAIN, **kwargs)


def preview_val_config(**kwargs: Any) -> tuple[dict[str, object], ConfigTrace]:
    return preview_config(task_kind=RUNTIME_TASK_VAL, **kwargs)


def preview_infer_config(**kwargs: Any) -> tuple[dict[str, object], ConfigTrace]:
    return preview_config(task_kind=RUNTIME_TASK_INFER, **kwargs)


__all__ = [
    "ConfigBuildError",
    "build_config",
    "build_infer_config",
    "build_train_config",
    "build_val_config",
    "preview_config",
    "preview_infer_config",
    "preview_train_config",
    "preview_val_config",
]
