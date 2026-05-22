"""Load runtime configuration values from YAML, CLI, or custom sources."""

from __future__ import annotations

import argparse
import locale
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import CONFIGS_DIR
from odp_platform.common.constants import (
    RUNTIME_TASK_INFER,
    RUNTIME_TASK_TRAIN,
    RUNTIME_TASK_VAL,
)
from odp_platform.config.base import RuntimeConfigBase, get_config_class


logger = get_logger(__name__)
_TASK_FILE_BASENAMES = {
    RUNTIME_TASK_TRAIN: "train",
    RUNTIME_TASK_VAL: "val",
    RUNTIME_TASK_INFER: "infer",
}


class ConfigLoadError(ValueError):
    """Raised when user-provided config input cannot be loaded safely."""


@dataclass(frozen=True)
class ConfigSourcePayload:
    """One normalized config source ready for merge."""

    source_name: str
    values: dict[str, Any]


def _resolve_yaml_path(config_path: str | Path, task_kind: str | None = None) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate

    if len(candidate.parts) == 1:
        config_dir = CONFIGS_DIR
        if candidate.suffix:
            return config_dir / candidate.name
        if task_kind is not None:
            basename = _TASK_FILE_BASENAMES[task_kind]
            return config_dir / f"{candidate.name}.{basename}.yaml"
        return config_dir / f"{candidate.name}.yaml"

    return (Path.cwd() / candidate).resolve()


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        fallback_encoding = locale.getpreferredencoding(False) or "utf-8"
        logger.warning("UTF-8 decode failed for %s; retrying with %s", path, fallback_encoding)
        return path.read_text(encoding=fallback_encoding)


def _flatten_mapping(payload: Mapping[str, Any], config_cls: type[RuntimeConfigBase]) -> dict[str, Any]:
    aliases = config_cls.field_name_aliases()
    values: dict[str, Any] = {}
    unknown_fields: list[str] = []

    for key, raw_value in payload.items():
        if raw_value is None:
            continue

        normalized_key = aliases.get(str(key))
        if normalized_key is not None:
            values[normalized_key] = raw_value
            continue

        if isinstance(raw_value, Mapping):
            for nested_key, nested_value in raw_value.items():
                if nested_value is None:
                    continue
                normalized_nested_key = aliases.get(str(nested_key))
                if normalized_nested_key is not None:
                    values[normalized_nested_key] = nested_value
                else:
                    unknown_fields.append(f"{key}.{nested_key}")
            continue

        unknown_fields.append(str(key))

    if unknown_fields:
        raise ConfigLoadError(
            "Unknown config field(s): " + ", ".join(sorted(unknown_fields))
        )

    return values


def load_yaml_config(
    *,
    task_kind: str,
    config_path: str | Path,
) -> ConfigSourcePayload:
    """Load YAML config values for one runtime task."""

    config_cls = get_config_class(task_kind)
    resolved_path = _resolve_yaml_path(config_path, task_kind=task_kind)
    if not resolved_path.exists():
        template_hint = f"odp-generate-config --task {task_kind} --output \"{resolved_path}\""
        raise ConfigLoadError(
            "Config file does not exist. "
            f"Expected path: {resolved_path}. "
            "Cannot continue because the requested YAML source is missing. "
            f"Generate a template with: {template_hint}"
        )

    text = _read_text_with_fallback(resolved_path)
    payload = yaml.safe_load(text)
    if payload is None:
        return ConfigSourcePayload(source_name=f"yaml:{resolved_path}", values={})
    if not isinstance(payload, Mapping):
        raise ConfigLoadError(
            f"YAML top-level must be a mapping, got {type(payload).__name__} from {resolved_path}"
        )

    values = _flatten_mapping(payload, config_cls)
    return ConfigSourcePayload(source_name=f"yaml:{resolved_path}", values=values)


def load_cli_config(
    *,
    task_kind: str,
    namespace: argparse.Namespace | Mapping[str, Any] | None,
    ignored_keys: set[str] | None = None,
) -> ConfigSourcePayload:
    """Load config values from a parsed CLI namespace or mapping."""

    if namespace is None:
        return ConfigSourcePayload(source_name="cli", values={})

    config_cls = get_config_class(task_kind)
    aliases = config_cls.field_name_aliases()
    ignored = set(ignored_keys or set())
    ignored.update({"help"})
    raw_values = vars(namespace) if isinstance(namespace, argparse.Namespace) else dict(namespace)

    values: dict[str, Any] = {}
    unknown_fields: list[str] = []
    for key, value in raw_values.items():
        if key in ignored or key.startswith("_") or value is None:
            continue
        normalized_key = aliases.get(key)
        if normalized_key is not None:
            values[normalized_key] = value
        else:
            unknown_fields.append(key)

    if unknown_fields:
        raise ConfigLoadError("Unknown CLI config field(s): " + ", ".join(sorted(unknown_fields)))
    return ConfigSourcePayload(source_name="cli", values=values)


def load_mapping_source(
    *,
    task_kind: str,
    source_name: str,
    values: Mapping[str, Any] | None,
) -> ConfigSourcePayload:
    """Load one arbitrary mapping source with the same validation rules as YAML."""

    config_cls = get_config_class(task_kind)
    normalized = _flatten_mapping(values or {}, config_cls)
    return ConfigSourcePayload(source_name=source_name, values=normalized)


__all__ = [
    "ConfigLoadError",
    "ConfigSourcePayload",
    "load_cli_config",
    "load_mapping_source",
    "load_yaml_config",
]
