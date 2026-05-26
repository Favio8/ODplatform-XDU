"""Teacher-style loader wrappers built on the current config subsystem."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from odp_platform.common.constants import (
    RUNTIME_TASK_INFER,
    RUNTIME_TASK_TRAIN,
    RUNTIME_TASK_VAL,
)
from odp_platform.common.paths import RUNTIME_CONFIGS_DIR
from odp_platform.config.loaders import ConfigSourcePayload, load_cli_config, load_yaml_config


_TASK_BY_STEM = {
    "train": RUNTIME_TASK_TRAIN,
    "val": RUNTIME_TASK_VAL,
    "infer": RUNTIME_TASK_INFER,
}


def _guess_task_kind(config_path: str | Path | None, fallback: str = RUNTIME_TASK_TRAIN) -> str:
    if config_path is None:
        return fallback
    stem = Path(config_path).stem.lower()
    return _TASK_BY_STEM.get(stem, fallback)


@dataclass
class YAMLLoader:
    config_dir: Path = RUNTIME_CONFIGS_DIR

    def load(self, config_path: str | Path) -> dict[str, Any]:
        candidate = Path(config_path)
        if not candidate.is_absolute() and len(candidate.parts) == 1:
            candidate = Path(self.config_dir) / candidate
        task_kind = _guess_task_kind(candidate)
        payload = load_yaml_config(task_kind=task_kind, config_path=candidate)
        return payload.values


@dataclass
class CLILoader:
    task_kind: str = RUNTIME_TASK_TRAIN
    exclude: list[str] | None = None

    def load(self, cli_args: argparse.Namespace | Mapping[str, Any] | None) -> dict[str, Any]:
        payload = load_cli_config(
            task_kind=self.task_kind,
            namespace=cli_args,
            ignored_keys=set(self.exclude or []),
        )
        return payload.values


def load_all_sources(
    *,
    task_kind: str = RUNTIME_TASK_TRAIN,
    yaml_path: str | Path | None = None,
    yaml_dir: str | Path | None = None,
    cli_args: argparse.Namespace | Mapping[str, Any] | None = None,
    cli_exclude: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    resolved_yaml_path = None
    if yaml_path is not None:
        candidate = Path(yaml_path)
        if yaml_dir is not None and not candidate.is_absolute() and len(candidate.parts) == 1:
            resolved_yaml_path = Path(yaml_dir) / candidate
        else:
            resolved_yaml_path = candidate

    yaml_values = (
        load_yaml_config(task_kind=task_kind, config_path=resolved_yaml_path).values
        if resolved_yaml_path is not None
        else {}
    )
    cli_values = load_cli_config(
        task_kind=task_kind,
        namespace=cli_args,
        ignored_keys=set(cli_exclude or []),
    ).values
    return {"yaml": yaml_values, "cli": cli_values}


__all__ = ["CLILoader", "ConfigSourcePayload", "YAMLLoader", "load_all_sources"]
