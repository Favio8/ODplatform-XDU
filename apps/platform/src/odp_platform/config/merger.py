"""Merge config sources and preserve full provenance traces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from odp_platform.config.base import ConfigTrace, FieldTrace, RuntimeConfigBase, SourceOverride


def build_default_source(config_cls: type[RuntimeConfigBase]) -> dict[str, Any]:
    """Return default values for one config class."""

    return {
        field_name: spec.default
        for field_name, spec in config_cls.field_specs().items()
    }


def merge_sources(
    *,
    config_cls: type[RuntimeConfigBase],
    ordered_sources: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], ConfigTrace]:
    """Merge sources from low to high priority and keep provenance."""

    merged = build_default_source(config_cls)
    history_by_field: dict[str, list[SourceOverride]] = {
        field_name: [SourceOverride(source_name="defaults", value=default_value)]
        for field_name, default_value in merged.items()
    }

    for source_name, values in ordered_sources:
        for field_name, value in values.items():
            if value is None:
                continue
            merged[field_name] = value
            history_by_field[field_name].append(SourceOverride(source_name=source_name, value=value))

    trace = ConfigTrace(
        by_field={
            field_name: FieldTrace(
                field_name=field_name,
                final_value=merged[field_name],
                final_source=history[-1].source_name,
                history=tuple(history),
                sensitive=config_cls.field_specs()[field_name].sensitive,
            )
            for field_name, history in history_by_field.items()
        }
    )
    return merged, trace


__all__ = [
    "build_default_source",
    "merge_sources",
]
