"""Teacher-style runtime-config merger with linked provenance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from odp_platform.config.base import ConfigTrace, RuntimeConfigBase
from odp_platform.config.merger import build_default_source


T = TypeVar("T", bound=BaseModel)


class ConfigSource(str, Enum):
    """Built-in config sources. Custom sources can still use plain strings."""

    DEFAULT = "DEFAULT"
    YAML = "YAML"
    CLI = "CLI"


@dataclass(frozen=True)
class ConfigMetadata:
    """Linked-list style provenance for one final config field."""

    key: str
    value: Any
    source: ConfigSource | str
    timestamp: datetime
    overridden_from: Optional["ConfigMetadata"] = None

    @property
    def source_label(self) -> str:
        if isinstance(self.source, ConfigSource):
            return self.source.value
        return str(self.source)

    def chain(self) -> list["ConfigMetadata"]:
        result = [self]
        current = self.overridden_from
        while current is not None:
            result.append(current)
            current = current.overridden_from
        return result

    def chain_str(self) -> str:
        return " ← ".join(f"{item.value}({item.source_label})" for item in self.chain())


class ConfigMerger:
    """Merge multiple config sources and retain teacher-style source provenance."""

    def __init__(self, track_sources: bool = True):
        self.track_sources = track_sources
        self._metadata: dict[str, ConfigMetadata] = {}
        self._overridden_keys: list[str] = []
        self._config_class: type[RuntimeConfigBase] | None = None
        self._merged: dict[str, Any] = {}
        self._completed_at: str = datetime.now().isoformat(timespec="seconds")

    @classmethod
    def from_trace(
        cls,
        trace: ConfigTrace,
        *,
        config_class: type[RuntimeConfigBase] | None = None,
        track_sources: bool = True,
    ) -> "ConfigMerger":
        merger = cls(track_sources=track_sources)
        merger._config_class = config_class
        merger._completed_at = trace.created_at or datetime.now().isoformat(timespec="seconds")

        for field_name, field_trace in trace.by_field.items():
            latest: ConfigMetadata | None = None
            for override in field_trace.history:
                source = cls._normalize_source_from_label(override.source_label, override.source_name)
                latest = ConfigMetadata(
                    key=field_name,
                    value=override.value,
                    source=source,
                    timestamp=datetime.fromisoformat(merger._completed_at),
                    overridden_from=latest,
                )
            if latest is not None:
                merger._metadata[field_name] = latest
            merger._merged[field_name] = field_trace.final_value
            if len(field_trace.history) > 1:
                merger._overridden_keys.append(field_name)
        return merger

    def merge(
        self,
        config_class: type[T],
        *,
        sources: list[tuple[ConfigSource | str, dict[str, Any]]] | None = None,
    ) -> T:
        merged = self._do_merge(config_class, sources)
        return config_class.model_validate(merged)

    def preview(
        self,
        config_class: type[RuntimeConfigBase],
        *,
        sources: list[tuple[ConfigSource | str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        return self._do_merge(config_class, sources)

    def get_metadata(self, key: str) -> ConfigMetadata | None:
        return self._metadata.get(key)

    def get_source_report(self) -> str:
        if not self.track_sources:
            return "配置溯源未启用"

        by_source: dict[str, list[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)

        builtin_order = [ConfigSource.CLI.value, ConfigSource.YAML.value, ConfigSource.DEFAULT.value]
        ordered_labels = [label for label in builtin_order if label in by_source]
        ordered_labels.extend(sorted(label for label in by_source if label not in builtin_order))

        lines = ["=" * 70, "配置来源报告".center(70), "=" * 70]
        for label in ordered_labels:
            keys = sorted(by_source[label])
            lines.append(f"\n{label} ({len(keys)} 项)")
            lines.append("-" * 70)
            for key in keys:
                meta = self._metadata[key]
                lines.append(f"  {key:<20}: {self._display_value(key, meta.value)}")
        return "\n".join(lines)

    def get_conflict_report(self) -> str:
        if not self.track_sources:
            return "配置溯源未启用"

        overridden = sorted(set(self._overridden_keys))
        lines = ["=" * 70, "配置覆盖报告".center(70), "=" * 70, f"\n共 {len(overridden)} 项配置被覆盖\n"]
        if not overridden:
            lines.append("  (无)")
            return "\n".join(lines)

        for key in overridden:
            meta = self._metadata.get(key)
            if meta is None or meta.overridden_from is None:
                continue
            newest = meta
            previous = meta.overridden_from
            lines.append(
                f"  {key}: "
                f"{self._display_value(key, previous.value)} ({previous.source_label}) "
                f"→ {self._display_value(key, newest.value)} ({newest.source_label})"
            )
        return "\n".join(lines)

    def to_audit_log(self) -> dict[str, Any]:
        if not self.track_sources:
            return {
                "merger_completed_at": self._completed_at,
                "track_sources": False,
            }

        by_source: dict[str, list[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)

        by_source = {label: sorted(keys) for label, keys in sorted(by_source.items())}
        overridden = sorted(set(self._overridden_keys))
        return {
            "merger_completed_at": self._completed_at,
            "track_sources": True,
            "fields_count_total": len(self._metadata),
            "fields_by_source": by_source,
            "overridden_count": len(overridden),
            "overridden_fields": overridden,
        }

    def _do_merge(
        self,
        config_class: type[RuntimeConfigBase],
        sources: list[tuple[ConfigSource | str, dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        self._config_class = config_class
        self._metadata = {}
        self._overridden_keys = []
        self._completed_at = datetime.now().isoformat(timespec="seconds")

        merged = build_default_source(config_class)
        if self.track_sources:
            for key, value in merged.items():
                if value is None:
                    continue
                self._metadata[key] = ConfigMetadata(
                    key=key,
                    value=value,
                    source=ConfigSource.DEFAULT,
                    timestamp=datetime.now(),
                )

        for source, payload in sources or []:
            for key, value in payload.items():
                if value is None:
                    continue
                merged[key] = value
                if not self.track_sources:
                    continue
                previous = self._metadata.get(key)
                if previous is not None:
                    self._overridden_keys.append(key)
                self._metadata[key] = ConfigMetadata(
                    key=key,
                    value=value,
                    source=self._normalize_source(source),
                    timestamp=datetime.now(),
                    overridden_from=previous,
                )

        self._merged = dict(merged)
        return dict(merged)

    def _display_value(self, key: str, value: Any) -> Any:
        if self._config_class is None:
            return value
        if key in self._config_class.sensitive_field_names() and value is not None:
            return "***"
        return value

    @staticmethod
    def _normalize_source(source: ConfigSource | str) -> ConfigSource | str:
        if isinstance(source, ConfigSource):
            return source

        source_text = str(source)
        source_upper = source_text.split(":", 1)[0].strip().upper()
        if source_upper == ConfigSource.DEFAULT.value:
            return ConfigSource.DEFAULT
        if source_upper == ConfigSource.YAML.value:
            return ConfigSource.YAML
        if source_upper == ConfigSource.CLI.value:
            return ConfigSource.CLI
        return source_upper or source_text

    @classmethod
    def _normalize_source_from_label(cls, source_label: str, source_name: str) -> ConfigSource | str:
        label_upper = source_label.upper()
        if label_upper == ConfigSource.DEFAULT.value:
            return ConfigSource.DEFAULT
        if label_upper == ConfigSource.YAML.value:
            return ConfigSource.YAML
        if label_upper == ConfigSource.CLI.value:
            return ConfigSource.CLI
        return cls._normalize_source(source_name)


__all__ = ["ConfigMerger", "ConfigMetadata", "ConfigSource"]
