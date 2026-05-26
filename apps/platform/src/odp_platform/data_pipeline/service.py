"""Dispatch helpers that call registered dataset converters."""

from __future__ import annotations

from typing import Any

from odp_platform.data_pipeline.registry import ConvertOptions, get_converter


class DataPipelineService:
    """Thin service layer over the format registry."""

    def get_converter(self, source_format: str) -> Any:
        return get_converter(source_format)

    def ensure_task_supported(self, converter: Any, task: str, source_format: str) -> None:
        supported_tasks = tuple(getattr(converter, "SUPPORTED_TASKS", ()))
        if task not in supported_tasks:
            supported = ", ".join(supported_tasks) if supported_tasks else "(none)"
            raise ValueError(
                f"Source format {source_format!r} does not support task {task!r}; supported tasks: {supported}"
            )

    def convert(self, options: ConvertOptions, /, **kwargs: Any) -> Any:
        converter = self.get_converter(options.source_format)
        self.ensure_task_supported(converter, options.task, options.source_format)
        if not hasattr(converter, "convert"):
            raise AttributeError(f"Converter for {options.source_format!r} has no 'convert' function")
        return converter.convert(options=options, **kwargs)


def convert_dataset(options: ConvertOptions, /, **kwargs: Any) -> Any:
    """Convenience wrapper around :class:`DataPipelineService`."""

    service = DataPipelineService()
    return service.convert(options, **kwargs)


__all__ = [
    "DataPipelineService",
    "convert_dataset",
]
