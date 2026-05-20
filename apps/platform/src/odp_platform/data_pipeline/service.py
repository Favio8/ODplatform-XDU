"""Dispatch helpers that call registered dataset converters."""

from __future__ import annotations

from typing import Any

from odp_platform.data_pipeline.registry import ConvertOptions, get_converter


class DataPipelineService:
    """Thin service layer over the format registry."""

    def get_converter(self, source_format: str) -> Any:
        return get_converter(source_format)

    def convert(self, options: ConvertOptions, /, **kwargs: Any) -> Any:
        converter = self.get_converter(options.source_format)
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
