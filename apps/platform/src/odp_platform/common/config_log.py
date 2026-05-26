#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : config_log.py
# @Project   : ODPlatform
# @Function  : 按字段维度打印配置参数信息 / 配置覆盖情况
"""Configuration logging helpers for D5/D6."""

from __future__ import annotations

import logging

from odp_platform.common.string_utils import pad_to_width
from odp_platform.runtime_config.base import ConfigTrace, RuntimeConfigBase


def log_effective_config(
    config: RuntimeConfigBase,
    trace: ConfigTrace,
    *,
    logger: logging.Logger | None = None,
    key_width: int = 20,
    section_width: int = 60,
) -> None:
    """Log the final effective config values field by field."""

    log = logger or logging.getLogger(__name__)
    log.info("=" * section_width)
    log.info("开始记录模型参数信息".center(section_width))
    log.info("=" * section_width)

    for field_name in config.to_runtime_dict():
        field_trace = trace.get_metadata(field_name)
        display_name = config.external_field_name(field_name)
        if field_trace is None:
            log.info("%s: %s", pad_to_width(display_name, key_width), getattr(config, field_name, None))
            continue
        log.info(
            "%s: %s  (来源: %s)",
            pad_to_width(display_name, key_width),
            field_trace.final_value,
            field_trace.final_source_label,
        )


def log_override_chains(
    config: RuntimeConfigBase,
    trace: ConfigTrace,
    *,
    logger: logging.Logger | None = None,
    key_width: int = 20,
    section_width: int = 60,
) -> None:
    """Log full override chains for every config field."""

    log = logger or logging.getLogger(__name__)
    log.info("-" * section_width)
    log.info("配置覆盖情况".center(section_width))
    log.info("-" * section_width)

    for field_name in config.to_runtime_dict():
        field_trace = trace.get_metadata(field_name)
        display_name = config.external_field_name(field_name)
        if field_trace is None:
            log.info("%s: %s", pad_to_width(display_name, key_width), getattr(config, field_name, None))
            continue
        chain_str = " <- ".join(
            f"{override.value}({override.source_label})"
            for override in field_trace.history
        )
        log.info("%s: %s", pad_to_width(display_name, key_width), chain_str)


__all__ = ["log_effective_config", "log_override_chains"]
