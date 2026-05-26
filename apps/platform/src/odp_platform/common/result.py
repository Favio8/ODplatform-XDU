#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :result.py
# @Project   :ODPlatform
# @Function  :训练/验证结果 dataclass + 日志输出函数
"""Training/validation result helpers shared by multiple subsystems."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from odp_platform.common.constants import TASK_DETECT, TASK_SEGMENT
from odp_platform.common.string_utils import pad_to_width


logger = logging.getLogger(__name__)

_METRIC_FIELDS_BY_TASK: dict[str, list[tuple[str, str]]] = {
    TASK_DETECT: [
        ("metrics/precision(B)", "Precision(B)"),
        ("metrics/recall(B)", "Recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)"),
    ],
    TASK_SEGMENT: [
        ("metrics/precision(B)", "Precision(B)"),
        ("metrics/recall(B)", "Recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)"),
        ("metrics/precision(M)", "Precision(M)"),
        ("metrics/recall(M)", "Recall(M)"),
        ("metrics/mAP50(M)", "mAP50(M)"),
        ("metrics/mAP50-95(M)", "mAP50-95(M)"),
    ],
}


def _safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TrainMetrics:
    """Structured metrics snapshot from one Ultralytics training or validation run."""

    task: str
    save_dir: Path
    timestamp: str
    speed_ms: dict[str, float]
    overall: dict[str, float]
    class_map_50_95: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yolo_results(
        cls,
        results: Any,
        *,
        model_trainer: Any = None,
        task_fallback: str | None = None,
    ) -> "TrainMetrics":
        task = getattr(results, "task", None) or task_fallback or "unknown"

        save_dir_raw = getattr(results, "save_dir", None)
        if save_dir_raw is None and model_trainer is not None:
            save_dir_raw = getattr(model_trainer, "save_dir", None)
        save_dir = Path(save_dir_raw) if save_dir_raw is not None else Path("unknown")

        speed_raw = getattr(results, "speed", {}) or {}
        speed_ms = {
            "preprocess": _safe_float(speed_raw.get("preprocess")),
            "inference": _safe_float(speed_raw.get("inference")),
            "loss": _safe_float(speed_raw.get("loss")),
            "postprocess": _safe_float(speed_raw.get("postprocess")),
        }
        valid_values = [value for value in speed_ms.values() if not math.isnan(value)]
        speed_ms["total"] = sum(valid_values) if valid_values else math.nan

        results_dict = getattr(results, "results_dict", {}) or {}
        overall = {"fitness": _safe_float(getattr(results, "fitness", None))}
        for key, value in results_dict.items():
            overall[key] = _safe_float(value)

        class_map: dict[str, float] = {}
        names = getattr(results, "names", {}) or {}
        maps = getattr(results, "maps", np.array([]))
        if isinstance(names, list):
            names = {index: name for index, name in enumerate(names)}
        if names and hasattr(maps, "size") and maps.size > 0:
            for index, class_name in names.items():
                if isinstance(index, int) and index < maps.size:
                    class_map[str(class_name)] = _safe_float(maps[index])

        return cls(
            task=task,
            save_dir=save_dir,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            speed_ms=speed_ms,
            overall=overall,
            class_map_50_95=class_map,
        )

    def to_dict(self) -> dict[str, Any]:
        def clean_nan(payload: dict[str, float]) -> dict[str, float | None]:
            return {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in payload.items()
            }

        return {
            "task": self.task,
            "save_dir": str(self.save_dir),
            "timestamp": self.timestamp,
            "speed_ms": clean_nan(self.speed_ms),
            "overall": clean_nan(self.overall),
            "class_map_50_95": clean_nan(self.class_map_50_95),
        }


def log_train_metrics(
    metrics: TrainMetrics,
    *,
    logger: logging.Logger | None = None,
    key_width: int = 20,
    section_width: int = 60,
) -> None:
    """Render a metrics summary into the provided logger."""

    log = logger or logging.getLogger(__name__)
    line = "=" * section_width
    thin = "-" * section_width

    log.info(line)
    log.info(f"训练结果 ({metrics.task.capitalize()} Task)".center(section_width))
    log.info(line)
    log.info("基本信息".center(section_width))
    log.info(thin)
    log.info("%s: %s", pad_to_width("任务类型", key_width), metrics.task)
    log.info("%s: %s", pad_to_width("保存目录", key_width), metrics.save_dir)
    log.info("%s: %s", pad_to_width("时间戳", key_width), metrics.timestamp)

    log.info("处理速度 (ms/image)".center(section_width))
    log.info(thin)
    for display_name, data_key in [
        ("预处理", "preprocess"),
        ("推理", "inference"),
        ("损失计算", "loss"),
        ("后处理", "postprocess"),
        ("总计", "total"),
    ]:
        value = metrics.speed_ms.get(data_key, math.nan)
        log.info("%s: %.3f ms", pad_to_width(display_name, key_width), value)

    log.info("整体评估指标".center(section_width))
    log.info(thin)
    log.info("%s: %.4f", pad_to_width("Fitness 分数", key_width), metrics.overall.get("fitness", math.nan))

    metric_fields = _METRIC_FIELDS_BY_TASK.get(metrics.task, [])
    if metric_fields:
        for raw_key, display_name in metric_fields:
            log.info("%s: %.4f", pad_to_width(display_name, key_width), metrics.overall.get(raw_key, math.nan))
    else:
        log.info("当前任务类型 '%s' 的详细评估指标未完全支持。", metrics.task)
        for key, value in metrics.overall.items():
            if key == "fitness":
                continue
            log.info("%s: %.4f", pad_to_width(key, key_width), value)

    if metrics.class_map_50_95:
        log.info("类别级 mAP@0.5:0.95 (Box)".center(section_width))
        log.info(thin)
        valid_items = {
            class_name: metric
            for class_name, metric in metrics.class_map_50_95.items()
            if not math.isnan(metric)
        }
        if valid_items:
            for class_name, metric in sorted(valid_items.items(), key=lambda item: item[1], reverse=True):
                log.info("%s: %.4f", pad_to_width(class_name, key_width), metric)
        else:
            log.warning("类别 mAP 全为 NaN，跳过打印。")

    log.info(line)


__all__ = ["TrainMetrics", "log_train_metrics"]
