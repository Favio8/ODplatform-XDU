#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :service.py
# @Project   :ODPlatform
# @Function  :TrainService — 编排 D5 配置 + D4 校验 + D2 日志 + ultralytics 训练
"""Training orchestration service for D6."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from odp_platform.common.config_log import log_effective_config, log_override_chains
from odp_platform.common.dataset_path import resolve_dataset_path
from odp_platform.common.logging_utils import ROOT_LOGGER_NAME, get_logger
from odp_platform.common.log_rename import rename_log_to_save_dir
from odp_platform.common.model_path import resolve_model_path
from odp_platform.common.paths import RUNS_DIR
from odp_platform.common.plot_style import apply_academic_style
from odp_platform.common.result import TrainMetrics, log_train_metrics
from odp_platform.common.system_utils import log_device_info
from odp_platform.data_validation import render_to_logger, validate_dataset
from odp_platform.runtime_config import build_train_config
from odp_platform.runtime_config.base import ConfigTrace, TrainConfig

from .archive import archive_checkpoints


logger = get_logger(__name__)


def _load_yolo_class() -> type[Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境未安装 ultralytics，无法启动训练。"
        ) from exc
    return YOLO


def _find_project_log_path() -> Path | None:
    root_logger = get_logger(ROOT_LOGGER_NAME)
    for handler in root_logger.handlers:
        if handler.__class__.__name__ == "FileHandler" and hasattr(handler, "baseFilename"):
            return Path(handler.baseFilename)
    return None


@dataclass(frozen=True)
class TrainResult:
    """Immutable snapshot of one training attempt."""

    success: bool
    output_dir: Path
    best_weight: Path | None = None
    last_weight: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    train_time: float | None = None
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None


class TrainService:
    """Orchestrate one end-to-end YOLO training run."""

    def train(
        self,
        yaml_path: str | Path | None = None,
        cli_args: argparse.Namespace | dict[str, Any] | None = None,
        *,
        pre_validate: bool = True,
        archive: bool = True,
        rename_log: bool = True,
        academic_plots: bool = False,
    ) -> TrainResult:
        started_at = datetime.now()
        output_dir: Path | None = None

        try:
            resolved_yaml_path = str(yaml_path) if yaml_path is not None else "train.yaml"
            config, trace = build_train_config(
                yaml_path=resolved_yaml_path,
                cli_args=cli_args,
                ignored_cli_keys={
                    "config",
                    "log_level",
                    "no_pre_validate",
                    "no_archive",
                    "no_rename_log",
                    "academic_plots",
                },
            )

            self._log_context(config, trace, academic_plots=academic_plots)
            raw_data = config.data
            if not raw_data:
                raise ValueError("训练配置缺少 data 字段，请提供数据集 YAML。")

            raw_model = config.model or "yolo11n.pt"
            data_path = resolve_dataset_path(raw_data)
            model_path = resolve_model_path(raw_model)

            logger.info("数据集(声明): %s", raw_data)
            logger.info("数据集(解析): %s", data_path)
            logger.info("模型(声明): %s", raw_model)
            logger.info("模型(解析): %s", model_path)

            if pre_validate:
                self._pre_validate_dataset(data_path, config)

            YOLO = _load_yolo_class()
            model = YOLO(str(model_path))

            yolo_kwargs = config.to_ultralytics_kwargs()
            yolo_kwargs["data"] = str(data_path)
            if not config.project or config.project == "runs":
                yolo_kwargs["project"] = str(RUNS_DIR / f"{config.task_type}_train")
            else:
                yolo_kwargs["project"] = config.project

            logger.info("=" * 60)
            logger.info("启动 YOLO 训练".center(60))
            logger.info("=" * 60)
            logger.info("输出目录(project): %s", yolo_kwargs["project"])

            yolo_results = model.train(**yolo_kwargs)
            output_dir = Path(yolo_results.save_dir)

            logger.info("=" * 60)
            logger.info("训练完成".center(60))
            logger.info("=" * 60)
            metrics = TrainMetrics.from_yolo_results(
                yolo_results,
                model_trainer=getattr(model, "trainer", None),
                task_fallback=config.task_type,
            )
            log_train_metrics(metrics, logger=logger)

            if rename_log:
                rename_log_to_save_dir(output_dir, Path(raw_model).stem)

            archived: dict[str, Path] = {}
            if archive:
                archived = archive_checkpoints(train_dir=output_dir, model_filename=raw_model)

            train_time = (datetime.now() - started_at).total_seconds()
            log_path = _find_project_log_path()
            audit_path = self._write_audit_snapshot(
                output_dir=output_dir,
                config=config,
                trace=trace,
                metrics=metrics,
                archived=archived,
                train_time=train_time,
                log_path=log_path,
            )

            best_weight = archived.get("best") or (output_dir / "weights" / "best.pt")
            last_weight = archived.get("last") or (output_dir / "weights" / "last.pt")

            logger.info("=" * 60)
            logger.info("训练总耗时: %.2f 秒", train_time)
            logger.info("输出目录: %s", output_dir)
            logger.info("最佳权重: %s", best_weight)
            if log_path is not None:
                logger.info("本次日志: %s", log_path)
            logger.info("=" * 60)

            return TrainResult(
                success=True,
                output_dir=output_dir,
                best_weight=best_weight if best_weight.exists() else None,
                last_weight=last_weight if last_weight.exists() else None,
                metrics=metrics.overall,
                train_time=train_time,
                audit_path=audit_path,
                log_path=log_path,
            )
        except Exception as exc:
            logger.error("训练失败: %s", exc, exc_info=True)
            return TrainResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                metrics={},
                train_time=(datetime.now() - started_at).total_seconds(),
                error=str(exc),
                log_path=_find_project_log_path(),
            )

    @staticmethod
    def _log_context(config: TrainConfig, trace: ConfigTrace, *, academic_plots: bool) -> None:
        logger.info("=" * 60)
        logger.info(f"开始 YOLO 训练 (task={config.task_type})".center(60))
        logger.info("=" * 60)
        log_device_info(logger)
        if academic_plots:
            apply_academic_style(logger_instance=logger)
        log_effective_config(config, trace, logger=logger)
        log_override_chains(config, trace, logger=logger)

    @staticmethod
    def _pre_validate_dataset(data_path: Path, config: TrainConfig) -> None:
        logger.info("=" * 60)
        logger.info("数据集预校验 (D4)".center(60))
        logger.info("=" * 60)
        report = validate_dataset(data_path, task_type=config.task_type)
        render_to_logger(report, logger=logger)
        if report.exit_code >= 2:
            error_count = len([result for result in report.results if getattr(result, "severity", None) == "ERROR"])
            raise RuntimeError(
                f"数据集校验失败 ({error_count} 个 ERROR 级问题). "
                f"请先运行 `odp-validate --dataset {data_path.stem} --task {config.task_type}` 修复。"
            )

    @staticmethod
    def _write_audit_snapshot(
        *,
        output_dir: Path,
        config: TrainConfig,
        trace: ConfigTrace,
        metrics: TrainMetrics,
        archived: dict[str, Path],
        train_time: float,
        log_path: Path | None,
    ) -> Path | None:
        audit_path = output_dir / "odp_audit.json"
        payload = {
            "config": config.to_audit_snapshot(),
            "merger": trace.to_audit_log(),
            "metrics": metrics.to_dict(),
            "result_summary": {
                "best_archive": str(archived.get("best")) if archived.get("best") else None,
                "last_archive": str(archived.get("last")) if archived.get("last") else None,
                "train_time_sec": train_time,
                "log_path": str(log_path) if log_path is not None else None,
            },
        }
        try:
            audit_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("审计快照: %s", audit_path)
            return audit_path
        except OSError as exc:
            logger.warning("写审计快照失败(不影响训练结果): %s", exc)
            return None


def train_yolo(
    yaml_path: str | Path | None = None,
    cli_args: argparse.Namespace | dict[str, Any] | None = None,
    *,
    pre_validate: bool = True,
    archive: bool = True,
    rename_log: bool = True,
    academic_plots: bool = False,
) -> TrainResult:
    """Convenience wrapper for one-line training calls."""

    return TrainService().train(
        yaml_path=yaml_path,
        cli_args=cli_args,
        pre_validate=pre_validate,
        archive=archive,
        rename_log=rename_log,
        academic_plots=academic_plots,
    )


__all__ = ["TrainResult", "TrainService", "train_yolo"]
