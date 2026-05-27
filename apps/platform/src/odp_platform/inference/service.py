"""Inference orchestration service."""

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
from odp_platform.common.model_path import resolve_model_path
from odp_platform.common.paths import RUNS_DIR
from odp_platform.common.system_utils import log_device_info
from odp_platform.frame_source import create_frame_source
from odp_platform.runtime_config import build_infer_config
from odp_platform.runtime_config.base import ConfigTrace, InferConfig

from .components import InferenceArtifact, InferenceSummary
from .pipeline import InferencePipeline


logger = get_logger(__name__)


def _load_yolo_class() -> type[Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 ultralytics，无法启动推理。") from exc
    return YOLO


def _find_project_log_path() -> Path | None:
    root_logger = get_logger(ROOT_LOGGER_NAME)
    for handler in root_logger.handlers:
        if handler.__class__.__name__ == "FileHandler" and hasattr(handler, "baseFilename"):
            return Path(handler.baseFilename)
    return None


@dataclass(frozen=True)
class InferenceResult:
    """Immutable snapshot of one inference attempt."""

    success: bool
    output_dir: Path
    summary: InferenceSummary | None = None
    artifact_paths: tuple[Path, ...] = field(default_factory=tuple)
    audit_path: Path | None = None
    log_path: Path | None = None
    error: str | None = None


class InferenceService:
    """Build config, run YOLO prediction, and save beautified outputs."""

    def infer(
        self,
        yaml_path: str | Path | None = None,
        cli_args: argparse.Namespace | dict[str, Any] | None = None,
    ) -> InferenceResult:
        started_at = datetime.now()
        output_dir: Path | None = None

        try:
            resolved_yaml_path = str(yaml_path) if yaml_path is not None else "infer.yaml"
            config, trace = build_infer_config(
                yaml_path=resolved_yaml_path,
                cli_args=cli_args,
                ignored_cli_keys={"config", "log_level", "preflight_only"},
            )

            self._log_context(config, trace)
            raw_source = config.source
            if not raw_source:
                raise ValueError("推理配置缺少 source 字段，请提供输入源。")
            raw_model = config.model
            if not raw_model:
                raise ValueError("推理配置缺少 model 字段，请提供模型权重。")

            model_path = resolve_model_path(raw_model)
            logger.info("输入源: %s", raw_source)
            logger.info("模型(声明): %s", raw_model)
            logger.info("模型(解析): %s", model_path)
            if config.data:
                logger.info("数据集(声明): %s", config.data)
                logger.info("数据集(解析): %s", resolve_dataset_path(config.data))

            YOLO = _load_yolo_class()
            model = YOLO(str(model_path))

            output_dir = self._resolve_output_dir(config)
            output_dir.mkdir(parents=True, exist_ok=True)
            source = create_frame_source(raw_source)
            pipeline = InferencePipeline(
                model=model,
                config=config,
                source=source,
                output_dir=output_dir,
            )
            summary = pipeline.run()
            log_path = _find_project_log_path()
            audit_path = self._write_audit_snapshot(
                output_dir=output_dir,
                config=config,
                trace=trace,
                summary=summary,
                log_path=log_path,
                started_at=started_at,
            )

            logger.info("=" * 60)
            logger.info("推理完成".center(60))
            logger.info("=" * 60)
            logger.info("输出目录: %s", output_dir)
            logger.info("处理帧数: %s", summary.frames_processed)
            logger.info("检测总数: %s", summary.detections_total)
            for artifact in summary.artifacts:
                logger.info("产物(%s): %s", artifact.kind, artifact.output_path)

            return InferenceResult(
                success=True,
                output_dir=output_dir,
                summary=summary,
                artifact_paths=tuple(artifact.output_path for artifact in summary.artifacts),
                audit_path=audit_path,
                log_path=log_path,
            )
        except Exception as exc:
            logger.error("推理失败: %s", exc, exc_info=True)
            return InferenceResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                log_path=_find_project_log_path(),
                error=str(exc),
            )

    @staticmethod
    def _log_context(config: InferConfig, trace: ConfigTrace) -> None:
        logger.info("=" * 60)
        logger.info(f"开始 YOLO 推理 (task={config.task_type})".center(60))
        logger.info("=" * 60)
        log_device_info(logger)
        log_effective_config(config, trace, logger=logger)
        log_override_chains(config, trace, logger=logger)
        if config.show_labels and not config.show_boxes:
            logger.warning("当前美化绘制会保留标签背景；show_boxes=False 时仍会按框位置放置标签。")

    @staticmethod
    def _resolve_output_dir(config: InferConfig) -> Path:
        project = Path(config.project) if config.project and config.project != "runs" else RUNS_DIR / f"{config.task_type}_infer"
        if config.name:
            candidate = project / config.name
            if candidate.exists() and not config.exist_ok:
                raise FileExistsError(f"输出目录已存在: {candidate}，如需复用请设置 exist_ok=True。")
            return candidate
        timestamp = datetime.now().strftime("infer-%Y%m%d-%H%M%S")
        return project / timestamp

    @staticmethod
    def _write_audit_snapshot(
        *,
        output_dir: Path,
        config: InferConfig,
        trace: ConfigTrace,
        summary: InferenceSummary,
        log_path: Path | None,
        started_at: datetime,
    ) -> Path | None:
        audit_path = output_dir / "odp_audit.json"
        payload = {
            "config": config.to_audit_snapshot(),
            "merger": trace.to_audit_log(),
            "result_summary": {
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "frames_processed": summary.frames_processed,
                "detections_total": summary.detections_total,
                "source": summary.source,
                "artifacts": [
                    {
                        "input_name": artifact.input_name,
                        "output_path": str(artifact.output_path),
                        "kind": artifact.kind,
                    }
                    for artifact in summary.artifacts
                ],
                "log_path": str(log_path) if log_path is not None else None,
            },
        }
        try:
            audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("审计快照: %s", audit_path)
            return audit_path
        except OSError as exc:
            logger.warning("写审计快照失败(不影响推理结果): %s", exc)
            return None


def run_inference(
    yaml_path: str | Path | None = None,
    cli_args: argparse.Namespace | dict[str, Any] | None = None,
) -> InferenceResult:
    """Convenience wrapper for one-line inference calls."""

    return InferenceService().infer(yaml_path=yaml_path, cli_args=cli_args)


__all__ = ["InferenceResult", "InferenceService", "run_inference"]
