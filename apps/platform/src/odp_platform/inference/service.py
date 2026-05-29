#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Teacher-aligned inference service with ODPlatform compatibility wrappers."""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from odp_platform.common.config_log import log_effective_config, log_override_chains
from odp_platform.common.constants import TASK_SEGMENT
from odp_platform.common.dataset_path import resolve_dataset_path
from odp_platform.common.log_rename import rename_log_to_save_dir
from odp_platform.common.logging_utils import ROOT_LOGGER_NAME, get_logger
from odp_platform.common.model_path import resolve_model_path
from odp_platform.common.paths import CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR, RUNS_DIR
from odp_platform.runtime_config import build_infer_config

from odp_platform.visualization import BeautifyVisualizer, DrawStyle

from .cancel import CancelToken
from .components import InferenceArtifact, InferenceSummary
from .hooks import InferHooks
from .pipeline import ThreadedPipeline
from .pipeline_config import PipelineConfig, load_pipeline_config
from .sinks import LocalFileSink, NullSink, OutputSink


logger = get_logger(__name__)


_PREDICT_KEYS: tuple[str, ...] = (
    "conf",
    "iou",
    "imgsz",
    "max_det",
    "classes",
    "agnostic_nms",
    "augment",
    "device",
    "retina_masks",
)

_BUILD_IGNORED_KEYS: frozenset[str] = frozenset(
    {
        "config",
        "log_level",
        "preflight_only",
        "pipeline_yaml",
        "threaded",
        "warmup",
        "warmup_frames",
        "window_name",
        "show_info",
        "beautify",
        "rename_log",
    }
)


def _load_yolo_class() -> type[Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("当前 Python 环境未安装 ultralytics，无法启动推理。") from exc
    return YOLO


def _find_project_log_path() -> Path | None:
    root_logger = get_logger(ROOT_LOGGER_NAME)
    for handler in root_logger.handlers:
        if handler.__class__.__name__ == "FileHandler" and hasattr(handler, "baseFilename"):
            return Path(handler.baseFilename)
    return None


def _resolve_output_dir(base: Path, name: str, *, exist_ok: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / name
    if exist_ok or not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    index = 2
    while (base / f"{name}{index}").exists():
        index += 1
    output_dir = base / f"{name}{index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_output_dir_from_config(config) -> Path:
    base = Path(config.project) if config.project and config.project != "runs" else RUNS_DIR / f"{config.task}_infer"
    run_name = config.experiment_name or config.name or "predict"
    return _resolve_output_dir(base, run_name, exist_ok=bool(getattr(config, "exist_ok", False)))


def _as_mapping(namespace_or_mapping: argparse.Namespace | dict[str, Any] | None) -> dict[str, Any]:
    if namespace_or_mapping is None:
        return {}
    if isinstance(namespace_or_mapping, dict):
        return dict(namespace_or_mapping)
    return vars(namespace_or_mapping)


def _get_cli_option(namespace_or_mapping: argparse.Namespace | dict[str, Any] | None, key: str, default: Any = None) -> Any:
    values = _as_mapping(namespace_or_mapping)
    return values.get(key, default)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 4), dtype=float)


def _empty_conf() -> np.ndarray:
    return np.zeros((0,), dtype=float)


def _collect_artifacts(output_dir: Path) -> tuple[InferenceArtifact, ...]:
    artifacts: list[InferenceArtifact] = []
    if not output_dir.exists():
        return tuple()

    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            artifacts.append(InferenceArtifact(input_name=path.stem, output_path=path, kind="video"))
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            artifacts.append(InferenceArtifact(input_name=path.name, output_path=path, kind="image"))
    return tuple(artifacts)


@dataclass
class InferStats:
    """Runtime statistics for one inference run."""

    frames: int = 0
    detections: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    infer_time_sec: float = 0.0
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    render_fps: float = 0.0
    loop_fps: float = 0.0
    current_fps: float = 0.0
    speed_ms: dict[str, float] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.infer_time_sec if self.infer_time_sec > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.infer_time_sec / self.frames * 1000.0) if self.frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "per_class": dict(self.per_class),
            "infer_time_sec": round(self.infer_time_sec, 4),
            "avg_fps": round(self.avg_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": {
                "capture": self.capture_fps,
                "infer": self.infer_fps,
                "render": self.render_fps,
                "loop": self.loop_fps,
                "current": self.current_fps,
            },
            "speed_ms": dict(self.speed_ms),
        }


def log_infer_stats(stats: InferStats, *, logger: logging.Logger = logger) -> None:
    logger.info("处理帧数:   %s", stats.frames)
    logger.info("检测总数:   %s", stats.detections)
    logger.info("平均延迟:   %.2f ms/帧", stats.avg_latency_ms)
    logger.info(
        "帧率(FPS):  捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f | 当前 %.1f",
        stats.capture_fps,
        stats.infer_fps,
        stats.render_fps,
        stats.loop_fps,
        stats.current_fps,
    )
    if stats.speed_ms:
        logger.info(
            "模型 speed(ms): 预处理 %.2f | 推理 %.2f | 后处理 %.2f",
            stats.speed_ms.get("preprocess", 0.0),
            stats.speed_ms.get("inference", 0.0),
            stats.speed_ms.get("postprocess", 0.0),
        )
    if stats.per_class:
        logger.info("各类别检测数:")
        for name, count in sorted(stats.per_class.items(), key=lambda item: -item[1]):
            logger.info("    %-20s %s", name, count)


@dataclass(frozen=True)
class InferResult:
    """Teacher-style inference result snapshot."""

    success: bool
    output_dir: Path
    stats: dict[str, Any] = field(default_factory=dict)
    infer_time: float | None = None
    saved: bool = False
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None
    source: str | None = None
    artifacts: tuple[InferenceArtifact, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class InferenceResult:
    """Compatibility result used by the current CLI and tests."""

    success: bool
    output_dir: Path
    summary: InferenceSummary | None = None
    artifact_paths: tuple[Path, ...] = field(default_factory=tuple)
    audit_path: Path | None = None
    log_path: Path | None = None
    error: str | None = None


class InferService:
    """Teacher-aligned inference orchestrator."""

    def predict(
        self,
        yaml_path: str | Path | None = None,
        pipeline_yaml: str | Path | None = None,
        cli_args: argparse.Namespace | dict[str, Any] | None = None,
        *,
        beautify: bool = True,
        rename_log: bool = True,
        threaded: bool = False,
        warmup_frames: int = 0,
        window_name: str = "odp-infer",
        show_info: bool = True,
        output_sink: OutputSink | None = None,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
    ) -> InferResult:
        del threaded  # 当前对齐老师版时统一走多线程流水线

        hooks = hooks or InferHooks()
        started = datetime.now()
        output_dir: Path | None = None
        raw_source: str | None = None

        try:
            config, trace = build_infer_config(
                yaml_path=str(yaml_path) if yaml_path is not None else "infer.yaml",
                cli_args=cli_args,
                ignored_cli_keys=set(_BUILD_IGNORED_KEYS),
            )
            pipe: PipelineConfig = load_pipeline_config(pipeline_yaml)

            logger.info("=" * 60)
            logger.info(f"开始 YOLO 推理 (task={config.task})".center(60))
            logger.info("=" * 60)

            raw_model = config.model or "yolo11n.pt"
            raw_source = config.source
            logger.info("任务类型:    %s", config.task)
            logger.info("输入源(声明): %r", raw_source)
            logger.info("模型(声明):  %s", raw_model)

            log_effective_config(config, trace, logger=logger)
            log_override_chains(config, trace, logger=logger)

            if raw_source is None:
                raise RuntimeError("未指定推理输入源. 请在 infer.yaml 写 source, 或用 odp-infer --source 传入.")

            model_path = resolve_model_path(raw_model, search_dirs=[CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR])
            logger.info("模型(解析):  %s", model_path)
            if config.data:
                logger.info("数据集(声明): %s", config.data)
                logger.info("数据集(解析): %s", resolve_dataset_path(config.data))

            YOLO = _load_yolo_class()
            model = YOLO(str(model_path))

            class_names: list[str]
            names = getattr(model, "names", {})
            if isinstance(names, dict):
                class_names = [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
            else:
                class_names = [str(value) for value in names]

            do_beautify = beautify and pipe.viz_enabled
            visualizer: BeautifyVisualizer | None = None
            if do_beautify:
                visualizer = BeautifyVisualizer(
                    labels=class_names,
                    label_mapping=pipe.label_mapping or None,
                    color_mapping=pipe.color_mapping or None,
                    default_color=pipe.default_color,
                    font_path=pipe.font_path,
                )
            else:
                logger.info("美化已关闭, 使用 YOLO 原生 plot() 绘制.")

            output_dir = _resolve_output_dir_from_config(config)
            logger.info("输出目录:    %s", output_dir)

            predict_kwargs = {
                key: getattr(config, key)
                for key in _PREDICT_KEYS
                if getattr(config, key, None) is not None
            }
            predict_kwargs["verbose"] = False

            want_save = bool(getattr(config, "save", True))
            want_show = bool(getattr(config, "show", False))
            if output_sink is None:
                output_sink = LocalFileSink() if want_save else NullSink()
            else:
                logger.info("使用调用方提供的 sink: %s", output_sink.__class__.__name__)

            logger.info("=" * 60)
            logger.info("启动推理".center(60))
            logger.info("=" * 60)

            stats = InferStats()
            camera_config = pipe.build_camera_config()
            processor = _FrameProcessor(
                model=model,
                predict_kwargs=predict_kwargs,
                do_beautify=do_beautify,
                visualizer=visualizer,
                use_label_mapping=pipe.use_label_mapping,
                style_overrides=pipe.style_overrides,
                names=model.names,
                task_type=config.task_type,
                line_width=config.line_width,
                show_boxes=config.show_boxes,
                show_labels=config.show_labels,
                show_conf=config.show_conf,
            )

            raw_batch = getattr(config, "batch", 16)
            batch_size = raw_batch if isinstance(raw_batch, int) and raw_batch >= 1 else 16

            pipeline = ThreadedPipeline(
                processor=processor,
                source=str(raw_source),
                camera_config=camera_config,
                output_dir=output_dir,
                output_sink=output_sink,
                batch_size=batch_size,
                save=want_save,
                show=want_show,
                show_info=show_info,
                window_name=window_name,
                warmup_frames=warmup_frames,
                hooks=hooks,
                cancel_token=cancel_token,
                save_frames=bool(getattr(config, "save_frames", False)),
                save_txt=bool(getattr(config, "save_txt", False)),
                save_conf=bool(getattr(config, "save_conf", False)),
                save_crop=bool(getattr(config, "save_crop", False)),
            )
            interrupted = pipeline.run(stats)
            if interrupted:
                logger.warning("推理被用户提前结束.")

            logger.info("=" * 60)
            logger.info("推理完成".center(60))
            logger.info("=" * 60)
            log_infer_stats(stats, logger=logger)

            model_stem = Path(raw_model).stem
            if rename_log:
                rename_log_to_save_dir(output_dir, model_stem)

            log_path = _find_project_log_path()
            artifacts = _collect_artifacts(output_dir)
            audit_path = self._write_audit_snapshot(
                output_dir=output_dir,
                config=config,
                trace=trace,
                pipe=pipe,
                stats=stats,
                saved=want_save,
                beautified=do_beautify,
                started=started,
                log_path=log_path,
            )

            infer_time = (datetime.now() - started).total_seconds()
            logger.info("=" * 60)
            logger.info("推理总耗时: %.2f 秒", infer_time)
            logger.info("输出目录:   %s", output_dir)
            if want_save:
                logger.info("结果已保存到上面的目录.")
            if log_path:
                logger.info("本次日志:   %s", log_path)
            logger.info("=" * 60)

            result = InferResult(
                success=True,
                output_dir=output_dir,
                stats=stats.to_dict(),
                infer_time=infer_time,
                saved=want_save,
                audit_path=audit_path,
                log_path=log_path,
                source=str(raw_source),
                artifacts=artifacts,
            )
            hooks.fire_complete(result)
            return result
        except Exception as exc:
            logger.error("推理失败: %s", exc, exc_info=True)
            hooks.fire_error(exc)
            return InferResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                stats={},
                infer_time=(datetime.now() - started).total_seconds(),
                error=str(exc),
                log_path=_find_project_log_path(),
                source=raw_source,
            )

    @staticmethod
    def _write_audit_snapshot(
        *,
        output_dir: Path,
        config,
        trace,
        pipe: PipelineConfig,
        stats: InferStats,
        saved: bool,
        beautified: bool,
        started: datetime,
        log_path: Path | None,
    ) -> Path | None:
        audit_path = output_dir / "odp_audit.json"
        payload = {
            "mode": "infer",
            "config": config.to_audit_snapshot(),
            "merger": trace.to_audit_log(),
            "pipeline": pipe.to_audit(),
            "stats": stats.to_dict(),
            "result_summary": {
                "output_dir": str(output_dir),
                "saved": saved,
                "beautified": beautified,
                "infer_time_sec": (datetime.now() - started).total_seconds(),
                "log_path": str(log_path) if log_path else None,
            },
        }
        try:
            audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("审计快照:   %s", audit_path)
            return audit_path
        except OSError as exc:
            logger.warning("写审计快照失败(不影响推理结果): %s", exc)
            return None


class InferenceService(InferService):
    """Compatibility wrapper used by the current CLI and tests."""

    def infer(
        self,
        yaml_path: str | Path | None = None,
        cli_args: argparse.Namespace | dict[str, Any] | None = None,
    ) -> InferenceResult:
        pipeline_yaml = _get_cli_option(cli_args, "pipeline_yaml")
        warmup_frames = int(_get_cli_option(cli_args, "warmup_frames", _get_cli_option(cli_args, "warmup", 0)) or 0)
        show_info = _get_cli_option(cli_args, "show_info", True)
        beautify = _get_cli_option(cli_args, "beautify", True)
        rename_log = _get_cli_option(cli_args, "rename_log", True)
        threaded = bool(_get_cli_option(cli_args, "threaded", False))
        window_name = str(_get_cli_option(cli_args, "window_name", "odp-infer"))

        result = self.predict(
            yaml_path=yaml_path,
            pipeline_yaml=pipeline_yaml,
            cli_args=cli_args,
            beautify=bool(beautify),
            rename_log=bool(rename_log),
            threaded=threaded,
            warmup_frames=warmup_frames,
            window_name=window_name,
            show_info=bool(show_info),
        )

        stats = result.stats or {}
        summary = InferenceSummary(
            frames_processed=int(stats.get("frames", 0) or 0),
            detections_total=int(stats.get("detections", 0) or 0),
            source=result.source or str(_get_cli_option(cli_args, "source", "unknown")),
            artifacts=result.artifacts,
        )
        return InferenceResult(
            success=result.success,
            output_dir=result.output_dir,
            summary=summary,
            artifact_paths=tuple(artifact.output_path for artifact in result.artifacts),
            audit_path=result.audit_path,
            log_path=result.log_path,
            error=result.error,
        )


@dataclass
class _FrameProcessor:
    model: Any
    predict_kwargs: dict[str, Any]
    do_beautify: bool
    visualizer: BeautifyVisualizer | None
    use_label_mapping: bool
    style_overrides: dict[str, Any]
    names: dict[int, str] | list[str]
    task_type: str
    line_width: int | None
    show_boxes: bool
    show_labels: bool
    show_conf: bool
    _style: DrawStyle | None = None

    def infer_batch(self, images: list[np.ndarray]):
        started = time.perf_counter()
        results = self.model(images, **self.predict_kwargs)
        batch_dt = time.perf_counter() - started
        labels_list: list[list[str]] = []
        counts: list[int] = []
        for result in results:
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            counts.append(count)
            if count == 0:
                labels_list.append([])
                continue
            cls_tensor = boxes.cls.int().cpu().tolist()
            labels_list.append([_class_name(self.names, class_id) for class_id in cls_tensor])
        return results, labels_list, counts, batch_dt

    def draw(self, image: np.ndarray, result: Any, labels: list[str], n: int) -> np.ndarray:
        if self.do_beautify and self.visualizer is not None:
            if self._style is None:
                height, width = image.shape[:2]
                style_kwargs = dict(self.style_overrides)
                if self.line_width is not None:
                    style_kwargs["line_width"] = self.line_width
                self._style = DrawStyle.from_image_size(height, width, **style_kwargs)

            base_image = image.copy()
            if self.task_type == TASK_SEGMENT and getattr(result, "masks", None) is not None:
                base_image = result.plot(
                    labels=False,
                    conf=False,
                    boxes=False,
                    line_width=self.line_width,
                )
                base_image = np.asarray(base_image).copy()

            boxes = result.boxes
            detections = BeautifyVisualizer.from_yolo_results(
                boxes=(boxes.xyxy.cpu().numpy() if n else _empty_boxes()),
                confidences=(boxes.conf.cpu().numpy() if n else _empty_conf()),
                labels=labels,
            )
            return self.visualizer.draw(
                base_image,
                detections,
                style=self._style,
                use_label_mapping=self.use_label_mapping,
                draw_boxes=self.show_boxes,
                show_labels=self.show_labels,
                show_conf=self.show_conf,
            )

        return result.plot(
            labels=self.show_labels,
            conf=self.show_conf,
            boxes=self.show_boxes,
            line_width=self.line_width,
        )


def _class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def infer_yolo(
    yaml_path: str | Path | None = None,
    pipeline_yaml: str | Path | None = None,
    cli_args: argparse.Namespace | dict[str, Any] | None = None,
    *,
    beautify: bool = True,
    rename_log: bool = True,
    threaded: bool = False,
    warmup_frames: int = 0,
    window_name: str = "odp-infer",
    show_info: bool = True,
    output_sink: OutputSink | None = None,
    hooks: InferHooks | None = None,
    cancel_token: CancelToken | None = None,
) -> InferResult:
    service = InferService()
    return service.predict(
        yaml_path=yaml_path,
        pipeline_yaml=pipeline_yaml,
        cli_args=cli_args,
        beautify=beautify,
        rename_log=rename_log,
        threaded=threaded,
        warmup_frames=warmup_frames,
        window_name=window_name,
        show_info=show_info,
        output_sink=output_sink,
        hooks=hooks,
        cancel_token=cancel_token,
    )


def run_inference(
    yaml_path: str | Path | None = None,
    cli_args: argparse.Namespace | dict[str, Any] | None = None,
) -> InferenceResult:
    return InferenceService().infer(yaml_path=yaml_path, cli_args=cli_args)


__all__ = [
    "InferResult",
    "InferService",
    "InferStats",
    "InferenceResult",
    "InferenceService",
    "infer_yolo",
    "run_inference",
]
