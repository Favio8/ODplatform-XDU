#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : ValService — 编排 D5 配置 + D4 校验 + D2 系统 + ultralytics 验证
"""验证服务编排器 — D7 ValService.

★ 核心纪律 (跟训练链路的已实现口径对齐):
  - 不重新发明 D5 / D4 / D2 已有的轮子.
  - 不写 YAMLLoader / CLILoader / ConfigMerger 调用(走 build_val_config)
  - 不读 data.yaml 数样本(走 validate_dataset)
  - 不配 logging handler / 不感知 FileHandler 细节
    (handler 由 D2 logging_utils.get_logger() 在 CLI 入口装好)

跟训练链路相比的 4 处差异:
  1. 调 build_val_config 而非 build_train_config (D5)
  2. model_path 用 search_dirs=[CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR]
     —— 优先从 D6 归档的 best/last 里找
  3. 调 model.val 而非 model.train
  4. ValResult 没有 best_weight / last_weight (val 不产权重),
     val_time 替换 train_time(语义化)

骨架 8 阶段流水线跟 D6 同款.

★ 评估【不归档】: 评估产物 ultralytics 已经写在 runs/<task>_val/val<N>/ 下,
odp_audit.json 也记录了 save_dir 路径. 评估报告没有"强下游消费者 + 原始位置
不好找"的组合, 再复制一份是为对称而对称的过度设计. 所以 evaluation/ 下没有
archive.py, evaluate() 也没有 archive 开关. 详见讲义"撞墙②: 对称强迫症".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from odp_platform.common.constants import ODP_META_KEY, TASK_DETECT, TASK_SEGMENT
from odp_platform.common.config_log import log_effective_config, log_override_chains
from odp_platform.common.dataset_path import resolve_dataset_path
from odp_platform.common.log_rename import rename_log_to_save_dir
from odp_platform.common.model_path import resolve_model_path
from odp_platform.common.paths import CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR, RUNS_DIR
from odp_platform.common.result import TrainMetrics, log_train_metrics
from odp_platform.data_validation import render_to_logger, validate_dataset
from odp_platform.runtime_config import build_val_config

logger = logging.getLogger(__name__)


# ============================================================================
# ValMetrics — TrainMetrics 的语义化别名
# ============================================================================
# D6 result.py 的 TrainMetrics dataclass 字段(task / fitness / mAP / precision /
# recall / class_map_50_95)在 val 和 train 完全一样 —— ultralytics 的 DetMetrics
# / SegmentMetrics 是同一种对象. 这里给个语义化别名, 让 D7 调用方读起来更直观:
#
#     from odp_platform.evaluation import ValMetrics    # 验证语义
#
# 而不是从 training 子系统拿一个名字带 "Train" 的对象做验证. 物理 SSoT 仍然
# 在 common.result, 这里只是逻辑别名 —— 跟 D6 training.__init__ 转再导出
# TrainMetrics 是同一思路.
ValMetrics = TrainMetrics


def _infer_task_type_from_dataset_yaml(data_path: Path) -> str | None:
    """Best-effort task inference from dataset yaml metadata."""
    try:
        payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(payload, dict):
        return None

    odp_meta = payload.get(ODP_META_KEY)
    if not isinstance(odp_meta, dict):
        return None

    candidate = str(odp_meta.get("task", "")).strip().lower()
    if candidate in {TASK_DETECT, TASK_SEGMENT}:
        return candidate
    return None


def _find_project_log_path() -> Path | None:
    """从 D2 'odp_platform' 根 logger 找 FileHandler 的实际文件路径.

    只读检查, 不操作 handler. 给 audit JSON 用 —— 让用户能从 odp_audit.json
    一眼看出"这次验证对应哪个 .log 文件".

    返回 None 如果根 logger 没挂 FileHandler.
    """
    root = logging.getLogger("odp_platform")
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)
    return None


# ============================================================================
# 验证结果 dataclass
# ============================================================================

@dataclass(frozen=True)
class ValResult:
    """验证结果一次性快照.

    跟 D6 TrainResult 的字段差异:
      - 没有 best_weight / last_weight (val 不产权重)
      - 没有归档字段 (评估不归档)
      - train_time → val_time (语义化命名)

    success=False 时 output_dir 可能是 'unknown' Path, error 字段填错误描述.
    success=True  时 metrics 至少含 fitness.
    """
    success:    bool
    output_dir: Path
    metrics:    dict[str, float] = field(default_factory=dict)
    val_time:   float | None = None       # 秒
    error:      str | None = None
    audit_path: Path | None = None        # odp_audit.json 的位置
    log_path:   Path | None = None        # 本次验证的日志文件位置(D2 logger 输出)


# ============================================================================
# ValService — 编排器
# ============================================================================

class ValService:
    """YOLO 验证流程编排.

    职责:
      - 调 D5 build_val_config 拿配置(1 行)
      - 调 D4 validate_dataset 预校验数据集(可关)
      - 调 ultralytics YOLO.val 执行验证
      - 提取 ValMetrics 并漂亮打印
      - 写 odp_audit.json 到 output_dir(给未来的 experiment_db 留落点)

    不负责:
      - 装 logging handler / 操作 FileHandler (那是 D2 logging_utils 的事)
      - 解析 argparse(cli 层做)
      - 归档评估报告 (评估不归档 —— ultralytics 已写在 runs/<task>_val/val<N>/ 下,
        没有强下游消费者. 要历史追溯, audit JSON 里的 save_dir + metrics 够了)
      - 持久化到数据库(留给独立 experiment_db 子系统)
    """

    def __init__(self) -> None:
        """__init__ 不接任何参数 —— 配置都通过 evaluate() 传, 跟 D5/D6 同款."""
        pass

    # ---------------------------------------------------------------------
    # 公开入口
    # ---------------------------------------------------------------------
    def evaluate(
        self,
        yaml_path: str | Path | None = None,
        cli_args: dict[str, Any] | None = None,
        *,
        pre_validate: bool = True,
        rename_log: bool = True,
    ) -> ValResult:
        """跑一次完整验证.

        Args:
            yaml_path:    YAML 路径(None 走 paths.runtime_config_path("val") 默认)
            cli_args:     CLI 字典(argparse.Namespace.__dict__ 也 OK)
            pre_validate: 验证前调 D4 validate_dataset(默认 True, fail-fast).
                          注意 D4 校验跟"我们这里说的验证"不是一回事 ——
                          D4 = 数据集本身的健康度校验; 这里 = 用模型评估指标.
            rename_log:   验证后把日志文件名改成 <save_dir.name>_<ts>_<model>.log
                          对齐 ultralytics 的 val/val2/val3 命名(默认 True).
                          本函数不持有 FileHandler —— log_rename 自己去 D2
                          'odp_platform' 根 logger 上找.

        Returns:
            ValResult —— 永不抛, 错误装进 .error.
        """
        start = datetime.now()
        output_dir: Path | None = None

        try:
            # ============================================================
            # 阶段 1: 配置加载 —— 1 行(★ D5 接口承诺兑现)
            # ============================================================
            config, merger = build_val_config(
                yaml_path=yaml_path,
                cli_args=cli_args,
            )

            # ============================================================
            # 阶段 2: 上下文日志(D2 系统 + D5 溯源) + 2 道 fail-fast 防御
            # ============================================================
            logger.info("=" * 60)
            logger.info(f"开始 YOLO 验证 (task={config.task_type})".center(60))
            logger.info("=" * 60)

            raw_model = config.model
            raw_data = config.data

            # ★ 防御 1: model 必填
            # 验证必须指定模型(没有"默认 yolo11n.pt"这种 fallback —— 验证官方
            # 预训练权重量出的是 COCO 指标, 跟用户数据集无关, 毫无意义).
            if not raw_model:
                raise ValueError(
                    "验证必须指定模型 (config.model). 通常是 D6 归档过的权重, 例如:\n"
                    "  odp-val --model train3-20260524-103045-yolo11n-best.pt\n"
                    "或在 val.yaml 里写 model: train3-...-best.pt"
                )
            # ★ 防御 2: data 必填
            # D5 YOLOValConfig.data 是 Optional, 漏到 service 层 raw_data 可能是 None,
            # resolve_dataset_path(None) 会 TypeError. 这里加跟 model 同款的硬要求.
            # (D6 训练能避开这个坑只是因为 YOLOTrainConfig.data 是 Pydantic 必填,
            #  早一步拦住了, 不代表 D7 不需要自己防.)
            if not raw_data:
                raise ValueError(
                    "验证必须指定数据集 (config.data). 通常是 D3 立的数据集 yaml, 例如:\n"
                    "  odp-val --data rsod.yaml\n"
                    "或在 val.yaml 里写 data: rsod.yaml\n"
                    "(如果 val.yaml 还没生成, 先跑: odp-gen-config val)"
                )

            logger.info(f"任务类型:    {config.task_type}")
            logger.info(f"数据集(声明): {raw_data}")
            data_path = resolve_dataset_path(raw_data)            # 内部 log 命中/未命中
            logger.info(f"数据集(解析): {data_path}")
            effective_task_type = config.task_type
            inferred_task_type = _infer_task_type_from_dataset_yaml(data_path)
            if inferred_task_type and inferred_task_type != config.task_type:
                logger.info(
                    "任务类型按数据集 YAML 元数据修正: %s -> %s",
                    config.task_type,
                    inferred_task_type,
                )
                effective_task_type = inferred_task_type
            logger.info(f"模型(声明):  {raw_model}")
            # ★ D7 跟训练产物唯一关键差异: search_dirs 优先 CHECKPOINTS_DIR
            #   (用户用训练归档的 best.pt 跑 val), fallback 到 PRETRAINED_MODELS_DIR.
            model_path = resolve_model_path(
                raw_model,
                search_dirs=[CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR],
            )
            logger.info(f"模型(解析):  {model_path}")

            # D2 系统快照 —— 不传 logger, D2 自己的 module logger 冒泡走到根 logger


            # D5 溯源 —— 按字段维度展示当前值/来源 + 完整来源链
            log_effective_config(config, merger, logger=logger)
            log_override_chains(config, merger, logger=logger)

            # ============================================================
            # 阶段 3: 数据集预校验 —— 1 行(D4, 可关)
            # ============================================================
            if pre_validate:
                logger.info("=" * 60)
                logger.info("数据集预校验 (D4)".center(60))
                logger.info("=" * 60)
                report = validate_dataset(data_path, task_type=effective_task_type)
                render_to_logger(report, logger=logger)
                # exit_code: 0=PASS/INFO 1=WARNING 2=ERROR
                if report.exit_code >= 2:
                    error_count = len([
                        r for r in report.results
                        if getattr(r, "severity", None) == "ERROR"
                    ])
                    raise RuntimeError(
                        f"数据集校验失败 ({error_count} 个 ERROR 级问题). "
                        f"请用 `odp-validate --dataset {data_path.stem} --task {effective_task_type}` "
                        f"修复后再验证. 如要跳过校验跑验证(不推荐), 加 --no-pre-validate."
                    )

            # ============================================================
            # 阶段 4: 加载模型(model_path 已在阶段 2 解析)
            # ============================================================
            model = YOLO(str(model_path))

            # ============================================================
            # 阶段 5: 执行验证 (ultralytics)
            # ============================================================
            yolo_kwargs = config.to_ultralytics_kwargs()
            # 用解析后的绝对路径覆盖 —— 防 ultralytics 拿 'rsod.yaml' 这种相对名
            # 在 cwd 找不到. resolve_dataset_path 在阶段 2 已经跑过.
            yolo_kwargs["data"] = str(data_path)
            # 用户没指定 project 时, 走 RUNS_DIR/<task>_val/ 作为输出根.
            # 扁平化命名 <task>_<mode>, 跟 D6 训练同款: 最终路径形如
            # runs/detect_val/val, runs/detect_val/val2. 顶层 ls runs/ 看到
            # detect_train / detect_val / segment_train / segment_val 平铺一行,
            # 不会有 detect/ 这种"穿过层".
            yolo_kwargs.setdefault("project", str(RUNS_DIR / f"{effective_task_type}_val"))

            logger.info("=" * 60)
            logger.info("启动验证".center(60))
            logger.info("=" * 60)
            logger.info(f"输出目录(project): {yolo_kwargs['project']}")

            yolo_results = model.val(**yolo_kwargs)
            # val 的 save_dir 在不同 ultralytics 版本暴露位置不一, 走兜底提取
            output_dir = self._extract_save_dir(yolo_results, model)

            # ============================================================
            # 阶段 6: 结果指标 (复用训练结果结构, ValMetrics = TrainMetrics)
            # ============================================================
            logger.info("=" * 60)
            logger.info("验证完成".center(60))
            logger.info("=" * 60)
            metrics = ValMetrics.from_yolo_results(
                yolo_results, model_trainer=getattr(model, "validator", None)
            )
            log_train_metrics(metrics, logger=logger)

            # ============================================================
            # 阶段 7: 整理输出 (D7 本地, 可关)
            # 评估【不归档】—— ultralytics 已经把 results.csv / confusion_matrix.png
            # / 曲线全写到 runs/<task>_val/val<N>/ 下了. 唯一保留的"整理输出"动作
            # 是日志改名(把日志文件名跟 ultralytics save_dir 对齐, 方便用户
            # ls logging/val/ 看哪份日志对应哪次 val).
            # ============================================================
            model_stem = Path(raw_model).stem

            # 7a. 重命名日志文件跟 save_dir 对齐
            if rename_log:
                rename_log_to_save_dir(output_dir, model_stem)

            # ============================================================
            # 阶段 8: 审计快照 (★ 写 odp_audit.json —— schema 跟 D6 兼容,
            #         给未来 experiment_db 跨 train/val 一次性读)
            # ============================================================
            audit_path = output_dir / "odp_audit.json"
            log_path = _find_project_log_path()
            try:
                audit_payload = {
                    "kind":    "val",                            # ★ 区分跟 train 的 audit
                    "config":  config.to_audit_snapshot(),       # D5
                    "merger":  merger.to_audit_log(),            # D5
                    "metrics": metrics.to_dict(),                # D6 (ValMetrics = TrainMetrics)
                    "result_summary": {
                        "val_time_sec": (datetime.now() - start).total_seconds(),
                        "log_path": str(log_path) if log_path else None,
                        # 注: save_dir 已经在 metrics["save_dir"] 里, 给 experiment_db
                        #     提供"哪次 val 的产物在 runs/ 哪里"的稳定引用.
                    },
                }
                audit_path.write_text(
                    json.dumps(audit_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(f"审计快照: {audit_path}")
            except OSError as e:
                logger.warning(f"写审计快照失败(不影响验证结果): {e}")
                audit_path = None

            # ============================================================
            # 收尾 —— ValResult
            # ============================================================
            val_time = (datetime.now() - start).total_seconds()

            logger.info("=" * 60)
            logger.info(f"验证总耗时: {val_time:.2f} 秒")
            logger.info(f"输出目录:   {output_dir}")
            if log_path:
                logger.info(f"本次日志:   {log_path}")
            logger.info("=" * 60)

            return ValResult(
                success=True,
                output_dir=output_dir,
                metrics=metrics.overall,
                val_time=val_time,
                audit_path=audit_path,
                log_path=log_path,
            )

        # =====================================================================
        # 顶层异常拦截 —— 永不抛, 打包成 ValResult.error
        # =====================================================================
        except Exception as e:
            logger.error(f"验证失败: {e}", exc_info=True)
            val_time = (datetime.now() - start).total_seconds()
            return ValResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                metrics={},
                val_time=val_time,
                error=str(e),
                log_path=_find_project_log_path(),       # 失败也带日志, 方便排查
            )

    # ---------------------------------------------------------------------
    # 内部辅助
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_save_dir(yolo_results: Any, model: Any) -> Path:
        """从 yolo_results / model.validator 提取 val 的 save_dir.

        ultralytics 不同版本的 val results 对象暴露 save_dir 的位置不一致:
        - 某些版本: results.save_dir 直接可用
        - 某些版本: results 没 save_dir, 要从 model.validator.save_dir 拿
        - 极端: 两个都没, 兜底用 Path('unknown') 让上游能感知到问题
        """
        save_dir = getattr(yolo_results, "save_dir", None)
        if save_dir is not None:
            return Path(save_dir)
        validator = getattr(model, "validator", None)
        if validator is not None:
            save_dir = getattr(validator, "save_dir", None)
            if save_dir is not None:
                return Path(save_dir)
        logger.warning(
            "无法从 ultralytics 提取 save_dir, 走 'unknown' 兜底. "
            "日志改名可能跳过."
        )
        return Path("unknown")


# ============================================================================
# 便捷函数 —— 给"我不要管 service 实例化"的用户用
# ============================================================================

def evaluate_yolo(
    yaml_path: str | Path | None = None,
    cli_args: dict[str, Any] | None = None,
    *,
    pre_validate: bool = True,
    rename_log: bool = True,
) -> ValResult:
    """一行启动验证 —— 风格跟 D5 build_val_config / D6 train_yolo 一致.
    True
    """
    service = ValService()
    return service.evaluate(
        yaml_path=yaml_path,
        cli_args=cli_args,
        pre_validate=pre_validate,
        rename_log=rename_log,
    )
