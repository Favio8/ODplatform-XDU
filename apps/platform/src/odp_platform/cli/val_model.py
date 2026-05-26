#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : evaluate_model.py
# @Project   : ODPlatform
# @Function  : odp-val entry-point —— YOLO 模型验证命令行入口
"""odp-val —— 验证一个训练好的 YOLO 模型, 量 mAP / precision / recall.

CLI 层只做三件事(跟 D6 odp-train 同款):
  1. 解析 argparse
  2. 调 D2 get_logger() 装 handler —— 全程序唯一装 handler 的地方
  3. 调 ValService().evaluate(), 按 ValResult.success 决定退出码

不做: 解析 yaml / 合并配置 / 碰 ultralytics —— 全在 service 及其下游.
"""
from __future__ import annotations

import argparse
import logging
import sys

from odp_platform.common.constants import SUPPORTED_TASKS
from odp_platform.common.logging_utils import get_logger, setup_logging
from odp_platform.common.paths import LOGGING_DIR
from odp_platform.evaluation.service import ValService


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-val",
        description="验证一个训练好的 YOLO 模型 (ODPlatform evaluation 子系统)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  odp-val --model train3-20260524-103045-yolo11n-best.pt --data rsod.yaml\n"
            "  odp-val --yaml my_val.yaml\n"
            "  odp-val --no-pre-validate          # 跳过 D4 数据集校验(不推荐)\n"
            "\n"
            "提示: 若 val.yaml 还没生成, 先跑 `odp-gen-config val`."
        ),
    )

    # --- 配置来源 ---
    parser.add_argument("--yaml", dest="yaml_path", type=str, default=None,
                        help="验证配置 YAML 路径(默认走 configs/runtime/val.yaml)")

    # --- 可被 CLI 覆盖的配置字段(透传给 D5 ConfigMerger) ---
    parser.add_argument("--model", type=str,
                        help="待验证的模型权重 —— 通常是 D6 归档的 best.pt, "
                             "如 train3-20260524-103045-yolo11n-best.pt")
    parser.add_argument("--data", type=str,
                        help="数据集 yaml —— D3 立的, 如 rsod.yaml")
    parser.add_argument("--imgsz", type=int, help="验证图像尺寸")
    parser.add_argument("--batch", type=int, help="验证 batch size")
    parser.add_argument("--device", type=str, help="设备, 如 0 / 0,1 / cpu")
    parser.add_argument("--task", dest="task_type", choices=list(SUPPORTED_TASKS),
                        help="验证任务语义: detect 或 segment")
    parser.add_argument("--split", type=str,
                        help="用哪个划分验证: val / test / train")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")

    # --- 行为开关(对应 evaluate() 的 keyword-only 参数) ---
    # 注: 没有 --no-archive —— 评估不归档, 没有归档动作可跳过.
    parser.add_argument("--no-pre-validate", dest="pre_validate",
                        action="store_false",
                        help="跳过验证前的 D4 数据集校验(不推荐)")
    parser.add_argument("--no-rename-log", dest="rename_log",
                        action="store_false",
                        help="不把日志文件名改成 <save_dir>_<ts>_<model>.log 形式")
    parser.set_defaults(pre_validate=True, rename_log=True)

    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()

        # ── 全程序唯一装 logging handler 的地方 ──
        # log_type="val" → 日志落 logging/val/, 跟训练日志 logging/train/ 物理分开
        setup_logging(
            base_path=LOGGING_DIR,
            log_type="val",
            model_name=args.model or None,
        )

        # 组 cli_args 字典 —— 只挑"可覆盖配置字段", 不含 yaml_path / 开关.
        # None 值留着没关系: D5 ConfigMerger 跳过 None.
        config_keys = ("model", "data", "imgsz", "batch", "device", "task_type", "split", "conf", "iou")
        cli_args = {k: getattr(args, k) for k in config_keys}

        service = ValService()
        result = service.evaluate(
            yaml_path=args.yaml_path,
            cli_args=cli_args,
            pre_validate=args.pre_validate,
            rename_log=args.rename_log,
        )

        # ValResult.success → 退出码. service 永不抛, 这里不用 try.
        if result.success:
            print(f"✅ 验证完成, 输出目录: {result.output_dir}")
            if result.metrics:
                fitness = result.metrics.get("fitness")
                if fitness is not None:
                    print(f"   fitness: {fitness:.4f}")
            return 0

        print(f"❌ 验证失败: {result.error}", file=sys.stderr)
        if result.log_path:
            print(f"   详见日志: {result.log_path}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  用户中断验证.", file=sys.stderr)
        return 130
    except Exception:
        logger.exception("验证命令发生未预期异常。")
        return 3


if __name__ == "__main__":
    sys.exit(main())
