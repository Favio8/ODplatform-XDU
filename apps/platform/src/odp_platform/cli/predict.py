"""Beautified single-image room segmentation CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from odp_platform.inference.room_segmentation import (
    RoomSegmentationOptions,
    run_room_segmentation,
    write_room_segmentation_outputs,
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-predict",
        description="YOLO 房间分割推理 - 面积占比 + JSON 输出 + 美化图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  odp-predict --model best.pt --source room.jpg
  odp-predict --model best.pt --source room.jpg --device cpu
  odp-predict --model best.pt --source room.jpg --output-json result.json --output-img pretty.jpg
        """,
    )
    parser.add_argument("--model", required=True, help="训练好的 YOLO 模型路径 (.pt)")
    parser.add_argument("--source", required=True, help="待推理的图片路径")
    parser.add_argument("--output-json", default="inference_result.json", help="输出 JSON 文件路径")
    parser.add_argument("--output-img", default="output_seg_beautiful.jpg", help="输出美化图片路径")
    parser.add_argument("--device", default="0", help="推理设备 (0 / cpu / 0,1)")
    parser.add_argument("--alpha", type=float, default=0.4, help="掩膜透明度 (0-1)")
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.02,
        help="面积占比低于此值的房间在图片中不标注 (0~1)",
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成美化图片")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("odp_predict")
    logger.info("启动 odp-predict")

    try:
        from ultralytics import YOLO

        logger.info("加载模型: %s", args.model)
        model = YOLO(args.model)
        logger.info("开始推理: %s", args.source)
        result = run_room_segmentation(
            model=model,
            source=args.source,
            source_name=args.source,
            options=RoomSegmentationOptions(
                device=args.device,
                alpha=args.alpha,
                label_threshold=args.label_threshold,
                render=not args.no_plot,
            ),
        )
        write_room_segmentation_outputs(
            result,
            output_json=args.output_json,
            output_img=None if args.no_plot else args.output_img,
        )
        logger.info("JSON 结果已保存至 %s", Path(args.output_json))
        if not args.no_plot:
            logger.info("美化图片已保存至 %s", Path(args.output_img))
        logger.info("---------- 房间面积占比 ----------")
        for index, detection in enumerate(result.payload["detections"], 1):
            logger.info(
                "Room %d (置信度: %.2f) - 面积: %.0f 像素, 占比: %.1f%%",
                index,
                detection["confidence"],
                detection["area"],
                detection["area_ratio"] * 100,
            )
        logger.info("----------------------------------")
        logger.info("推理流程结束。")
        return 0
    except Exception as exc:
        logger.exception("运行异常: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
