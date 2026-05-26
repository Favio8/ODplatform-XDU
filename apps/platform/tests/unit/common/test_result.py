from __future__ import annotations

from pathlib import Path

import numpy as np

from odp_platform.common.result import TrainMetrics


class _FakeResults:
    task = ""
    save_dir = "runs/detect_train/train1"
    speed = {"preprocess": 0.1, "inference": 0.2, "loss": 0.0, "postprocess": 0.3}
    results_dict = {"metrics/precision(B)": 0.9, "metrics/mAP50(B)": 0.8}
    fitness = 0.7
    names = {0: "ship"}
    maps = np.array([0.55])


def test_train_metrics_uses_task_fallback() -> None:
    metrics = TrainMetrics.from_yolo_results(_FakeResults(), task_fallback="detect")

    assert metrics.task == "detect"
    assert metrics.save_dir == Path("runs/detect_train/train1")
    assert metrics.class_map_50_95["ship"] == 0.55
