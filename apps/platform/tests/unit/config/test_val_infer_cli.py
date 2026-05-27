from __future__ import annotations

import json
from pathlib import Path

from odp_platform.cli.infer import main as infer_main
from odp_platform.cli.val import main as val_main
from odp_platform.common.logging_utils import reset_logging


def test_val_cli_builds_config_and_writes_log(tmp_path: Path) -> None:
    yaml_path = tmp_path / "val.yaml"
    yaml_path.write_text(
        "task: detect\n"
        "batch: 16\n"
        "split: val\n",
        encoding="utf-8",
    )

    try:
        exit_code = val_main(
            [
                "--config",
                str(yaml_path),
                "--batch",
                "8",
                "--model",
                "yolov8n.pt",
                "--log-level",
                "INFO",
            ]
        )
        assert exit_code == 0
    finally:
        reset_logging()

    log_files = sorted(Path("apps/platform/logging/val").glob("*.jsonl"))
    assert log_files
    messages = [
        json.loads(line)["message"]
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]
    assert any("验证配置预检完成" in message for message in messages)


def test_infer_cli_builds_config_and_writes_log(tmp_path: Path) -> None:
    yaml_path = tmp_path / "infer.yaml"
    yaml_path.write_text(
        "task: detect\n"
        "conf: 0.25\n",
        encoding="utf-8",
    )

    try:
        exit_code = infer_main(
            [
                "--config",
                str(yaml_path),
                "--preflight-only",
                "--source",
                "demo.jpg",
                "--save-conf",
                "--save-txt",
                "--model",
                "yolov8n.pt",
                "--log-level",
                "INFO",
            ]
        )
        assert exit_code == 0
    finally:
        reset_logging()

    log_files = sorted(Path("apps/platform/logging/infer").glob("*.jsonl"))
    assert log_files
    messages = [
        json.loads(line)["message"]
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]
    assert any("推理配置预检完成" in message for message in messages)
