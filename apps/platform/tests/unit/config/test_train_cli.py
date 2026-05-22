from __future__ import annotations

import json
from pathlib import Path

from odp_platform.cli.train import main
from odp_platform.common.logging_utils import reset_logging


def test_train_cli_builds_config_and_writes_provenance_log(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(
        "task: detect\n"
        "epochs: 100\n"
        "batch: 16\n"
        "workers: 8\n",
        encoding="utf-8",
    )

    try:
        exit_code = main(
            [
                "--config",
                str(yaml_path),
                "--epochs",
                "10",
                "--model",
                "yolov8n.pt",
                "--log-level",
                "INFO",
            ]
        )
        assert exit_code == 0
    finally:
        reset_logging()

    log_files = sorted(Path("apps/platform/logging/train").glob("*.jsonl"))
    assert log_files
    messages = [
        json.loads(line)["message"]
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]
    assert any("epochs: 100(DEFAULT) <- 100(YAML) <- 10(CLI)" in message for message in messages)
    assert any("训练配置预检完成" in message for message in messages)
