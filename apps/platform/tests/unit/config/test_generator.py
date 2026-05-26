from __future__ import annotations

from pathlib import Path

from odp_platform.common.constants import RUNTIME_TASK_TRAIN
from odp_platform.config.generator import generate_template
from odp_platform.config.loaders import load_yaml_config


def test_generate_template_writes_yaml_that_can_be_loaded(tmp_path: Path) -> None:
    target = tmp_path / "train.yaml"
    generated = generate_template(RUNTIME_TASK_TRAIN, target)

    assert generated == target
    assert "odp-gen-config" in generated.read_text(encoding="utf-8")
    payload = load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path=generated)
    assert payload.values["epochs"] == 100


def test_generate_template_skips_existing_without_force(tmp_path: Path) -> None:
    target = tmp_path / "train.yaml"
    target.write_text("epochs: 1\n", encoding="utf-8")

    generate_template(RUNTIME_TASK_TRAIN, target, force=False)
    assert target.read_text(encoding="utf-8") == "epochs: 1\n"


def test_generate_template_force_creates_backup(tmp_path: Path) -> None:
    target = tmp_path / "train.yaml"
    target.write_text("epochs: 1\n", encoding="utf-8")

    generate_template(RUNTIME_TASK_TRAIN, target, force=True, backup=True)
    backups = list(tmp_path.glob("train.yaml.bak.*"))
    assert backups
