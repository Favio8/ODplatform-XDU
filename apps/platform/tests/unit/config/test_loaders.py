from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from odp_platform.common.constants import RUNTIME_TASK_TRAIN
from odp_platform.config import ConfigLoadError
from odp_platform.config.loaders import load_cli_config, load_mapping_source, load_yaml_config


def test_load_yaml_config_reads_flat_values(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("epochs: 12\nimgsz: 960\n", encoding="utf-8")

    payload = load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path=yaml_path)
    assert payload.values["epochs"] == 12
    assert payload.values["imgsz"] == 960


def test_load_yaml_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError):
        load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path=yaml_path)


def test_load_yaml_config_missing_file_contains_template_hint(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigLoadError) as exc_info:
        load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path=missing)

    message = str(exc_info.value)
    assert "Expected path" in message
    assert "odp-generate-config" in message


def test_load_cli_config_keeps_falsey_values() -> None:
    namespace = argparse.Namespace(batch=0, save=False, project="", _internal="x", help=None)
    payload = load_cli_config(
        task_kind=RUNTIME_TASK_TRAIN,
        namespace=namespace,
        ignored_keys={"project"},
    )

    assert payload.values["batch"] == 0
    assert payload.values["save"] is False
    assert "project" not in payload.values


def test_load_mapping_source_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigLoadError):
        load_mapping_source(
            task_kind=RUNTIME_TASK_TRAIN,
            source_name="custom",
            values={"not_a_field": 1},
        )
