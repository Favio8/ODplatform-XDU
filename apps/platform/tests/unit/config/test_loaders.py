from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from odp_platform.common.constants import RUNTIME_TASK_TRAIN
from odp_platform.runtime_config import ConfigLoadError
from odp_platform.runtime_config.loaders_core import load_cli_config, load_mapping_source, load_yaml_config


def test_load_yaml_config_reads_flat_values(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("epochs: 12\nimgsz: 960\n", encoding="utf-8")

    payload = load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path=yaml_path)
    assert payload.values["epochs"] == 12
    assert payload.values["imgsz"] == 960


def test_load_yaml_config_supports_task_alias_key(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("task: detect\nepochs: 12\n", encoding="utf-8")

    payload = load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path=yaml_path)
    assert payload.values["task_type"] == "detect"


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
    assert "odp-gen-config" in message


def test_load_yaml_config_resolves_runtime_directory_by_task_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr("odp_platform.runtime_config.loaders_core.RUNTIME_CONFIGS_DIR", runtime_dir)
    monkeypatch.setattr(
        "odp_platform.runtime_config.loaders_core.runtime_config_path",
        lambda name: runtime_dir / f"{name}.yaml",
    )
    (runtime_dir / "train.yaml").write_text("epochs: 12\n", encoding="utf-8")

    payload = load_yaml_config(task_kind=RUNTIME_TASK_TRAIN, config_path="train")
    assert payload.values["epochs"] == 12


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
