from __future__ import annotations

from pathlib import Path

from odp_platform.runtime_config import (
    BaseConfig,
    CLILoader,
    ConfigGenerator,
    ConfigMerger,
    ConfigSource,
    YAMLLoader,
    load_all_sources,
)
from odp_platform.runtime_config.train import YOLOTrainConfig


def test_runtime_config_exports_teacher_compatible_symbols(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("epochs: 12\n", encoding="utf-8")

    yaml_loader = YAMLLoader()
    cli_loader = CLILoader()

    assert BaseConfig is not None
    assert YOLOTrainConfig is not None
    assert yaml_loader.load(yaml_path)["epochs"] == 12
    assert cli_loader.load({"epochs": 24})["epochs"] == 24


def test_runtime_config_load_all_sources_returns_yaml_and_cli(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("epochs: 12\n", encoding="utf-8")

    sources = load_all_sources(
        yaml_path=yaml_path,
        cli_args={"epochs": 24},
    )

    assert sources["yaml"]["epochs"] == 12
    assert sources["cli"]["epochs"] == 24


def test_runtime_config_merger_supports_merge_and_linked_metadata() -> None:
    merger = ConfigMerger(track_sources=True)
    config = merger.merge(
        YOLOTrainConfig,
        sources=[
            (ConfigSource.YAML, {"epochs": 24}),
            (ConfigSource.CLI, {"epochs": 12}),
        ],
    )

    meta = merger.get_metadata("epochs")
    assert config.epochs == 12
    assert meta is not None
    assert meta.source_label == "CLI"
    assert meta.overridden_from is not None
    assert meta.overridden_from.source_label == "YAML"
    assert meta.chain_str() == "12(CLI) ← 24(YAML) ← 100(DEFAULT)"
    assert hasattr(merger, "preview")
    assert hasattr(merger, "get_source_report")
    assert hasattr(merger, "to_audit_log")


def test_runtime_config_generator_class_generates_template(tmp_path: Path) -> None:
    target = tmp_path / "train.yaml"
    generated = ConfigGenerator().generate(
        YOLOTrainConfig,
        target,
        overwrite=False,
    )

    assert generated is True
    assert target.exists()
    assert "YOLO 训练配置" in target.read_text(encoding="utf-8")


def test_runtime_base_config_exposes_teacher_style_field_reflection() -> None:
    config = YOLOTrainConfig()

    groups = config.get_field_groups()
    metadata = config.get_field_metadata("imgsz")

    assert "input" in groups
    assert "imgsz" in groups["input"]
    assert metadata["default"] == 640
    assert metadata["group"] == "input"
    assert "description" in metadata
