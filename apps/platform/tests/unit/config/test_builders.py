from __future__ import annotations

from pathlib import Path

from odp_platform.common.constants import RUNTIME_TASK_INFER, RUNTIME_TASK_TRAIN, RUNTIME_TASK_VAL
from odp_platform.runtime_config import (
    build_infer_config,
    build_config,
    build_train_config,
    build_val_config,
    generate_template,
    load_mapping_source,
    preview_train_config,
)


def test_build_train_config_prefers_cli_over_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("epochs: 12\nimgsz: 960\n", encoding="utf-8")

    config, trace = build_train_config(
        yaml_path=str(yaml_path),
        cli_args={"epochs": 24},
    )

    assert config.epochs == 24
    assert config.imgsz == 960
    assert trace.get("epochs").final_source == "cli"


def test_preview_train_config_returns_merged_dict_without_validation(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("save: false\nsave_period: 10\n", encoding="utf-8")

    merged, trace = preview_train_config(yaml_path=str(yaml_path))
    assert merged["save"] is False
    assert merged["save_period"] == 10
    assert trace.get("save_period").final_source.startswith("yaml:")


def test_generated_templates_build_successfully(tmp_path: Path) -> None:
    train_yaml = generate_template(RUNTIME_TASK_TRAIN, tmp_path / "train.yaml", force=True)
    val_yaml = generate_template(RUNTIME_TASK_VAL, tmp_path / "val.yaml", force=True)
    infer_yaml = generate_template(RUNTIME_TASK_INFER, tmp_path / "infer.yaml", force=True)

    train_config, _ = build_train_config(yaml_path=str(train_yaml))
    val_config, _ = build_val_config(yaml_path=str(val_yaml))
    infer_config, _ = build_infer_config(yaml_path=str(infer_yaml))

    assert train_config.epochs == 100
    assert val_config.batch == 16
    assert infer_config.conf == 0.25


def test_build_config_supports_three_source_merge_and_trace(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("epochs: 20\nbatch: 16\n", encoding="utf-8")
    profile = load_mapping_source(
        task_kind=RUNTIME_TASK_TRAIN,
        source_name="profile",
        values={"epochs": 30, "batch": 12},
    )

    config, trace, warnings = build_config(
        task_kind=RUNTIME_TASK_TRAIN,
        yaml_path=str(yaml_path),
        extra_sources=[profile],
        cli_args={"epochs": 40},
    )

    assert warnings == []
    assert config.epochs == 40
    assert config.batch == 12
    assert trace.get("epochs").final_source_label == "CLI"
    assert trace.get("epochs").to_override_chain() == "epochs: 100(DEFAULT) <- 20(YAML) <- 30(PROFILE) <- 40(CLI)"


def test_build_train_config_accepts_teacher_train_yaml() -> None:
    yaml_path = Path(
        r"D:\AAA_Favio_2026\AI_exploring\personal\ODplatform\老师提供的文档\train.yaml"
    )

    config, trace = build_train_config(yaml_path=str(yaml_path))

    assert config.task_type == "detect"
    assert config.workers == 8
    assert config.optimizer == "auto"
    assert trace.get("task_type").final_source_label == "YAML"
