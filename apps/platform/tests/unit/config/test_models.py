from __future__ import annotations

from odp_platform.config import InferConfig, TrainConfig, ValConfig
from odp_platform.config.merger import merge_sources


def test_configs_have_metadata_and_defaults() -> None:
    train = TrainConfig()
    val = ValConfig()
    infer = InferConfig()

    assert train.epochs == 100
    assert val.batch == 16
    assert infer.conf == 0.25
    assert "task_type" in TrainConfig.field_specs()
    assert "epochs" in TrainConfig.field_specs()


def test_to_ultralytics_kwargs_filters_internal_and_empty_values() -> None:
    config = TrainConfig(experiment_name="demo", name="", data="dataset.yaml", model="model.pt", task_type="detect")
    kwargs = config.to_ultralytics_kwargs()

    assert "experiment_name" not in kwargs
    assert "task_kind" not in kwargs
    assert "name" not in kwargs
    assert "task_type" not in kwargs
    assert kwargs["task"] == "detect"
    assert kwargs["data"] == "dataset.yaml"
    assert kwargs["model"] == "model.pt"


def test_snapshot_roundtrip_restores_config() -> None:
    config = InferConfig(model="weights.pt", source="demo.jpg")
    snapshot = config.to_snapshot()
    restored = InferConfig.from_snapshot(snapshot)

    assert restored == config


def test_trace_dict_and_human_views_are_stable() -> None:
    _merged, trace = merge_sources(
        config_cls=TrainConfig,
        ordered_sources=[
            ("yaml", {"epochs": 12}),
            ("cli", {"epochs": 24}),
        ],
    )

    field_trace = trace.get("epochs")
    assert field_trace.final_source_label == "CLI"
    assert field_trace.to_effective_line() == "epochs: 24  (来源: CLI)"
    assert field_trace.to_human_readable() == "epochs: 100(DEFAULT) <- 12(YAML) <- 24(CLI)"
    assert field_trace.to_dict()["history"][0]["source_label"] == "DEFAULT"
