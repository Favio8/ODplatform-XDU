from __future__ import annotations

from odp_platform.config import InferConfig, TrainConfig, ValConfig


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
    config = TrainConfig(experiment_name="demo", name="", data="dataset.yaml", model="model.pt")
    kwargs = config.to_ultralytics_kwargs()

    assert "experiment_name" not in kwargs
    assert "task_kind" not in kwargs
    assert "name" not in kwargs
    assert kwargs["data"] == "dataset.yaml"
    assert kwargs["model"] == "model.pt"


def test_snapshot_roundtrip_restores_config() -> None:
    config = InferConfig(model="weights.pt", source="demo.jpg")
    snapshot = config.to_snapshot()
    restored = InferConfig.from_snapshot(snapshot)

    assert restored == config
