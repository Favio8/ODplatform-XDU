from __future__ import annotations

from odp_platform.config import load_mapping_source
from odp_platform.config import TrainConfig
from odp_platform.config.merger import merge_sources


def test_merge_sources_applies_priority_and_trace() -> None:
    merged, trace = merge_sources(
        config_cls=TrainConfig,
        ordered_sources=[
            ("yaml", {"epochs": 12, "save": True}),
            ("cli", {"epochs": 99, "save": False}),
        ],
    )

    assert merged["epochs"] == 99
    assert merged["save"] is False
    assert trace.get("epochs").final_source == "cli"
    assert trace.get("epochs").history[0].source_name == "defaults"
    assert trace.get("epochs").history[-1].source_name == "cli"


def test_merge_sources_supports_three_source_override_chain() -> None:
    profile = load_mapping_source(
        task_kind="train",
        source_name="profile",
        values={"epochs": 36, "batch": 8},
    )

    merged, trace = merge_sources(
        config_cls=TrainConfig,
        ordered_sources=[
            ("yaml", {"epochs": 24, "batch": 16}),
            (profile.source_name, profile.values),
            ("cli", {"epochs": 12}),
        ],
    )

    assert merged["epochs"] == 12
    assert merged["batch"] == 8
    assert trace.get("epochs").to_override_chain() == "epochs: 100(DEFAULT) <- 24(YAML) <- 36(PROFILE) <- 12(CLI)"
    assert trace.get("epochs").to_effective_line() == "epochs: 12  (来源: CLI)"
