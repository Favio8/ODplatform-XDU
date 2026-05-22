from __future__ import annotations

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
