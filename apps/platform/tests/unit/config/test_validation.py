from __future__ import annotations

import pytest

from odp_platform.common.constants import TASK_DETECT
from odp_platform.runtime_config import TrainConfig
from odp_platform.runtime_config.merger_core import merge_sources
from odp_platform.runtime_config.validator import ConfigBuildError, validate_config


def _trace_for(values: dict[str, object]):
    _, trace = merge_sources(config_cls=TrainConfig, ordered_sources=[("cli", values)])
    return trace


def test_validate_config_rejects_invalid_task_type() -> None:
    values = TrainConfig().model_dump()
    values["task_type"] = "classification"
    trace = _trace_for({"task_type": "classification"})

    with pytest.raises(ConfigBuildError) as exc_info:
        validate_config(TrainConfig, values, trace)

    assert "experiment_name" in str(exc_info.value)


def test_validate_config_rejects_contradictory_fields() -> None:
    values = TrainConfig(save=False, save_period=5).model_dump()
    trace = _trace_for({"save": False, "save_period": 5})

    with pytest.raises(ConfigBuildError):
        validate_config(TrainConfig, values, trace)


def test_validate_config_warns_for_redundant_but_harmless_combo() -> None:
    values = TrainConfig(task_type=TASK_DETECT, cache=False, batch=0).model_dump()
    trace = _trace_for({"cache": False, "batch": 0})

    _config, warnings = validate_config(TrainConfig, values, trace)
    assert warnings
    assert warnings[0].field_name == "batch"
