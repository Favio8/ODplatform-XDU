from __future__ import annotations

import logging

import pytest

from odp_platform.common.performance_utils import format_duration, time_it


def test_format_duration_uses_human_friendly_units() -> None:
    assert format_duration(0.0005) == "500.00 us"
    assert format_duration(0.25) == "250.00 ms"
    assert format_duration(2.5) == "2.50 s"
    assert format_duration(65.0) == "1 min 5.00 s"


def test_format_duration_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        format_duration(-0.1)


def test_time_it_logs_single_execution(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("odp_platform.tests.performance.single")

    @time_it(name="single-run", logger_instance=logger)
    def sample() -> str:
        return "done"

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = sample()

    assert result == "done"
    assert "性能报告: single-run 执行时间:" in caplog.text


def test_time_it_logs_total_and_average_for_multiple_iterations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("odp_platform.tests.performance.multi")

    @time_it(iterations=2, name="multi-run", logger_instance=logger)
    def sample() -> int:
        return 7

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = sample()

    assert result == 7
    assert "性能报告: multi-run 执行了 2 次" in caplog.text
    assert "总耗时:" in caplog.text
    assert "平均耗时:" in caplog.text


def test_time_it_requires_positive_iterations() -> None:
    with pytest.raises(ValueError):
        time_it(iterations=0)
