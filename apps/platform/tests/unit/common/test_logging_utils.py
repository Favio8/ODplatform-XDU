from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from odp_platform.common.logging_utils import ROOT_LOGGER_NAME, get_logger, reset_logging, setup_logging


def _unique_logger_name() -> str:
    return f"{ROOT_LOGGER_NAME}.tests.{uuid4().hex}"


@pytest.fixture
def logger_name() -> str:
    name = _unique_logger_name()
    yield name
    reset_logging(name)


def _get_log_file(logger: logging.Logger) -> Path:
    file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    assert len(file_handlers) == 1
    return Path(file_handlers[0].baseFilename)


def _read_log_payloads(log_file: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_setup_logging_creates_json_log_file(tmp_path: Path, logger_name: str) -> None:
    logger = setup_logging(base_path=tmp_path, log_type="train", logger_name=logger_name)

    log_file = _get_log_file(logger)

    assert log_file.parent == tmp_path / "train"
    assert log_file.suffix == ".jsonl"
    assert log_file.exists()


def test_setup_logging_is_idempotent_for_same_logger(tmp_path: Path, logger_name: str) -> None:
    first = setup_logging(base_path=tmp_path, logger_name=logger_name)
    second = setup_logging(base_path=tmp_path / "other", logger_name=logger_name)

    assert first is second
    assert len(first.handlers) == 2
    assert _get_log_file(first).parent == tmp_path / "general"


def test_environment_log_level_is_used_and_explicit_value_wins(
    tmp_path: Path,
    logger_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ODP_LOG_LEVEL", "ERROR")
    logger = setup_logging(base_path=tmp_path, logger_name=logger_name)

    assert logger.level == logging.ERROR

    reset_logging(logger_name)
    monkeypatch.setenv("ODP_LOG_LEVEL", "ERROR")
    logger = setup_logging(base_path=tmp_path, logger_name=logger_name, log_level="DEBUG")

    assert logger.level == logging.DEBUG


def test_child_logger_messages_propagate_to_parent_handlers(tmp_path: Path, logger_name: str) -> None:
    logger = setup_logging(base_path=tmp_path, logger_name=logger_name, log_type="infer")
    child_logger = logging.getLogger(f"{logger_name}.worker")

    child_logger.info("child message")

    payloads = _read_log_payloads(_get_log_file(logger))

    assert any(payload["message"] == "child message" for payload in payloads)
    assert any(payload["logger"] == f"{logger_name}.worker" for payload in payloads)


def test_json_log_records_include_required_fields(tmp_path: Path, logger_name: str) -> None:
    logger = setup_logging(
        base_path=tmp_path,
        logger_name=logger_name,
        log_type="validate",
        model_name="yolo11n.pt",
        temp_log=True,
    )
    logger.warning("dataset warning")

    payload = _read_log_payloads(_get_log_file(logger))[-1]

    assert payload["message"] == "dataset warning"
    assert payload["log_type"] == "validate"
    assert payload["model_name"] == "yolo11n.pt"
    assert payload["temp_log"] is True
    for field in (
        "timestamp",
        "level",
        "logger",
        "message",
        "module",
        "filename",
        "lineno",
        "funcName",
        "process",
        "thread",
        "log_type",
        "model_name",
        "temp_log",
    ):
        assert field in payload


def test_reset_logging_removes_handlers(tmp_path: Path, logger_name: str) -> None:
    logger = setup_logging(base_path=tmp_path, logger_name=logger_name)

    reset_logging(logger_name)

    assert logger.handlers == []
    assert logger.propagate is True


def test_temp_log_and_model_name_affect_file_name(tmp_path: Path, logger_name: str) -> None:
    logger = setup_logging(
        base_path=tmp_path,
        logger_name=logger_name,
        log_type="train_run",
        model_name="yolo11n/best.pt",
        temp_log=True,
    )

    log_file = _get_log_file(logger)

    assert log_file.name.startswith("temp_")
    assert "yolo11n_best_pt" in log_file.name


def test_get_logger_defaults_to_root_logger() -> None:
    assert get_logger() is logging.getLogger(ROOT_LOGGER_NAME)
