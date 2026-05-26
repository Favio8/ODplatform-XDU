from __future__ import annotations

import logging
from pathlib import Path

from odp_platform.common.log_rename import rename_log_to_save_dir
from odp_platform.common.logging_utils import ROOT_LOGGER_NAME, reset_logging, setup_logging


def test_rename_log_to_save_dir_rotates_named_root_file(tmp_path: Path) -> None:
    reset_logging()
    logger = setup_logging(base_path=tmp_path, log_type="train", model_name="demo")
    logger.info("before rename")

    renamed = rename_log_to_save_dir(tmp_path / "runs" / "train9", "yolo11n")

    assert renamed is not None
    assert renamed.exists()
    assert "train9" in renamed.name

    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    file_handlers = [handler for handler in root_logger.handlers if isinstance(handler, logging.FileHandler)]
    assert file_handlers
    assert Path(file_handlers[0].baseFilename) == renamed
    reset_logging()
