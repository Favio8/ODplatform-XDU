import importlib
from pathlib import Path

import odp_platform
from odp_platform.common import paths


def test_package_exposes_version() -> None:
    assert odp_platform.__version__ == "0.1.0"


def test_cli_modules_are_importable() -> None:
    for module_name in ("trans", "validate", "init_project", "reset_project", "train", "val", "infer"):
        module = importlib.import_module(f"odp_platform.cli.{module_name}")
        assert callable(module.main)


def test_workspace_paths_are_resolved_from_marker() -> None:
    assert paths.ROOT_DIR.name == "ODplatform"
    assert (paths.ROOT_DIR / paths.WORKSPACE_MARKER).exists()
    assert paths.find_workspace_root(paths.__file__) == paths.ROOT_DIR


def test_common_workspace_path_constants() -> None:
    assert paths.DATA_DIR == paths.workspace_path("data")
    assert paths.APP_DIR == paths.app_path()
    assert paths.LOGGING_DIR == paths.app_path("logging")
    assert paths.META_DIR == paths.workspace_path(".odp-meta")
    assert paths.META_LOGGING_DIR == paths.META_DIR / "logs"
    assert paths.MODELS_DIR == paths.workspace_path("models")
    assert paths.CONFIGS_DIR == paths.PLATFORM_CONFIG_DIR
    assert paths.PLATFORM_CONFIG_DIR == paths.app_path("configs")
    assert paths.data_path("RSOD", "raw") == paths.DATA_DIR / "RSOD" / "raw"
    assert paths.RAW_DATA_DIR == paths.data_path("RSOD", "raw")
    assert paths.TRAIN_DIR == paths.workspace_path("data", "train")
    assert paths.VAL_DIR == paths.workspace_path("data", "val")
    assert paths.TEST_DIR == paths.workspace_path("data", "test")
    assert paths.model_path("pretrained") == paths.MODELS_DIR / "pretrained"
    assert paths.run_path("demo") == paths.RUNS_DIR / "demo"
    assert paths.is_within_workspace(Path(paths.ROOT_DIR) / "apps")
    assert paths.RAW_DATA_DIR in paths.get_dirs_to_initialize()
    assert paths.META_LOGGING_DIR in paths.get_dirs_to_initialize()
    assert paths.RUNS_DIR in paths.get_dirs_to_reset()
