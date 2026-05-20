import importlib
from pathlib import Path

import odp_platform
from odp_platform.common import constants
from odp_platform.common import paths
from odp_platform.data_pipeline import ConvertOptions


def test_package_exposes_version() -> None:
    assert odp_platform.__version__ == "0.1.0"


def test_cli_modules_are_importable() -> None:
    for module_name in (
        "trans",
        "validate",
        "init_project",
        "reset_project",
        "transform_data",
        "train",
        "val",
        "infer",
    ):
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
    assert paths.RAW_DATASETS_DIR == paths.data_path("raw")
    assert paths.YOLO_DATASETS_DIR == paths.data_path("yolo")
    assert paths.DATASET_CONFIGS_DIR == paths.app_path("configs", "datasets")
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
    assert paths.DATASET_CONFIGS_DIR in paths.get_dirs_to_initialize()
    assert paths.RUNS_DIR in paths.get_dirs_to_reset()


def test_data_pipeline_constants() -> None:
    assert constants.FORMAT_PASCAL_VOC == "pascal_voc"
    assert constants.FORMAT_COCO == "coco"
    assert constants.FORMAT_YOLO == "yolo"
    assert constants.SUPPORTED_TASKS == ("detect", "segment")
    assert constants.SUPPORTED_SPLITS == ("train", "val", "test")
    assert constants.SCHEMA_VERSION == 1


def test_convert_options_can_be_initialized_without_registry() -> None:
    options = ConvertOptions(
        dataset_name="rsod",
        source_format=constants.FORMAT_PASCAL_VOC,
    )
    assert options.dataset_name == "rsod"
    assert options.source_format == constants.FORMAT_PASCAL_VOC
    assert options.task == constants.TASK_DETECT
    assert options.classes is None
