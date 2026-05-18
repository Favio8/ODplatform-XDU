import importlib

import odp_platform


def test_package_exposes_version() -> None:
    assert odp_platform.__version__ == "0.1.0"


def test_cli_modules_are_importable() -> None:
    for module_name in ("trans", "validate", "train", "val", "infer"):
        module = importlib.import_module(f"odp_platform.cli.{module_name}")
        assert callable(module.main)
