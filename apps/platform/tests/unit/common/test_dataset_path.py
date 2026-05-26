from __future__ import annotations

from pathlib import Path

from odp_platform.common.dataset_path import resolve_dataset_path
from odp_platform.common.paths import DATASET_CONFIGS_DIR


def test_resolve_dataset_path_supports_dataset_name(monkeypatch) -> None:
    dataset_file = DATASET_CONFIGS_DIR / "demo_dataset.yaml"
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == str(dataset_file))

    resolved = resolve_dataset_path("demo_dataset")

    assert resolved == dataset_file
