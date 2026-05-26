from __future__ import annotations

from pathlib import Path

from odp_platform.common.model_path import resolve_model_path


def test_resolve_model_path_prefers_search_dirs(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    model_file = weights_dir / "demo.pt"
    model_file.write_text("x", encoding="utf-8")

    resolved = resolve_model_path("demo.pt", search_dirs=[weights_dir])

    assert resolved == model_file
