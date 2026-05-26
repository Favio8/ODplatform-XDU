from __future__ import annotations

from pathlib import Path

from odp_platform.training.archive import archive_checkpoints


def test_archive_checkpoints_copies_best_and_last(tmp_path: Path) -> None:
    train_dir = tmp_path / "runs" / "train1"
    weights_dir = train_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_text("best", encoding="utf-8")
    (weights_dir / "last.pt").write_text("last", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"

    archived = archive_checkpoints(
        train_dir=train_dir,
        model_filename="yolo11n.pt",
        checkpoint_dir=checkpoint_dir,
    )

    assert "best" in archived
    assert "last" in archived
    assert archived["best"].exists()
    assert archived["last"].exists()
