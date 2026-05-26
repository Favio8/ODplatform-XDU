from __future__ import annotations

from pathlib import Path

from odp_platform.data_pipeline.split.manifest import PreparedSample
from odp_platform.data_pipeline.split.materializer import materialize_splits


def test_materialize_splits_resets_existing_targets(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sample_image = src_dir / "sample.jpg"
    sample_label = src_dir / "sample.txt"
    sample_image.write_text("image", encoding="utf-8")
    sample_label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    image_dir = tmp_path / "data" / "train" / "images"
    label_dir = tmp_path / "data" / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "stale.jpg").write_text("old", encoding="utf-8")
    (label_dir / "stale.txt").write_text("old", encoding="utf-8")

    split_map = {
        "train": [
            PreparedSample(
                stem="sample",
                image_path=sample_image,
                label_path=sample_label,
                class_names=("target",),
            )
        ]
    }

    result = materialize_splits(
        split_map,
        image_dir_by_split={"train": image_dir},
        label_dir_by_split={"train": label_dir},
    )

    assert not (image_dir / "stale.jpg").exists()
    assert not (label_dir / "stale.txt").exists()
    assert (image_dir / "sample.jpg").exists()
    assert (label_dir / "sample.txt").exists()
    assert result["train"][0].image_path == image_dir / "sample.jpg"
