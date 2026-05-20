"""Copy split datasets into their final directory layout."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from odp_platform.data_pipeline.split.manifest import PreparedSample


def materialize_splits(
    split_map: Mapping[str, list[PreparedSample]],
    *,
    image_dir_by_split: Mapping[str, Path],
    label_dir_by_split: Mapping[str, Path],
) -> dict[str, list[PreparedSample]]:
    """Copy images and labels into caller-provided split directories."""

    materialized: dict[str, list[PreparedSample]] = {}

    for split_name, samples in split_map.items():
        image_dir = Path(image_dir_by_split[split_name])
        label_dir = Path(label_dir_by_split[split_name])
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        materialized_samples: list[PreparedSample] = []
        for sample in samples:
            target_image = image_dir / sample.image_path.name
            target_label = label_dir / sample.label_path.name
            shutil.copy2(sample.image_path, target_image)
            shutil.copy2(sample.label_path, target_label)
            materialized_samples.append(
                replace(
                    sample,
                    image_path=target_image,
                    label_path=target_label,
                )
            )

        materialized[split_name] = materialized_samples

    return materialized


__all__ = [
    "materialize_splits",
]
