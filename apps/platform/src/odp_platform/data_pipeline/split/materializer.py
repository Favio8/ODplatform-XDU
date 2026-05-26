"""Copy split datasets into their final directory layout."""

from __future__ import annotations

import shutil
import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from odp_platform.data_pipeline.split.manifest import PreparedSample


def _windows_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _copy2(src: Path, dst: Path) -> None:
    shutil.copy2(_windows_long_path(src), _windows_long_path(dst))


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(_windows_long_path(path))
    path.mkdir(parents=True, exist_ok=True)


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
        _reset_dir(image_dir)
        _reset_dir(label_dir)

        materialized_samples: list[PreparedSample] = []
        for sample in samples:
            target_image = image_dir / f"{sample.stem}{sample.image_path.suffix}"
            target_label = label_dir / sample.label_path.name
            _copy2(sample.image_path, target_image)
            _copy2(sample.label_path, target_label)
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
