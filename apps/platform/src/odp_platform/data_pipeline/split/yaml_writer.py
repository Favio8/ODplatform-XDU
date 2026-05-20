"""Write Ultralytics-compatible dataset yaml files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

import yaml

from odp_platform.common.constants import (
    ODP_META_KEY,
    SCHEMA_VERSION,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from odp_platform.data_pipeline.split.manifest import ConversionManifest, PreparedSample


def write_dataset_yaml(
    *,
    yaml_path: Path,
    data_root: Path,
    manifest: ConversionManifest,
    split_map: Mapping[str, list[PreparedSample]],
    train_rate: float,
    val_rate: float,
    test_rate: float,
    random_state: int,
) -> Path:
    """Persist one dataset config for Ultralytics training."""

    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        SPLIT_TRAIN: len(split_map.get(SPLIT_TRAIN, [])),
        SPLIT_VAL: len(split_map.get(SPLIT_VAL, [])),
        SPLIT_TEST: len(split_map.get(SPLIT_TEST, [])),
    }

    payload = {
        "path": str(Path(data_root).resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(manifest.classes),
        "names": {index: class_name for index, class_name in enumerate(manifest.classes)},
        ODP_META_KEY: {
            "dataset": manifest.dataset_name,
            "source_format": manifest.source_format,
            "task": manifest.task,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "split": {
                "train_rate": train_rate,
                "val_rate": val_rate,
                "test_rate": test_rate,
                "random_state": random_state,
                "counts": {
                    SPLIT_TRAIN: counts[SPLIT_TRAIN],
                    SPLIT_VAL: counts[SPLIT_VAL],
                    SPLIT_TEST: counts[SPLIT_TEST],
                    "total": sum(counts.values()),
                },
            },
            "schema_version": SCHEMA_VERSION,
        },
    }

    yaml_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


__all__ = [
    "write_dataset_yaml",
]
