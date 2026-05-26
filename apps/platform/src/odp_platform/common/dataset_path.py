"""Dataset-yaml resolution helpers shared by training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from odp_platform.common.paths import DATASET_CONFIGS_DIR


logger = logging.getLogger(__name__)


def _candidate_names(data_path: Path) -> list[str]:
    names = [data_path.name]
    if data_path.suffix != ".yaml":
        names.append(f"{data_path.name}.yaml")
    return names


def resolve_dataset_path(data: str | Path) -> Path:
    """Resolve a dataset reference to a concrete dataset yaml path."""

    data_path = Path(data)

    if data_path.is_absolute():
        return data_path

    if data_path.exists():
        return data_path.resolve()

    for candidate_name in _candidate_names(data_path):
        candidate = DATASET_CONFIGS_DIR / candidate_name
        if candidate.exists():
            logger.info("从数据集配置目录加载: %s", candidate)
            return candidate

    logger.warning(
        "数据集配置未在 %s 找到: %s",
        DATASET_CONFIGS_DIR,
        data_path,
    )
    return data_path if data_path.suffix else Path(f"{data_path}.yaml")


__all__ = ["resolve_dataset_path"]
