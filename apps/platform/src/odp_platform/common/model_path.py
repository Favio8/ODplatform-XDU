"""Model-path resolution helpers shared by training, validation, and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from odp_platform.common.paths import PRETRAINED_MODELS_DIR


logger = logging.getLogger(__name__)


def resolve_model_path(
    model: str | Path,
    search_dirs: Sequence[Path] | None = None,
) -> Path:
    """Resolve a model reference to a concrete path when possible.

    Resolution order:
    1. Absolute path.
    2. Existing relative path from the current working directory.
    3. Search by filename under the provided search directories.
    4. Fallback to the original value so Ultralytics may still auto-download.
    """

    model_path = Path(model)

    if model_path.is_absolute():
        return model_path

    if model_path.exists():
        return model_path.resolve()

    directories = list(search_dirs) if search_dirs is not None else [PRETRAINED_MODELS_DIR]
    for directory in directories:
        candidate = directory / model_path.name
        if candidate.exists():
            logger.info("从模型目录加载模型: %s", candidate)
            return candidate

    logger.warning(
        "模型文件未在任何搜索目录命中: %s\n  搜索目录: %s\n  将把原始值交给下游继续处理。",
        model_path.name,
        [str(directory) for directory in directories],
    )
    return model_path


__all__ = ["resolve_model_path"]
