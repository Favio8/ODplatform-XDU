"""Workspace-level path discovery helpers for the ODPlatform monorepo."""

from __future__ import annotations

from pathlib import Path
from typing import Final


WORKSPACE_MARKER: Final[str] = ".odp-workspace"


def find_workspace_root(start: Path | str | None = None) -> Path:
    """Locate the repository root by walking upward until the workspace marker is found."""

    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / WORKSPACE_MARKER).exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate {WORKSPACE_MARKER!r} from {current}")


# 工作区根目录（包含 .odp-workspace 标记的仓库根目录）
ROOT_DIR: Final[Path] = find_workspace_root(__file__)

# apps/ 目录 - 包含所有应用程序子包
APPS_DIR: Final[Path] = ROOT_DIR / "apps"

# apps/platform/ - 核心平台应用程序
PLATFORM_APP_DIR: Final[Path] = APPS_DIR / "platform"

# apps/platform/src/ - 平台源代码
PLATFORM_SRC_DIR: Final[Path] = PLATFORM_APP_DIR / "src"

# apps/platform/configs/ - 平台配置文件
PLATFORM_CONFIG_DIR: Final[Path] = PLATFORM_APP_DIR / "configs"

# apps/platform/logging/ - 日志配置和模板
PLATFORM_LOGGING_DIR: Final[Path] = PLATFORM_APP_DIR / "logging"

# apps/platform/logging/ 的兼容别名
LOGGING_DIR: Final[Path] = PLATFORM_LOGGING_DIR

# 顶层 data 目录 - 工作区共享的数据资源
DATA_DIR: Final[Path] = ROOT_DIR / "data"

# RSOD 数据集目录
RSOD_DATA_DIR: Final[Path] = DATA_DIR / "RSOD"
RAW_DATA_DIR: Final[Path] = RSOD_DATA_DIR / "raw"
YOLO_DATA_DIR: Final[Path] = RSOD_DATA_DIR / "yolo"

# 顶层 models 目录 - 存储的模型产物
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
PRETRAINED_MODELS_DIR: Final[Path] = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR: Final[Path] = MODELS_DIR / "checkpoints"

# 顶层 runs 目录 - 实验运行输出
RUNS_DIR: Final[Path] = ROOT_DIR / "runs"

# 顶层 docs 目录 - 项目文档
DOCS_DIR: Final[Path] = ROOT_DIR / "docs"

# 顶层 tests 目录 - 集成测试和端到端测试
TESTS_DIR: Final[Path] = ROOT_DIR / "tests"

# 顶层 scripts 目录 - 工具脚本和维护脚本
SCRIPTS_DIR: Final[Path] = ROOT_DIR / "scripts"


def workspace_path(*parts: str | Path) -> Path:
    """Build a path relative to the workspace root."""

    return ROOT_DIR.joinpath(*(str(part) for part in parts))


def app_path(*parts: str | Path) -> Path:
    """Build a path relative to the core platform application."""

    return PLATFORM_APP_DIR.joinpath(*(str(part) for part in parts))


def data_path(*parts: str | Path) -> Path:
    """Build a path relative to the shared data directory."""

    return DATA_DIR.joinpath(*(str(part) for part in parts))


def model_path(*parts: str | Path) -> Path:
    """Build a path relative to the shared model directory."""

    return MODELS_DIR.joinpath(*(str(part) for part in parts))


def run_path(*parts: str | Path) -> Path:
    """Build a path relative to the shared run-output directory."""

    return RUNS_DIR.joinpath(*(str(part) for part in parts))


def is_within_workspace(path: Path | str) -> bool:
    """Return whether a path resolves inside the current workspace."""

    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT_DIR)
    except ValueError:
        return False
    return True


def get_dirs_to_initialize() -> list[Path]:
    """Return the core directories that should exist for the platform workspace."""

    return [
        DATA_DIR,
        RSOD_DATA_DIR,
        RAW_DATA_DIR,
        YOLO_DATA_DIR,
        MODELS_DIR,
        PRETRAINED_MODELS_DIR,
        CHECKPOINTS_DIR,
        RUNS_DIR,
        DOCS_DIR,
        TESTS_DIR,
        SCRIPTS_DIR,
        PLATFORM_CONFIG_DIR,
        PLATFORM_LOGGING_DIR,
    ]


__all__ = [
    "WORKSPACE_MARKER",
    "ROOT_DIR",
    "APPS_DIR",
    "PLATFORM_APP_DIR",
    "PLATFORM_SRC_DIR",
    "PLATFORM_CONFIG_DIR",
    "PLATFORM_LOGGING_DIR",
    "LOGGING_DIR",
    "DATA_DIR",
    "RSOD_DATA_DIR",
    "RAW_DATA_DIR",
    "YOLO_DATA_DIR",
    "MODELS_DIR",
    "PRETRAINED_MODELS_DIR",
    "CHECKPOINTS_DIR",
    "RUNS_DIR",
    "DOCS_DIR",
    "TESTS_DIR",
    "SCRIPTS_DIR",
    "find_workspace_root",
    "get_dirs_to_initialize",
    "workspace_path",
    "app_path",
    "data_path",
    "model_path",
    "run_path",
    "is_within_workspace",
]
