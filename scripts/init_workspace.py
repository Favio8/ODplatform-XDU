from __future__ import annotations

from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "apps/platform/src/odp_platform/common",
    "apps/platform/src/odp_platform/config",
    "apps/platform/src/odp_platform/data_pipline/core",
    "apps/platform/src/odp_platform/data_validation/core",
    "apps/platform/src/odp_platform/training",
    "apps/platform/src/odp_platform/evaluation",
    "apps/platform/src/odp_platform/inference",
    "apps/platform/src/odp_platform/cli",
    "apps/platform/tests/unit/common",
    "apps/platform/tests/unit/config",
    "apps/platform/tests/unit/data_validation",
    "apps/platform/tests/integration",
    "apps/platform/configs",
    "apps/platform/logging",
    "apps/web-backend",
    "apps/web-frontend",
    "apps/desktop",
    "tests",
    "docs/architecture",
    "docs/srs",
    "docs/teaching",
    "docs/api",
    "scripts",
    "data/RSOD/raw",
    "data/RSOD/yolo",
    "models/pretrained",
    "models/checkpoints",
    "runs",
]

FILES = {
    ".odp-workspace": "",
    "README.md": """
    # ODPlatform

    ODPlatform is a monorepo for a general-purpose object detection development platform.
    The current milestone focuses on repository bootstrap, package structure, and Git workflow.

    ## Workspace Layout

    - `apps/platform`: core Python application and CLI entrypoints
    - `apps/web-backend`: placeholder for a future web backend
    - `apps/web-frontend`: placeholder for a future web frontend
    - `apps/desktop`: placeholder for a future desktop client
    - `docs`: architecture, teaching, and project-level documentation
    - `data`: dataset layout skeleton
    - `models`: pretrained weights and checkpoints skeleton
    - `runs`: experiment outputs (ignored by Git)

    ## Quick Start

    ```powershell
    conda activate odplatform-gpu
    python scripts/init_workspace.py
    ```

    ## Git Workflow

    - Default branch: `main`
    - Feature branches: `feature/<topic>`
    - Documentation branches: `docs/<topic>`
    - Fix branches: `fix/<topic>`
    """,
    "LICENSE": """
    MIT License

    Copyright (c) 2026 ODPlatform

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """,
    ".gitignore": """
    __pycache__/
    *.py[cod]
    *.egg-info/
    .coverage
    .pytest_cache/
    .ruff_cache/
    .mypy_cache/
    .venv/
    .idea/
    .vscode/
    htmlcov/

    apps/platform/logging/*
    !apps/platform/logging/.gitkeep

    runs/*
    !runs/.gitkeep

    data/RSOD/raw/*
    !data/RSOD/raw/.gitkeep

    data/RSOD/yolo/*
    !data/RSOD/yolo/.gitkeep

    models/pretrained/*
    !models/pretrained/.gitkeep

    models/checkpoints/*
    !models/checkpoints/.gitkeep
    """,
    ".gitattributes": """
    *.pt filter=lfs diff=lfs merge=lfs -text
    *.pth filter=lfs diff=lfs merge=lfs -text
    *.onnx filter=lfs diff=lfs merge=lfs -text
    *.ckpt filter=lfs diff=lfs merge=lfs -text
    """,
    "pyproject.toml": """
    [build-system]
    requires = ["hatchling>=1.27.0"]
    build-backend = "hatchling.build"

    [project]
    name = "odplatform-workspace"
    version = "0.1.0"
    description = "Workspace metadata for the ODPlatform monorepo."
    readme = "README.md"
    requires-python = ">=3.11,<3.12"
    license = { text = "MIT" }
    authors = [{ name = "ODPlatform Team" }]
    dependencies = []

    [project.optional-dependencies]
    dev = [
      "pytest>=8.0",
      "ruff>=0.6.0",
    ]

    [tool.pytest.ini_options]
    addopts = "-ra"
    testpaths = [
      "apps/platform/tests",
      "tests",
    ]

    [tool.ruff]
    target-version = "py311"
    line-length = 100
    src = [
      "apps/platform/src",
      "apps/platform/tests",
      "tests",
    ]

    [tool.ruff.lint]
    select = ["E", "F", "I", "B", "UP"]
    """,
    "scripts/bootstrap.sh": """
    #!/usr/bin/env bash
    set -euo pipefail

    python scripts/init_workspace.py
    echo "Workspace skeleton is ready. Activate the odplatform-gpu Conda environment before installing runtime dependencies."
    """,
    "scripts/release.py": """
    \"\"\"Release workflow placeholder for the ODPlatform workspace.\"\"\"


    def main() -> int:
        print("TODO: implement workspace release automation.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
    "apps/platform/README.md": """
    # Platform App

    The `platform` app is the core Python application for ODPlatform.
    It owns the layered package structure for configuration, data processing,
    training orchestration, evaluation orchestration, and inference orchestration.
    """,
    "apps/platform/pyproject.toml": """
    [build-system]
    requires = ["hatchling>=1.27.0"]
    build-backend = "hatchling.build"

    [project]
    name = "odp-platform"
    version = "0.1.0"
    description = "Core engine package for ODPlatform."
    readme = "README.md"
    requires-python = ">=3.11,<3.12"
    license = { text = "MIT" }
    authors = [{ name = "ODPlatform Team" }]
    dependencies = []

    [project.scripts]
    odp-trans = "odp_platform.cli.trans:main"
    odp-validate = "odp_platform.cli.validate:main"
    odp-train = "odp_platform.cli.train:main"
    odp-val = "odp_platform.cli.val:main"
    odp-infer = "odp_platform.cli.infer:main"

    [tool.hatch.build.targets.wheel]
    packages = ["src/odp_platform"]
    """,
    "apps/platform/configs/train.example.yaml": """
    experiment:
      name: rsod-train-example
    data:
      source: data/RSOD/raw
      target: data/RSOD/yolo
    model:
      weights: models/pretrained/yolo11n.pt
    train:
      epochs: 100
      image_size: 640
    """,
    "apps/platform/configs/val.example.yaml": """
    experiment:
      name: rsod-val-example
    data:
      source: data/RSOD/yolo
    model:
      checkpoint: models/checkpoints/latest.pt
    validation:
      batch_size: 16
    """,
    "apps/platform/configs/infer.example.yaml": """
    experiment:
      name: rsod-infer-example
    data:
      source: data/RSOD/yolo
    model:
      checkpoint: models/checkpoints/latest.pt
    inference:
      conf_threshold: 0.25
    """,
    "apps/platform/src/odp_platform/__init__.py": """
    \"\"\"ODPlatform core package.\"\"\"

    from ._version import __version__

    __all__ = ["__version__"]
    """,
    "apps/platform/src/odp_platform/_version.py": """
    __version__ = "0.1.0"
    """,
    "apps/platform/src/odp_platform/common/__init__.py": """
    \"\"\"Common infrastructure helpers for the ODPlatform package.\"\"\"
    """,
    "apps/platform/src/odp_platform/common/paths.py": """
    \"\"\"Workspace path helpers.\"\"\"

    from __future__ import annotations

    from pathlib import Path


    WORKSPACE_MARKER = ".odp-workspace"


    def find_workspace_root(start: Path | None = None) -> Path:
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / WORKSPACE_MARKER).exists():
                return candidate
        raise FileNotFoundError(f"Unable to locate {WORKSPACE_MARKER!r} from {current}")


    def workspace_path(*parts: str) -> Path:
        return find_workspace_root(Path(__file__)).joinpath(*parts)
    """,
    "apps/platform/src/odp_platform/common/logging_utils.py": """
    \"\"\"Logging helpers for ODPlatform.\"\"\"

    from __future__ import annotations

    import logging


    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    """,
    "apps/platform/src/odp_platform/common/system_utils.py": """
    \"\"\"System inspection helpers.\"\"\"

    from __future__ import annotations

    import platform
    import sys


    def python_runtime() -> str:
        return sys.version


    def platform_summary() -> str:
        return platform.platform()
    """,
    "apps/platform/src/odp_platform/common/string_utils.py": """
    \"\"\"String helpers.\"\"\"

    from __future__ import annotations

    import re


    def slugify(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
        return normalized.strip("-")
    """,
    "apps/platform/src/odp_platform/common/performance_utils.py": """
    \"\"\"Performance measurement helpers.\"\"\"

    from __future__ import annotations

    from contextlib import contextmanager
    from time import perf_counter
    from typing import Iterator


    @contextmanager
    def timer() -> Iterator[float]:
        start = perf_counter()
        yield start
        _ = perf_counter() - start
    """,
    "apps/platform/src/odp_platform/config/__init__.py": """
    \"\"\"Configuration models and loaders.\"\"\"
    """,
    "apps/platform/src/odp_platform/config/base.py": """
    \"\"\"Base configuration model.\"\"\"

    from __future__ import annotations

    from dataclasses import dataclass, field
    from typing import Any


    @dataclass(slots=True)
    class BaseConfig:
        name: str = "default"
        extras: dict[str, Any] = field(default_factory=dict)
    """,
    "apps/platform/src/odp_platform/config/train_config.py": """
    \"\"\"Training configuration placeholders.\"\"\"

    from __future__ import annotations

    from dataclasses import dataclass

    from .base import BaseConfig


    @dataclass(slots=True)
    class TrainConfig(BaseConfig):
        epochs: int = 100
        image_size: int = 640
    """,
    "apps/platform/src/odp_platform/config/val_config.py": """
    \"\"\"Validation configuration placeholders.\"\"\"

    from __future__ import annotations

    from dataclasses import dataclass

    from .base import BaseConfig


    @dataclass(slots=True)
    class ValConfig(BaseConfig):
        batch_size: int = 16
    """,
    "apps/platform/src/odp_platform/config/infer_config.py": """
    \"\"\"Inference configuration placeholders.\"\"\"

    from __future__ import annotations

    from dataclasses import dataclass

    from .base import BaseConfig


    @dataclass(slots=True)
    class InferConfig(BaseConfig):
        conf_threshold: float = 0.25
    """,
    "apps/platform/src/odp_platform/config/loaders.py": """
    \"\"\"Configuration loading placeholders.\"\"\"

    from __future__ import annotations

    from pathlib import Path


    def load_config(config_path: str | Path) -> Path:
        return Path(config_path)
    """,
    "apps/platform/src/odp_platform/config/merger.py": """
    \"\"\"Configuration merge placeholders.\"\"\"

    from __future__ import annotations

    from typing import Any


    def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for config in configs:
            merged.update(config)
        return merged
    """,
    "apps/platform/src/odp_platform/config/generator.py": """
    \"\"\"Configuration template generation placeholders.\"\"\"

    from __future__ import annotations

    from typing import Any


    def generate_template(template_name: str) -> dict[str, Any]:
        return {"template": template_name, "status": "not-implemented"}
    """,
    "apps/platform/src/odp_platform/data_pipline/__init__.py": """
    \"\"\"Dataset conversion layer.\"\"\"
    """,
    "apps/platform/src/odp_platform/data_pipline/service.py": """
    \"\"\"Data conversion orchestration placeholders.\"\"\"


    class DataPipelineService:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_pipline/core/__init__.py": """
    \"\"\"Format-specific data conversion helpers.\"\"\"
    """,
    "apps/platform/src/odp_platform/data_pipline/core/pascal_voc.py": """
    \"\"\"Pascal VOC conversion placeholders for the RSOD dataset.\"\"\"

    SUPPORTED_SOURCE_FORMAT = "pascal_voc"
    """,
    "apps/platform/src/odp_platform/data_pipline/core/coco.py": """
    \"\"\"COCO conversion placeholders.\"\"\"

    SUPPORTED_SOURCE_FORMAT = "coco"
    """,
    "apps/platform/src/odp_platform/data_validation/__init__.py": """
    \"\"\"Dataset validation layer.\"\"\"
    """,
    "apps/platform/src/odp_platform/data_validation/service.py": """
    \"\"\"Data validation orchestration placeholders.\"\"\"


    class DataValidationService:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_validation/core/__init__.py": """
    \"\"\"Validation core components.\"\"\"
    """,
    "apps/platform/src/odp_platform/data_validation/core/validators.py": """
    \"\"\"Validation rule placeholders.\"\"\"


    def validate_dataset() -> str:
        return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_validation/core/analyzers.py": """
    \"\"\"Dataset analysis placeholders.\"\"\"


    def analyze_dataset() -> str:
        return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_validation/core/cleaners.py": """
    \"\"\"Dataset cleaning placeholders.\"\"\"


    def clean_dataset() -> str:
        return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_validation/core/visualizers.py": """
    \"\"\"Visualization placeholders.\"\"\"


    def visualize_dataset() -> str:
        return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_validation/core/chart_generator.py": """
    \"\"\"Chart generation placeholders.\"\"\"


    def generate_charts() -> str:
        return "not-implemented"
    """,
    "apps/platform/src/odp_platform/data_validation/core/reporters.py": """
    \"\"\"Reporting placeholders.\"\"\"


    def build_report() -> str:
        return "not-implemented"
    """,
    "apps/platform/src/odp_platform/training/__init__.py": """
    \"\"\"Training orchestration layer.\"\"\"
    """,
    "apps/platform/src/odp_platform/training/service.py": """
    \"\"\"Training service placeholders.\"\"\"


    class TrainingService:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/evaluation/__init__.py": """
    \"\"\"Evaluation orchestration layer.\"\"\"
    """,
    "apps/platform/src/odp_platform/evaluation/service.py": """
    \"\"\"Evaluation service placeholders.\"\"\"


    class EvaluationService:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/inference/__init__.py": """
    \"\"\"Inference orchestration layer.\"\"\"
    """,
    "apps/platform/src/odp_platform/inference/service.py": """
    \"\"\"Inference service placeholders.\"\"\"


    class InferenceService:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/inference/pipeline.py": """
    \"\"\"Inference pipeline placeholders.\"\"\"


    class InferencePipeline:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/inference/components.py": """
    \"\"\"Inference component placeholders.\"\"\"


    class PipelineComponent:
        def status(self) -> str:
            return "not-implemented"
    """,
    "apps/platform/src/odp_platform/cli/__init__.py": """
    \"\"\"CLI entrypoints for ODPlatform.\"\"\"
    """,
    "apps/platform/src/odp_platform/cli/trans.py": """
    \"\"\"CLI placeholder for dataset conversion.\"\"\"


    def main() -> int:
        print("TODO: implement dataset conversion command.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
    "apps/platform/src/odp_platform/cli/validate.py": """
    \"\"\"CLI placeholder for dataset validation.\"\"\"


    def main() -> int:
        print("TODO: implement dataset validation command.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
    "apps/platform/src/odp_platform/cli/train.py": """
    \"\"\"CLI placeholder for training orchestration.\"\"\"


    def main() -> int:
        print("TODO: implement training command.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
    "apps/platform/src/odp_platform/cli/val.py": """
    \"\"\"CLI placeholder for evaluation orchestration.\"\"\"


    def main() -> int:
        print("TODO: implement evaluation command.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
    "apps/platform/src/odp_platform/cli/infer.py": """
    \"\"\"CLI placeholder for inference orchestration.\"\"\"


    def main() -> int:
        print("TODO: implement inference command.")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """,
    "apps/platform/tests/conftest.py": """
    from __future__ import annotations

    import sys
    from pathlib import Path


    SRC_DIR = Path(__file__).resolve().parents[1] / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    """,
    "apps/platform/tests/integration/test_init_project.py": """
    import importlib

    import odp_platform


    def test_package_exposes_version() -> None:
        assert odp_platform.__version__ == "0.1.0"


    def test_cli_modules_are_importable() -> None:
        for module_name in ("trans", "validate", "train", "val", "infer"):
            module = importlib.import_module(f"odp_platform.cli.{module_name}")
            assert callable(module.main)
    """,
    "apps/platform/tests/integration/test_train_pipeline.py": """
    from odp_platform.training.service import TrainingService


    def test_training_service_placeholder_status() -> None:
        service = TrainingService()
        assert service.status() == "not-implemented"
    """,
    "apps/platform/tests/unit/common/.gitkeep": "",
    "apps/platform/tests/unit/config/.gitkeep": "",
    "apps/platform/tests/unit/data_validation/.gitkeep": "",
    "apps/platform/logging/.gitkeep": "",
    "apps/web-backend/README.md": """
    # Web Backend

    TBD - V1.1 startup placeholder.
    """,
    "apps/web-frontend/README.md": """
    # Web Frontend

    TBD - V1.1 startup placeholder.
    """,
    "apps/desktop/README.md": """
    # Desktop

    TBD - V2.0 startup placeholder.
    """,
    "tests/conftest.py": """
    from __future__ import annotations

    import sys
    from pathlib import Path


    SRC_DIR = Path(__file__).resolve().parents[1] / "apps" / "platform" / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    """,
    "tests/README.md": """
    # Cross-App Tests

    This directory is reserved for future end-to-end and cross-application tests.
    """,
    "docs/architecture/ADR-001-monorepo.md": """
    # ADR-001 Monorepo

    - Status: accepted
    - Decision: keep ODPlatform as a single root repository with app-level boundaries
    - Rationale: the teaching plan emphasizes a clear internal architecture before any multi-repo split
    """,
    "docs/architecture/ADR-002-paths-strategy.md": """
    # ADR-002 Paths Strategy

    - Status: accepted
    - Decision: use `.odp-workspace` as the workspace root marker
    - Rationale: utilities in `common.paths` can locate the workspace consistently across scripts and tests
    """,
    "docs/architecture/ADR-003-naming.md": """
    # ADR-003 Naming

    - Status: accepted
    - Decision: keep the package name `odp_platform` and preserve the teacher-provided directory names
    - Note: `data_pipline` remains unchanged during bootstrap for compatibility with the current specification
    """,
    "docs/srs/ODPlatform_SRS_V1.0.md": """
    # ODPlatform SRS V1.0

    This placeholder will host the software requirements specification for the ODPlatform project.
    """,
    "docs/teaching/D1-architecture.md": """
    # D1 Architecture

    The first teaching milestone focuses on:

    - repository layout
    - layered package architecture
    - initialization workflow
    - Git collaboration conventions
    """,
    "docs/teaching/code-review-checklist.md": """
    # Code Review Checklist

    - Verify module boundaries stay aligned with the layered architecture
    - Keep CLI, service, core, and common responsibilities separated
    - Avoid committing datasets, checkpoints, or runtime outputs
    - Preserve idempotency for repository setup scripts
    """,
    "docs/teaching/实训方案.md": """
    # 实训方案

    当前阶段先完成项目初始化、目录骨架、Git 工作流约定，以及后续目标检测平台开发所需的基础工程结构。
    """,
    "docs/api/README.md": """
    # API Docs

    Future generated API documentation will be published here.
    """,
    "data/README.md": """
    # Data Layout

    - `RSOD/raw`: source dataset in Pascal VOC format
    - `RSOD/yolo`: converted YOLO-format dataset

    Keep large dataset files outside Git history.
    """,
    "data/RSOD/raw/.gitkeep": "",
    "data/RSOD/yolo/.gitkeep": "",
    "models/pretrained/.gitkeep": "",
    "models/checkpoints/.gitkeep": "",
    "runs/.gitkeep": "",
}


def normalize(content: str) -> str:
    if not content:
        return ""
    normalized = dedent(content).lstrip("\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def ensure_dir(relative_path: str) -> bool:
    target = ROOT / relative_path
    target.mkdir(parents=True, exist_ok=True)
    return True


def ensure_file(relative_path: str, content: str) -> bool:
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    target.write_text(normalize(content), encoding="utf-8")
    return True


def main() -> int:
    created_dirs = 0
    created_files = 0

    for relative_dir in DIRS:
        ensure_dir(relative_dir)
        created_dirs += 1

    for relative_file, content in FILES.items():
        if ensure_file(relative_file, content):
            created_files += 1

    print(f"Workspace root: {ROOT}")
    print(f"Ensured directories: {created_dirs}")
    print(f"Created files: {created_files}")
    print("Initialization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
