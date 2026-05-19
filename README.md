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

## Initialization Commands

- `python .\scripts\init_workspace.py`
  - 用于第一次搭建或补齐整个 monorepo 工作区骨架
  - 会检查并补齐 `apps/`、`docs/`、`data/`、`models/`、`runs/` 等目录与模板文件
- `python .\scripts\init_project.py`
  - 用于 platform 应用级初始化
  - 会检查 platform 运行依赖的核心目录，并输出 `data/RSOD/raw` 的状态汇总

日常开发里，如果仓库骨架已经齐全，通常优先使用 `python .\scripts\init_project.py`。

## Git Workflow

- Default branch: `main`
- Feature branches: `feature/<topic>`
- Documentation branches: `docs/<topic>`
- Fix branches: `fix/<topic>`
