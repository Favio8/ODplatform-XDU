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
