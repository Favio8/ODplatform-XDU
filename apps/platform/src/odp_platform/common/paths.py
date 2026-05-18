"""Workspace path helpers."""

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
