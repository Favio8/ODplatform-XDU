from __future__ import annotations

from fastapi import APIRouter

from ..services.workspace import list_checkpoints, list_training_runs

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/training/runs")
def get_runs() -> dict:
    return {"items": list_training_runs()}


@router.get("/checkpoints")
def get_checkpoints() -> dict:
    return {"items": list_checkpoints()}

