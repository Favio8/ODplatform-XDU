from __future__ import annotations

from fastapi import APIRouter

from ..services.workspace import pipeline_stages

router = APIRouter(prefix="/api", tags=["pipeline"])


@router.get("/pipeline/status")
def get_pipeline_status() -> dict:
    return {"stages": pipeline_stages()}

