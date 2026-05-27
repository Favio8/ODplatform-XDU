from __future__ import annotations

from fastapi import APIRouter

from ..services.workspace import overview_payload, project_status

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview")
def get_overview() -> dict:
    return overview_payload()


@router.get("/project/status")
def get_project_status() -> dict:
    return project_status()
