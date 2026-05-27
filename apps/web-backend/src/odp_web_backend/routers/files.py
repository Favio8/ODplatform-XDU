from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..services.evaluation_reports import safe_file_path


router = APIRouter(prefix="/api", tags=["files"])


@router.get("/files")
def get_file(path: str = Query(...)) -> FileResponse:
    try:
        return FileResponse(safe_file_path(path))
    except FileNotFoundError:
        raise HTTPException(404, "File not found")
