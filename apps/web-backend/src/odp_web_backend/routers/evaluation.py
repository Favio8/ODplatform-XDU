from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.evaluation_reports import get_evaluation_report, list_evaluation_reports


router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/reports")
def get_reports() -> dict:
    return {"items": list_evaluation_reports()}


@router.get("/reports/{report_id}")
def get_report(report_id: str) -> dict:
    try:
        return get_evaluation_report(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
