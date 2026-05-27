from __future__ import annotations

from fastapi import APIRouter

from ..services.floorplans import delete_floorplan_record, get_floorplan_record, list_floorplan_records

router = APIRouter(prefix="/api", tags=["floorplans"])


@router.get("/floorplans")
def get_floorplans() -> dict:
    return {"items": list_floorplan_records()}


@router.get("/floorplans/{record_id}")
def get_floorplan(record_id: str) -> dict:
    return get_floorplan_record(record_id)


@router.delete("/floorplans/{record_id}")
def delete_floorplan(record_id: str) -> dict:
    return delete_floorplan_record(record_id)
