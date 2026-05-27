from __future__ import annotations

from fastapi import APIRouter

from ..services.workspace import latest_inference_result

router = APIRouter(prefix="/api", tags=["inference"])


@router.get("/inference/latest")
def get_latest_inference() -> dict:
    return latest_inference_result()

