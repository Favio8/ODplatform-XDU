from __future__ import annotations

from fastapi import APIRouter

from ..services.serving_models import list_pretrained_models, list_serving_models


router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models/serving")
def get_serving_models() -> dict:
    return {"items": list_serving_models()}


@router.get("/models/pretrained")
def get_pretrained_models() -> dict:
    return {"items": list_pretrained_models()}
