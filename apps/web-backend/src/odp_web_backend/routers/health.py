from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now().isoformat(timespec="seconds")}

