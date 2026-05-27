from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from ..services.agent_runtime.runtime import analyze_floorplan, chat, get_session_payload, stream_chat
from ..services.workspace import latest_agent_report

router = APIRouter(prefix="/api", tags=["agent"])


@router.get("/agent/report/latest")
def get_latest_report() -> dict:
    return latest_agent_report()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), requirements: str = Form("{}")) -> dict:
    image_bytes = await file.read()
    try:
        parsed_requirements = json.loads(requirements) if requirements else {}
    except json.JSONDecodeError:
        parsed_requirements = {"notes": requirements}
    return analyze_floorplan(
        image_bytes,
        filename=file.filename or "floorplan.jpg",
        requirements=parsed_requirements if isinstance(parsed_requirements, dict) else {},
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    return get_session_payload(session_id)


@router.post("/chat/{session_id}")
async def post_chat(session_id: str, message: str = Form(...)) -> dict:
    return chat(session_id=session_id, message=message)


@router.post("/chat/{session_id}/stream")
async def post_chat_stream(session_id: str, message: str = Form(...)) -> StreamingResponse:
    return StreamingResponse(
        stream_chat(session_id=session_id, message=message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
