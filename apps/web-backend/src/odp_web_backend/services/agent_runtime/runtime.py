from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps app importable.
    def load_dotenv(*_args, **_kwargs):
        return False

from .agent import Agent
from .memory import memory
from .model_handler import ModelHandler
from ..floorplans import create_floorplan_record, update_floorplan_agent_result


ROOT = Path(__file__).resolve().parents[6]
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "apps" / "web-backend" / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@lru_cache(maxsize=1)
def get_model_handler() -> ModelHandler:
    model_path = _env("MODEL_PATH") or str(_latest_seg_checkpoint())
    try:
        return ModelHandler(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load segmentation model from MODEL_PATH={model_path!r}. "
            "Set MODEL_PATH to a valid local YOLO segmentation checkpoint."
        ) from exc


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    api_key = _env("LLM_API_KEY")
    base_url = _env("LLM_BASE_URL", "https://api.openai.com/v1")
    model = _env("LLM_MODEL", "gpt-4o")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")
    return Agent(api_key=api_key, base_url=base_url, model=model)


def _run_agent_background(
    session_id: str,
    visualization: str,
    rooms: list,
    image_size: dict,
    requirements: dict | None = None,
    record_id: str | None = None,
) -> None:
    try:
        sess = memory.get(session_id)
        if sess:
            sess["status"] = "analyzing"

        result = get_agent().analyze(
            image_base64=visualization,
            rooms=rooms,
            image_size=image_size,
            requirements=requirements or {},
            session_id=session_id,
            memory=memory,
        )
        memory.add_analysis(session_id, result["analysis"])

        if sess:
            sess["status"] = "done"
            sess["analysis"] = result["analysis"]
        if record_id:
            update_floorplan_agent_result(
                record_id=record_id,
                status="done",
                analysis=result["analysis"],
            )
    except Exception as exc:
        sess = memory.get(session_id)
        if sess:
            sess["status"] = "error"
            sess["error"] = str(exc)
        if record_id:
            update_floorplan_agent_result(
                record_id=record_id,
                status="error",
                error=str(exc),
            )


def _latest_seg_checkpoint() -> Path:
    candidates = sorted(
        CHECKPOINTS_DIR.glob("*seg-best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return CHECKPOINTS_DIR / "room_separation_3-best.pt"


def analyze_floorplan(image_bytes: bytes, filename: str = "floorplan.jpg", requirements: dict | None = None) -> dict:
    try:
        results = get_model_handler().predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO segmentation failed: {exc}") from exc

    session_id = memory.create(
        yolo_rooms=results["rooms"],
        image_size=results["image_size"],
        visualization=results["visualization"],
    )
    sess = memory.get(session_id)
    if sess:
        sess["status"] = "analyzing"
        sess["requirements"] = requirements or {}

    record = create_floorplan_record(
        filename=filename,
        image_bytes=image_bytes,
        visualization_base64=results["visualization"],
        rooms=results["rooms"],
        image_size=results["image_size"],
        session_id=session_id,
        requirements=requirements or {},
    )

    threading.Thread(
        target=_run_agent_background,
        args=(
            session_id,
            results["visualization"],
            results["rooms"],
            results["image_size"],
            requirements or {},
            record["record_id"],
        ),
        daemon=True,
    ).start()

    return {
        "record_id": record["record_id"],
        "session_id": session_id,
        "image_size": results["image_size"],
        "visualization": results["visualization"],
        "yolo_rooms": results["rooms"],
        "requirements": requirements or {},
        "status": "analyzing",
    }


def get_session_payload(session_id: str) -> dict:
    sess = memory.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session does not exist or has expired.")
    return {
        "session_id": session_id,
        "status": sess.get("status", "unknown"),
        "analyses": sess["analyses"],
        "messages": sess["messages"],
        "reasoning_steps": sess["reasoning_steps"],
        "analysis": sess.get("analysis"),
        "image_size": sess["image_size"],
        "visualization": sess["visualization"],
        "yolo_rooms": sess["yolo_rooms"],
        "requirements": sess.get("requirements", {}),
        "error": sess.get("error"),
    }


def chat(session_id: str, message: str) -> dict:
    try:
        result = get_agent().chat(session_id, message, memory)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {exc}") from exc
    return {
        "session_id": session_id,
        "reply": result["reply"],
        "reasoning_steps": result["reasoning_steps"],
    }


def stream_chat(session_id: str, message: str) -> Iterator[str]:
    sess = memory.get(session_id)
    if not sess:
        yield f"data: {json.dumps({'token': '会话已过期，请重新上传。'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        agent = get_agent()
        messages = _chat_messages(session_id=session_id, message=message, sess=sess)
        stream = agent.client.chat.completions.create(
            model=agent.model,
            messages=messages,
            max_completion_tokens=2048,
            temperature=0.7,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                token = delta.content
                full_reply += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        memory.add_message(session_id, "user", message)
        memory.add_message(session_id, "assistant", full_reply)
    except Exception as exc:
        yield f"data: {json.dumps({'token': f'[错误: {exc}]'}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def _chat_messages(session_id: str, message: str, sess: dict) -> list[dict]:
    chat_system = (
        "你是一个有用的AI助手，同时也是室内设计专家。"
        "你正在与用户讨论一份已分析过的户型图。"
        "可以回答装修、户型、设计相关问题，也可以闲聊。"
        "用中文回复，简洁友好。"
    )
    messages = [{"role": "system", "content": chat_system}]
    if sess.get("analyses"):
        last = sess["analyses"][-1]
        ctx = (
            f"之前分析：户型={last.get('house_type', '未知')}，"
            f"评级={last.get('rating', 'N/A')}，"
            f"{len(sess['yolo_rooms'])}个房间。"
        )
        messages.append({"role": "user", "content": ctx})
        messages.append({"role": "assistant", "content": "了解。请问想进一步了解什么？"})
    messages.extend(sess.get("messages", [])[-6:])
    messages.append({"role": "user", "content": message})
    return messages
