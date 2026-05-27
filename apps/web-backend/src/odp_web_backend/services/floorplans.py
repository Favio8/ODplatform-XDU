from __future__ import annotations

import base64
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .workspace import ROOT


FLOORPLAN_RUNS_DIR = ROOT / "runs" / "agent" / "floorplans"
FLOORPLAN_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _record_path(record_id: str) -> Path:
    return FLOORPLAN_RUNS_DIR / record_id / "record.json"


def _safe_name(filename: str) -> str:
    name = filename.replace("\\", "_").replace("/", "_").strip()
    return name or "floorplan.jpg"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_floorplan_record(
    *,
    filename: str,
    image_bytes: bytes,
    visualization_base64: str,
    rooms: list[dict[str, Any]],
    image_size: dict[str, Any],
    session_id: str,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    record_dir = FLOORPLAN_RUNS_DIR / record_id
    record_dir.mkdir(parents=True, exist_ok=True)

    original_name = _safe_name(filename)
    original_path = record_dir / f"original{Path(original_name).suffix or '.jpg'}"
    visualization_path = record_dir / "visualization.jpg"
    original_path.write_bytes(image_bytes)
    visualization_path.write_bytes(base64.b64decode(visualization_base64))

    created_at = _now()
    record = {
        "record_id": record_id,
        "filename": original_name,
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": created_at,
        "agent_status": "analyzing",
        "agent_error": None,
        "analysis": None,
        "image_size": image_size,
        "rooms": rooms,
        "room_count": len(rooms),
        "requirements": requirements or {},
        "original_path": str(original_path),
        "visualization_path": str(visualization_path),
        "visualization": visualization_base64,
        "summary": f"已识别 {len(rooms)} 个空间区域，Agent 正在生成装修建议。",
    }
    _write_json(_record_path(record_id), record)
    return record


def update_floorplan_agent_result(
    *,
    record_id: str,
    status: str,
    analysis: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    path = _record_path(record_id)
    if not path.exists():
        return None
    record = _read_json(path)
    record["agent_status"] = status
    record["agent_error"] = error
    record["analysis"] = analysis
    record["updated_at"] = _now()
    if analysis:
        record["summary"] = str(
            analysis.get("overall_assessment")
            or analysis.get("overall_suggestions")
            or record.get("summary")
        )
    elif error:
        record["summary"] = f"图像分割已完成，Agent 建议暂时不可用：{error}"
    _write_json(path, record)
    return record


def list_floorplan_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in FLOORPLAN_RUNS_DIR.glob("*/record.json"):
        try:
            records.append(_read_json(path))
        except Exception:
            continue
    return sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)


def get_floorplan_record(record_id: str) -> dict[str, Any]:
    path = _record_path(record_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Floorplan record {record_id} not found")
    return _read_json(path)


def delete_floorplan_record(record_id: str) -> dict[str, Any]:
    record_dir = FLOORPLAN_RUNS_DIR / record_id
    if not record_dir.exists():
        raise HTTPException(status_code=404, detail=f"Floorplan record {record_id} not found")
    shutil.rmtree(record_dir)
    return {"success": True, "record_id": record_id}
