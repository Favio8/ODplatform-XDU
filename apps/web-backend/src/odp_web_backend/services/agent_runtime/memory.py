"""会话记忆 —— 支持多轮对话和跨分析上下文。"""

import uuid
import time
from typing import Optional


class SessionMemory:
    """简单的内存会话存储。"""

    def __init__(self):
        self._sessions: dict = {}

    def create(self, yolo_rooms: list, image_size: dict, visualization: str) -> str:
        sid = str(uuid.uuid4())[:8]
        self._sessions[sid] = {
            "session_id": sid,
            "created_at": time.time(),
            "yolo_rooms": yolo_rooms,
            "image_size": image_size,
            "visualization": visualization,
            "analyses": [],           # 多轮分析记录
            "messages": [],           # 对话历史（OpenAI 格式）
            "reasoning_steps": [],    # CoT 推理步骤
        }
        return sid

    def get(self, sid: str) -> Optional[dict]:
        return self._sessions.get(sid)

    def add_analysis(self, sid: str, analysis: dict):
        s = self.get(sid)
        if s:
            s["analyses"].append({"timestamp": time.time(), **analysis})

    def add_message(self, sid: str, role: str, content):
        s = self.get(sid)
        if s:
            s["messages"].append({"role": role, "content": content})

    def add_reasoning(self, sid: str, step: dict):
        s = self.get(sid)
        if s:
            s["reasoning_steps"].append({"timestamp": time.time(), **step})

    def get_messages(self, sid: str) -> list:
        s = self.get(sid)
        return s["messages"] if s else []

    def get_reasoning(self, sid: str) -> list:
        s = self.get(sid)
        return s["reasoning_steps"] if s else []

    def get_analyses(self, sid: str) -> list:
        s = self.get(sid)
        return s["analyses"] if s else []

    def delete(self, sid: str):
        self._sessions.pop(sid, None)


# 全局单例
memory = SessionMemory()
