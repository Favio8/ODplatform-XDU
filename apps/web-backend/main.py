import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import Agent
from memory import memory
from model_handler import ModelHandler

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

app = FastAPI(title="ODPlatform Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_handler = ModelHandler(MODEL_PATH)
agent = Agent(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)

FRONTEND = (Path(__file__).parent.parent / "web-frontend").resolve()
FRONTEND.mkdir(parents=True, exist_ok=True)


def _run_agent_background(session_id: str, visualization: str, rooms: list, image_size: dict):
    """后台线程：运行 Agent 推理，结果写入 session memory。"""
    try:
        sess = memory.get(session_id)
        if sess:
            sess["status"] = "analyzing"

        result = agent.analyze(
            image_base64=visualization,
            rooms=rooms,
            image_size=image_size,
            session_id=session_id,
            memory=memory,
        )
        memory.add_analysis(session_id, result["analysis"])

        if sess:
            sess["status"] = "done"
            sess["analysis"] = result["analysis"]
    except Exception as e:
        sess = memory.get(session_id)
        if sess:
            sess["status"] = "error"
            sess["error"] = str(e)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # 阶段 1: YOLO 分割（同步，1-2 秒完成）
    results = model_handler.predict(image_bytes)

    # 创建会话
    session_id = memory.create(
        yolo_rooms=results["rooms"],
        image_size=results["image_size"],
        visualization=results["visualization"],
    )
    memory.get(session_id)["status"] = "analyzing"

    # 阶段 2: Agent 多步推理（后台异步）
    threading.Thread(
        target=_run_agent_background,
        args=(session_id, results["visualization"], results["rooms"], results["image_size"]),
        daemon=True,
    ).start()

    # 立即返回 YOLO 结果 + session_id
    return {
        "session_id": session_id,
        "image_size": results["image_size"],
        "visualization": results["visualization"],
        "yolo_rooms": results["rooms"],
        "status": "analyzing",
    }


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, message: str = Form(...)):
    result = agent.chat(session_id, message, memory)
    return {
        "session_id": session_id,
        "reply": result["reply"],
        "reasoning_steps": result["reasoning_steps"],
    }


@app.post("/api/chat/{session_id}/stream")
async def chat_stream(session_id: str, message: str = Form(...)):
    """流式对话端点，SSE 逐 token 输出。"""

    def generate():
        sess = memory.get(session_id)
        if not sess:
            yield f"data: {json.dumps({'token': '会话已过期，请重新上传。'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 构建对话上下文
        messages = [{"role": "system", "content": agent._system_prompt()}]

        if sess.get("analyses"):
            last = sess["analyses"][-1]
            ctx = (
                f"之前分析：户型={last.get('house_type','未知')}，"
                f"评级={last.get('rating','N/A')}，"
                f"{len(sess['yolo_rooms'])}个房间。"
            )
            messages.append({"role": "user", "content": ctx})
            messages.append({"role": "assistant", "content": "了解。请问想进一步了解什么？"})

        messages.extend(sess.get("messages", [])[-6:])
        messages.append({"role": "user", "content": message})

        try:
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
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # 保存到记忆
            memory.add_message(session_id, "user", message)
            memory.add_message(session_id, "assistant", full_reply)

        except Exception as e:
            yield f"data: {json.dumps({'token': f'[错误: {str(e)}]'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    sess = memory.get(session_id)
    if not sess:
        return {"error": "会话不存在或已过期"}
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
        "error": sess.get("error"),
    }


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND / "index.html"))
