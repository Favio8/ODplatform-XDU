import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # 阶段 1: YOLO 分割
    results = model_handler.predict(image_bytes)

    # 创建会话
    session_id = memory.create(
        yolo_rooms=results["rooms"],
        image_size=results["image_size"],
        visualization=results["visualization"],
    )

    # 阶段 2: Agent 多步推理
    result = agent.analyze(
        image_base64=results["visualization"],
        rooms=results["rooms"],
        image_size=results["image_size"],
        session_id=session_id,
        memory=memory,
    )

    # 保存分析结果到会话
    memory.add_analysis(session_id, result["analysis"])

    return {
        "session_id": session_id,
        "image_size": results["image_size"],
        "visualization": results["visualization"],
        "yolo_rooms": results["rooms"],
        "analysis": result["analysis"],
        "reasoning_steps": result["reasoning_steps"],
    }


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, message: str = Form(...)):
    """交互式对话：基于已分析的户型进行追问。"""
    result = agent.chat(session_id, message, memory)
    return {
        "session_id": session_id,
        "reply": result["reply"],
        "reasoning_steps": result["reasoning_steps"],
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话历史。"""
    sess = memory.get(session_id)
    if not sess:
        return {"error": "会话不存在或已过期"}
    return {
        "session_id": session_id,
        "analyses": sess["analyses"],
        "messages": sess["messages"],
        "reasoning_steps": sess["reasoning_steps"],
    }


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND / "index.html"))
