import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent import Agent
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

    results = model_handler.predict(image_bytes)

    analysis = agent.analyze(results["visualization"], results["rooms"])

    return {
        "image_size": results["image_size"],
        "visualization": results["visualization"],
        "yolo_rooms": results["rooms"],
        "analysis": analysis,
    }


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND / "index.html"))
