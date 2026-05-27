from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import agent, datasets, evaluation, files, floorplans, health, inference, jobs, models, overview, pipeline, runs, uploads


def create_app() -> FastAPI:
    app = FastAPI(title="ODPlatform Web Backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(overview.router)
    app.include_router(pipeline.router)
    app.include_router(datasets.router)
    app.include_router(evaluation.router)
    app.include_router(files.router)
    app.include_router(floorplans.router)
    app.include_router(models.router)
    app.include_router(runs.router)
    app.include_router(inference.router)
    app.include_router(agent.router)
    app.include_router(jobs.router)
    app.include_router(uploads.router)
    return app


app = create_app()
