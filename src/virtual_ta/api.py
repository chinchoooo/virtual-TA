"""HTTP API. Serves the experiment named by ``VTA_CONFIG`` (default: configs/baseline.yaml).

Run with ``vta serve`` or ``uvicorn --factory virtual_ta.api:create_app``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .config import ExperimentConfig
from .llm import LLMError
from .pipeline import RAGPipeline
from .settings import PROJECT_ROOT, Settings

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    image: str | None = Field(None, description="Base64-encoded image or data URI.")
    term: str | None = Field(
        None,
        pattern=r"^20\d\d-(01|05|09)$",
        description="Term the question is about, e.g. 2025-01. Used if the config has term_filter.",
    )


class LinkModel(BaseModel):
    url: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    links: list[LinkModel]


def create_app(config: ExperimentConfig | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if config is None:
        path = os.getenv("VTA_CONFIG", str(PROJECT_ROOT / "configs" / "baseline.yaml"))
        config = ExperimentConfig.from_yaml(path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("Building pipeline for experiment '%s'", config.name)
        app.state.pipeline = RAGPipeline.from_config(config, settings)
        yield

    app = FastAPI(title="Virtual TA", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "experiment": config.name, "fingerprint": config.fingerprint()}

    @app.post("/api/", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        pipeline: RAGPipeline = app.state.pipeline
        try:
            result = pipeline.answer(req.question, req.image, req.term)
        except LLMError as e:
            log.exception("Upstream model call failed")
            raise HTTPException(status_code=502, detail="Upstream model call failed") from e
        return QueryResponse(
            answer=result.answer,
            links=[LinkModel(url=link.url, text=link.text) for link in result.links],
        )

    return app
