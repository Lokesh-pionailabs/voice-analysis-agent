"""
Application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8080

In production (Cloud Run), the Dockerfile invokes uvicorn directly with
multiple workers disabled in favor of Cloud Run's own container
concurrency model (see Dockerfile / README for details).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.models.response import ErrorResponse
from app.utils.config import get_settings
from app.utils.helpers import new_correlation_id
from app.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan hook: validate configuration at startup."""
    settings = get_settings()
    logger.info(
        "Starting AI Voice Analysis Agent (env=%s, project=%s, region=%s)",
        settings.app_env,
        settings.gcp_project_id,
        settings.gcp_region,
    )
    yield
    logger.info("Shutting down AI Voice Analysis Agent.")


app = FastAPI(
    title="AI Voice Analysis Agent",
    description="Production-grade voice call analysis powered by Google Speech-to-Text and Vertex AI Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()
import os
if _settings.google_application_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _settings.google_application_credentials

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = new_correlation_id()
    logger.warning("[%s] HTTPException %s: %s", request_id, exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(request_id=request_id, error="request_failed", detail=str(exc.detail)).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = new_correlation_id()
    logger.exception("[%s] Unhandled exception: %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            request_id=request_id, error="internal_server_error", detail="An unexpected error occurred."
        ).model_dump(),
    )


@app.get("/health", tags=["ops"], summary="Liveness/readiness probe")
async def health_check() -> dict[str, str]:
    """Basic health check for Cloud Run / load balancer probes."""
    return {"status": "ok"}


@app.get("/", tags=["ops"], summary="Service info")
async def root() -> dict[str, str]:
    return {"service": "ai-voice-analysis-agent", "docs": "/docs"}
