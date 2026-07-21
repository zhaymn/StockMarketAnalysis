"""FastAPI application entry point.

Run:  .venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analysis, macro, markets, meta, news
from app.core.config import get_settings
from app.core.errors import StockIntelError
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="StockIntel API",
    description=(
        "AI stock market prediction and intelligence platform. "
        "Predictions are gated on demonstrated out-of-sample skill; where a model "
        "does not beat its naive baseline, the API reports NO_EDGE rather than a "
        "directional call."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(StockIntelError)
async def handle_application_error(request: Request, exc: StockIntelError) -> JSONResponse:
    """Render typed errors as structured payloads.

    The frontend switches on `code` to pick the right empty state -- NOT
    CONFIGURED, DATA UNAVAILABLE, INSUFFICIENT HISTORY -- so an error is
    displayed as an honest state rather than a generic failure toast.
    """
    logger.info("%s -> %s: %s", request.url.path, exc.code, exc.message)
    return JSONResponse(status_code=exc.http_status, content=exc.to_payload())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "An unexpected error occurred.",
            "detail": str(exc)[:300],
        },
    )


app.include_router(meta.router)
app.include_router(markets.router)
app.include_router(analysis.router)
app.include_router(news.router)
app.include_router(macro.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, object]:
    return {"status": "ok", "environment": settings.stockintel_env}
