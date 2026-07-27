"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response

from triageai.config import get_settings
from triageai.infrastructure.api.routes import router
from triageai.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="TriageAI", version="0.1.0")
    app.include_router(router)

    @app.middleware("http")
    async def request_context_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        response.headers["x-request-id"] = request_id
        logger.info(
            "http_request",
            method=request.method,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    return app


app = create_app()
