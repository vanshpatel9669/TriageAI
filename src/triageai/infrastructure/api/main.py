"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response

from triageai.config import get_settings
from triageai.infrastructure.api.dependencies import get_cached_settings, shutdown_resources
from triageai.infrastructure.api.routes import router
from triageai.infrastructure.observability import configure_tracing, instrument_app
from triageai.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_cached_settings()
    if not settings.use_in_memory_graph:
        from triageai.infrastructure.api.dependencies import get_neo4j_connection
        from triageai.infrastructure.persistence.neo4j_repositories import ensure_indexes

        connection = get_neo4j_connection()
        await connection.verify_connectivity()
        await ensure_indexes(connection)
        logger.info("neo4j_ready", uri=settings.neo4j_uri)

    yield

    await shutdown_resources()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(
        settings.service_name, settings.otel_console_export, settings.otel_exporter_otlp_endpoint
    )

    app = FastAPI(title="TriageAI", version="0.1.0", lifespan=_lifespan)
    app.include_router(router)
    instrument_app(app)

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
