"""Verifies /readyz actually checks Neo4j connectivity when real Neo4j is
configured, rather than always reporting ready regardless of reachability
(a real gap found and fixed while writing docs/api.md).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_neo4j_enabled(monkeypatch):
    monkeypatch.setenv("TRIAGEAI_LLM_PROVIDER", "fake")
    monkeypatch.setenv("TRIAGEAI_USE_IN_MEMORY_GRAPH", "false")

    from triageai.infrastructure.api import dependencies

    for cached in (
        dependencies.get_cached_settings,
        dependencies.get_vector_repo,
        dependencies.get_neo4j_connection,
        dependencies.get_graph_repo,
        dependencies.get_ticket_repo,
        dependencies.get_pipeline,
    ):
        cached.cache_clear()

    yield dependencies

    for cached in (
        dependencies.get_cached_settings,
        dependencies.get_vector_repo,
        dependencies.get_neo4j_connection,
        dependencies.get_graph_repo,
        dependencies.get_ticket_repo,
        dependencies.get_pipeline,
    ):
        cached.cache_clear()


def test_readyz_returns_ready_when_using_in_memory_graph(monkeypatch):
    monkeypatch.setenv("TRIAGEAI_LLM_PROVIDER", "fake")
    monkeypatch.setenv("TRIAGEAI_USE_IN_MEMORY_GRAPH", "true")

    from triageai.infrastructure.api import dependencies

    dependencies.get_cached_settings.cache_clear()

    from triageai.infrastructure.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    dependencies.get_cached_settings.cache_clear()


def test_readyz_returns_503_when_neo4j_unreachable(app_with_neo4j_enabled, monkeypatch):
    class _AlwaysFailsConnection:
        async def verify_connectivity(self) -> None:
            raise ConnectionError("simulated: Neo4j not reachable")

    from fastapi import FastAPI

    from triageai.infrastructure.api import routes

    # routes.py imports get_neo4j_connection by name, so the patch target is
    # the name as bound in routes' own namespace, not the dependencies module
    # attribute (patching the latter wouldn't affect routes' already-bound
    # reference — a classic monkeypatch-an-imported-name gotcha).
    monkeypatch.setattr(routes, "get_neo4j_connection", lambda: _AlwaysFailsConnection())

    # Build the app directly from the router, bypassing create_app()'s
    # lifespan (which would itself fail startup against the same
    # unreachable Neo4j) — this isolates the /readyz route logic itself.
    app = FastAPI()
    app.include_router(routes.router)

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert "Neo4j not reachable" in response.json()["detail"]
