"""Exercises the real FastAPI app wired to real Neo4j (fake LLM, so this
test isolates the Neo4j integration from LLM availability/quality — see
test_ollama_provider.py for the LLM side).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import NEO4J_PASSWORD, requires_neo4j


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("TRIAGEAI_LLM_PROVIDER", "fake")
    monkeypatch.setenv("TRIAGEAI_USE_IN_MEMORY_GRAPH", "false")
    monkeypatch.setenv("TRIAGEAI_NEO4J_PASSWORD", NEO4J_PASSWORD)

    # Clear every lru_cache singleton so this test's env vars actually apply,
    # instead of reusing whatever a prior test in the same process cached.
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

    from triageai.infrastructure.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client

    for cached in (
        dependencies.get_cached_settings,
        dependencies.get_vector_repo,
        dependencies.get_neo4j_connection,
        dependencies.get_graph_repo,
        dependencies.get_ticket_repo,
        dependencies.get_pipeline,
    ):
        cached.cache_clear()


@requires_neo4j
def test_full_ticket_lifecycle_with_real_neo4j(api_client):
    response = api_client.post(
        "/tickets",
        json={
            "subject": "Locked out of my account",
            "body": "I can't log in, too many failed password attempts.",
            "requester_email": "inttest@example.com",
        },
    )
    assert response.status_code == 200
    body = response.json()
    ticket_id = body["ticket"]["ticket_id"]

    fetched = api_client.get(f"/tickets/{ticket_id}")
    assert fetched.status_code == 200
    assert fetched.json()["ticket"]["ticket_id"] == ticket_id


@requires_neo4j
def test_readyz_verifies_neo4j_connectivity(api_client):
    response = api_client.get("/readyz")
    assert response.status_code == 200
