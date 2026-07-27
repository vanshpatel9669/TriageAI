"""Exercises the real FastAPI app wired to real Neo4j (fake LLM, so this
test isolates the Neo4j integration from LLM availability/quality — see
test_ollama_provider.py for the LLM side).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, requires_neo4j
from triageai.infrastructure.persistence.neo4j_repositories import Neo4jConnection

# Every ticket this file creates through the real API uses this address, so
# teardown can find and remove them by requester_email — the ticket_id itself
# is server-generated, not the client-chosen 'inttest-' prefix
# tests/integration/test_neo4j_repositories.py uses.
TEST_REQUESTER_EMAIL = "inttest@example.com"


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
        dependencies.get_audit_repo,
        dependencies.get_pipeline,
    ):
        cached.cache_clear()

    from triageai.infrastructure.api.main import create_app

    app = create_app()
    # Clean up INSIDE the TestClient context, before it exits — exiting
    # triggers the app's lifespan shutdown, which closes the DI-managed
    # Neo4j driver (shutdown_resources() in dependencies.py). A cleanup
    # attempt after that point hits a closed driver and errors instead of
    # deleting anything, which is exactly what happened the first time this
    # was written: teardown silently failed and test data kept leaking.
    # Using a separate, independently-owned connection here sidesteps that
    # ordering entirely rather than trying to race the app's own lifecycle.
    with TestClient(app) as client:
        yield client
        _delete_test_tickets_sync()

    for cached in (
        dependencies.get_cached_settings,
        dependencies.get_vector_repo,
        dependencies.get_neo4j_connection,
        dependencies.get_graph_repo,
        dependencies.get_ticket_repo,
        dependencies.get_audit_repo,
        dependencies.get_pipeline,
    ):
        cached.cache_clear()


def _delete_test_tickets_sync() -> None:
    import asyncio

    # asyncio.get_event_loop() depends on ambient per-thread loop state that
    # other async fixtures/tests running earlier in the same pytest session
    # can leave in a state where no implicit loop exists to fall back on —
    # raises "There is no current event loop in thread 'MainThread'" in that
    # case, found by running this file alongside test_neo4j_repositories.py
    # rather than alone. asyncio.run() creates and tears down its own loop
    # every call, so it doesn't depend on that state at all.
    asyncio.run(_delete_test_tickets())


async def _delete_test_tickets() -> None:
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    async with connection.session() as session:
        # Tickets created through the real API now also get an AuditEvent
        # (pipeline.run() records one on every run) — deleting only the
        # Ticket node via DETACH DELETE would leave that AuditEvent orphaned,
        # same gap just fixed in conftest.py's neo4j_connection teardown.
        await session.run(
            "MATCH (t:Ticket) WHERE t.requester_email = $email "
            "OPTIONAL MATCH (t)-[:AUDITED_BY]->(a:AuditEvent) "
            "DETACH DELETE t, a",
            email=TEST_REQUESTER_EMAIL,
        )
    await connection.close()


@requires_neo4j
def test_full_ticket_lifecycle_with_real_neo4j(api_client):
    response = api_client.post(
        "/tickets",
        json={
            "subject": "Locked out of my account",
            "body": "I can't log in, too many failed password attempts.",
            "requester_email": TEST_REQUESTER_EMAIL,
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
