"""Shared fixtures for integration tests.

Every fixture here SKIPS (with an explicit reason) rather than silently
passing when its external dependency isn't reachable. A skipped integration
suite is an honest, visible signal; a green suite that quietly tested nothing
is not.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from triageai.infrastructure.persistence.neo4j_repositories import (
    Neo4jConnection,
    ensure_indexes,
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "triageai-dev-password"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def neo4j_reachable() -> bool:
    return _port_open("localhost", 7687)


def ollama_reachable() -> bool:
    if not _port_open("localhost", 11434):
        return False
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.0)
        tags = response.json().get("models", [])
        return any(m.get("name", "").startswith(OLLAMA_MODEL.split(":")[0]) for m in tags)
    except (httpx.HTTPError, ValueError):
        return False


requires_neo4j = pytest.mark.skipif(
    not neo4j_reachable(),
    reason=f"Neo4j not reachable at {NEO4J_URI} — run `docker compose up neo4j` first",
)

requires_ollama = pytest.mark.skipif(
    not ollama_reachable(),
    reason=(
        f"Ollama not reachable at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL} pulled — "
        "run `ollama serve` and `ollama pull llama3.1:8b` first"
    ),
)


@pytest_asyncio.fixture
async def neo4j_connection() -> AsyncIterator[Neo4jConnection]:
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    await connection.verify_connectivity()
    await ensure_indexes(connection)
    yield connection
    # Clean up only the test-tagged nodes this suite creates, never a blanket
    # wipe of whatever else might be in the database. Two tags are needed:
    # ticket_id STARTS WITH 'inttest-' catches tickets built directly against
    # the repository (test_neo4j_repositories.py sets that prefix itself),
    # but tickets created through the real API get a server-generated UUID
    # instead — those are tagged by requester_email instead (see
    # test_api_full_stack.py). Missing the second tag was a real gap: it let
    # API-created test tickets leak into later runs' similar-ticket search
    # results, found while manually smoke-testing the full stack.
    async with connection.session() as session:
        await session.run(  # type: ignore[attr-defined]
            "MATCH (t:Ticket) "
            "WHERE t.ticket_id STARTS WITH 'inttest-' OR t.requester_email = 'inttest@example.com' "
            "DETACH DELETE t"
        )
    await connection.close()
