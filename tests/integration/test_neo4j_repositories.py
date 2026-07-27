from __future__ import annotations

from tests.integration.conftest import requires_neo4j
from triageai.domain.models import Classification, Ticket, TicketCategory, Urgency
from triageai.infrastructure.persistence.neo4j_repositories import (
    Neo4jGraphMemoryRepository,
    Neo4jTicketRepository,
)


def _ticket(suffix: str, subject: str, body: str) -> Ticket:
    return Ticket(
        ticket_id=f"inttest-{suffix}",
        subject=subject,
        body=body,
        requester_email="inttest@example.com",
    )


@requires_neo4j
async def test_record_and_search_similar_tickets(neo4j_connection):
    repo = Neo4jGraphMemoryRepository(neo4j_connection)
    past = _ticket("past-1", "VPN not connecting", "My VPN client fails to connect every time")
    classification = Classification(
        category=TicketCategory.NETWORK, urgency=Urgency.HIGH, confidence=0.9, rationale="r"
    )
    await repo.record_ticket_relationship(past, classification)

    new_ticket = _ticket("new-1", "VPN connection failing", "VPN client won't connect at all")
    results = await repo.search_similar_tickets(new_ticket, top_k=5)

    assert any(r.source_id == past.ticket_id for r in results)


@requires_neo4j
async def test_search_similar_tickets_excludes_self(neo4j_connection):
    repo = Neo4jGraphMemoryRepository(neo4j_connection)
    ticket = _ticket("self-1", "Locked out", "I can't log in, account locked")
    classification = Classification(
        category=TicketCategory.ACCESS, urgency=Urgency.HIGH, confidence=0.9, rationale="r"
    )
    await repo.record_ticket_relationship(ticket, classification)

    results = await repo.search_similar_tickets(ticket, top_k=5)

    assert all(r.source_id != ticket.ticket_id for r in results)


@requires_neo4j
async def test_ticket_repository_save_and_get_round_trip(neo4j_connection):
    import tempfile

    from triageai.application.pipeline import TriagePipeline
    from triageai.domain.routing_strategies.confidence_and_risk import (
        ConfidenceAndRiskRoutingStrategy,
    )
    from triageai.infrastructure.llm.fake_provider import FakeLLMProvider
    from triageai.infrastructure.persistence.chroma_knowledge_repository import (
        ChromaKnowledgeRepository,
    )
    from triageai.infrastructure.persistence.in_memory_repositories import (
        InMemoryGraphMemoryRepository,
    )

    ticket_repo = Neo4jTicketRepository(neo4j_connection)

    with tempfile.TemporaryDirectory() as tmp_dir:
        vector_repo = ChromaKnowledgeRepository(tmp_dir, "inttest_docs")
        pipeline = TriagePipeline(
            llm=FakeLLMProvider(),
            vector_repo=vector_repo,
            graph_repo=InMemoryGraphMemoryRepository(),
            ticket_repo=ticket_repo,
            routing_strategy=ConfidenceAndRiskRoutingStrategy(confidence_threshold=0.75),
        )
        ticket = _ticket("roundtrip-1", "Locked out of account", "Password locked me out, urgent")
        result = await pipeline.run(ticket)

    fetched = await ticket_repo.get(ticket.ticket_id)

    assert fetched is not None
    assert fetched.ticket.ticket_id == result.ticket.ticket_id
    assert fetched.classification.category == result.classification.category


@requires_neo4j
async def test_get_missing_ticket_returns_none(neo4j_connection):
    ticket_repo = Neo4jTicketRepository(neo4j_connection)
    result = await ticket_repo.get("inttest-does-not-exist")
    assert result is None
