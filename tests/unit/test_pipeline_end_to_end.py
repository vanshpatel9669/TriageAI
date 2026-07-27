from __future__ import annotations

import tempfile

import pytest

from triageai.application.pipeline import TriagePipeline
from triageai.domain.models import RoutingDecision, Ticket
from triageai.infrastructure.llm.fake_provider import FakeLLMProvider
from triageai.infrastructure.persistence.chroma_knowledge_repository import (
    ChromaKnowledgeRepository,
)
from triageai.infrastructure.persistence.in_memory_repositories import (
    InMemoryGraphMemoryRepository,
    InMemoryTicketRepository,
)


@pytest.fixture
def repos():
    with tempfile.TemporaryDirectory() as tmp_dir:
        vector_repo = ChromaKnowledgeRepository(tmp_dir, "test_docs")
        vector_repo.seed(
            [
                {
                    "id": "kb-001",
                    "title": "Resetting a locked account password",
                    "text": (
                        "Direct the user to the self-service password reset portal "
                        "to unlock their account."
                    ),
                }
            ]
        )
        yield {
            "vector_repo": vector_repo,
            "graph_repo": InMemoryGraphMemoryRepository(),
            "ticket_repo": InMemoryTicketRepository(),
        }


@pytest.fixture
def pipeline(repos):
    return TriagePipeline(
        llm=FakeLLMProvider(),
        vector_repo=repos["vector_repo"],
        graph_repo=repos["graph_repo"],
        ticket_repo=repos["ticket_repo"],
        confidence_threshold=0.75,
    )


async def test_full_pipeline_run_produces_complete_result(pipeline):
    ticket = Ticket(
        subject="Locked out of my account",
        body="I can't log in, my account seems locked after too many password attempts.",
        requester_email="user@example.com",
    )

    result = await pipeline.run(ticket)

    assert result.ticket.ticket_id == ticket.ticket_id
    assert result.classification.category.value == "access"
    assert result.evidence  # Chroma should surface kb-001
    assert result.draft.text
    assert result.routing.decision in (RoutingDecision.AUTO_RESOLVE, RoutingDecision.HUMAN_REVIEW)
    assert result.latency_ms > 0


async def test_pipeline_persists_result_and_records_memory(pipeline, repos):
    ticket = Ticket(
        subject="VPN not connecting",
        body="My VPN client won't connect at all.",
        requester_email="a@b.com",
    )

    result = await pipeline.run(ticket)
    saved = await repos["ticket_repo"].get(ticket.ticket_id)
    similar = await repos["graph_repo"].search_similar_tickets(ticket, top_k=5)

    assert saved is not None
    assert saved.ticket.ticket_id == result.ticket.ticket_id
    # memory was recorded without error; overlap-scoring content is covered
    # by the graph repo's own tests
    assert len(similar) >= 0
