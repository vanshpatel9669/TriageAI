from __future__ import annotations

from triageai.application.nodes.retrieval import RetrievalNode
from triageai.domain.models import EvidenceSnippet, Ticket


class _StubVectorRepo:
    def __init__(self, results: list[EvidenceSnippet]) -> None:
        self._results = results

    async def search_knowledge_docs(self, query: str, top_k: int) -> list[EvidenceSnippet]:
        return self._results[:top_k]


class _StubGraphRepo:
    def __init__(self, results: list[EvidenceSnippet]) -> None:
        self._results = results

    async def search_similar_tickets(self, ticket: Ticket, top_k: int) -> list[EvidenceSnippet]:
        return self._results[:top_k]

    async def record_ticket_relationship(self, ticket, classification) -> None:
        pass


async def test_retrieval_merges_and_ranks_by_similarity(sample_ticket):
    doc = EvidenceSnippet(
        source_id="kb-001", source_type="knowledge_doc", title="Doc", text="text", similarity=0.5
    )
    similar_ticket = EvidenceSnippet(
        source_id="t-42",
        source_type="similar_ticket",
        title="Past ticket",
        text="text",
        similarity=0.9,
    )
    node = RetrievalNode(
        vector_repo=_StubVectorRepo([doc]),
        graph_repo=_StubGraphRepo([similar_ticket]),
    )

    result = await node({"ticket": sample_ticket})
    evidence = result["evidence"]

    assert len(evidence) == 2
    assert evidence[0].source_id == "t-42"  # higher similarity first


async def test_retrieval_handles_no_results(sample_ticket):
    node = RetrievalNode(vector_repo=_StubVectorRepo([]), graph_repo=_StubGraphRepo([]))
    result = await node({"ticket": sample_ticket})
    assert result["evidence"] == []
