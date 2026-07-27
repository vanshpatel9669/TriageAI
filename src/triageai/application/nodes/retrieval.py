"""Stage 3: retrieval — semantic KB search (vector store) + similar-ticket
graph lookup (Neo4j / in-memory stand-in), merged and ranked by similarity.
"""

from __future__ import annotations

from triageai.application.state import PipelineState
from triageai.domain.interfaces import GraphMemoryRepository, VectorSearchRepository


class RetrievalNode:
    def __init__(
        self,
        vector_repo: VectorSearchRepository,
        graph_repo: GraphMemoryRepository,
        top_k_docs: int = 3,
        top_k_similar_tickets: int = 2,
    ) -> None:
        self._vector_repo = vector_repo
        self._graph_repo = graph_repo
        self._top_k_docs = top_k_docs
        self._top_k_similar_tickets = top_k_similar_tickets

    async def __call__(self, state: PipelineState) -> dict:
        ticket = state["ticket"]
        query = f"{ticket.subject} {ticket.body}"

        docs = await self._vector_repo.search_knowledge_docs(query, self._top_k_docs)
        similar_tickets = await self._graph_repo.search_similar_tickets(
            ticket, self._top_k_similar_tickets
        )

        evidence = sorted(docs + similar_tickets, key=lambda e: e.similarity, reverse=True)
        return {"evidence": evidence}
