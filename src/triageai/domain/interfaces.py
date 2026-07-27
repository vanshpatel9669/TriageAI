"""Abstract boundaries (Repository Pattern + provider abstraction).

Infrastructure implementations depend on these Protocols; the application
layer (pipeline nodes) depends only on these Protocols too. Nothing in this
file imports from `infrastructure`. That's the Clean Architecture dependency
rule: dependencies point inward, never outward.
"""

from __future__ import annotations

from typing import Protocol

from triageai.domain.models import (
    Classification,
    DraftResponse,
    EvidenceSnippet,
    RoutingResult,
    Ticket,
    TriageResult,
)


class VectorSearchRepository(Protocol):
    """Semantic similarity search over knowledge-base documents (Chroma)."""

    async def search_knowledge_docs(self, query: str, top_k: int) -> list[EvidenceSnippet]: ...


class GraphMemoryRepository(Protocol):
    """Knowledge-graph relationships and ticket memory (Neo4j).

    Kept separate from VectorSearchRepository (Interface Segregation): a
    caller that only needs semantic doc search shouldn't have to depend on
    graph-memory methods it never calls, and either side can be swapped
    independently.
    """

    async def search_similar_tickets(self, ticket: Ticket, top_k: int) -> list[EvidenceSnippet]: ...

    async def record_ticket_relationship(
        self, ticket: Ticket, classification: Classification
    ) -> None: ...


class TicketRepository(Protocol):
    """Persistence for ticket workflow state and history."""

    async def save(self, result: TriageResult) -> None: ...

    async def get(self, ticket_id: str) -> TriageResult | None: ...


class LLMProvider(Protocol):
    """Uniform interface every LLM backend (Ollama, OpenAI, ...) implements.

    Adding a new provider means writing one adapter class in
    `infrastructure/llm/` that satisfies this Protocol — the pipeline nodes
    never change.
    """

    name: str

    async def classify(self, ticket: Ticket) -> Classification: ...

    async def draft_response(
        self, ticket: Ticket, classification: Classification, evidence: list[EvidenceSnippet]
    ) -> str: ...


class RoutingStrategy(Protocol):
    """Uniform interface every routing policy implements.

    Adding a new routing strategy means writing one class satisfying this
    Protocol and registering it in
    `domain/routing_strategies/registry.py` — `RoutingNode`
    (application/nodes/routing.py) never changes. Pure decision logic, no
    I/O, so this stays in `domain/` rather than `infrastructure/`.
    """

    name: str

    def evaluate(self, classification: Classification, draft: DraftResponse) -> RoutingResult: ...
