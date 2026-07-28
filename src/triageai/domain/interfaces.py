"""Abstract boundaries (Repository Pattern + provider abstraction).

Infrastructure implementations depend on these Protocols; the application
layer (pipeline nodes) depends only on these Protocols too. Nothing in this
file imports from `infrastructure`. That's the Clean Architecture dependency
rule: dependencies point inward, never outward.
"""

from __future__ import annotations

from typing import Protocol

from triageai.domain.models import (
    AuditEvent,
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


class AuditLogRepository(Protocol):
    """Append-only audit trail, one record per triage decision.

    Distinct from TicketRepository: a Ticket's stored result can be
    overwritten if reprocessed, but an audit event is never updated or
    deleted through this interface — see domain/models.py's AuditEvent for
    why that's a different guarantee worth its own repository.
    """

    async def record_event(self, event: AuditEvent) -> None: ...


class LLMProvider(Protocol):
    """Uniform interface every LLM backend (Ollama, OpenAI, ...) implements.

    Adding a new provider means writing one adapter class in
    `infrastructure/llm/` that satisfies this Protocol — the pipeline nodes
    never change.
    """

    name: str
    # The specific model string (e.g. "llama3.1:8b"), distinct from `name`
    # (the provider, e.g. "ollama"). Two decisions both made by "ollama" but
    # different model versions are not the same decision for audit/
    # reproducibility purposes — this is what AuditEvent.model records.
    model: str

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
