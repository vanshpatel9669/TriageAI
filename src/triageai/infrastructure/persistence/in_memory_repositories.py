"""In-memory implementations used until Neo4j is wired in (Phase 3) and in
unit tests. Both satisfy the same Protocols the real Neo4j-backed classes
will satisfy, so swapping them in `infrastructure/api/dependencies.py` is the
only change Phase 3 needs to make here.
"""

from __future__ import annotations

from triageai.domain.models import Classification, EvidenceSnippet, Ticket, TriageResult


class InMemoryGraphMemoryRepository:
    """Satisfies GraphMemoryRepository with a plain dict. No persistence across restarts."""

    def __init__(self) -> None:
        self._history: list[tuple[Ticket, Classification]] = []

    async def search_similar_tickets(self, ticket: Ticket, top_k: int) -> list[EvidenceSnippet]:
        query_words = set(f"{ticket.subject} {ticket.body}".lower().split())
        scored: list[tuple[float, Ticket]] = []
        for past_ticket, _classification in self._history:
            past_words = set(f"{past_ticket.subject} {past_ticket.body}".lower().split())
            if not past_words:
                continue
            overlap = len(query_words & past_words) / len(query_words | past_words)
            if overlap > 0:
                scored.append((overlap, past_ticket))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            EvidenceSnippet(
                source_id=past.ticket_id,
                source_type="similar_ticket",
                title=past.subject,
                text=past.body,
                similarity=round(score, 4),
            )
            for score, past in scored[:top_k]
        ]

    async def record_ticket_relationship(
        self, ticket: Ticket, classification: Classification
    ) -> None:
        self._history.append((ticket, classification))


class InMemoryTicketRepository:
    """Satisfies TicketRepository with a plain dict."""

    def __init__(self) -> None:
        self._store: dict[str, TriageResult] = {}

    async def save(self, result: TriageResult) -> None:
        self._store[result.ticket.ticket_id] = result

    async def get(self, ticket_id: str) -> TriageResult | None:
        return self._store.get(ticket_id)
