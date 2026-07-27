"""Neo4j-backed implementations of GraphMemoryRepository and TicketRepository.

Neo4j stores both the ticket workflow history and knowledge-graph
relationships as graph nodes/edges, satisfying the same Protocols the
in-memory stand-ins satisfy (see in_memory_repositories.py) — swapping one
for the other in infrastructure/api/dependencies.py is the only change
needed anywhere in the codebase.

Graph model:
  (:Ticket {ticket_id, subject, body, created_at})
    -[:CLASSIFIED_AS {confidence}]->(:Category {name})
  (:Ticket)-[:RESULTED_IN]->(:TriageOutcome {routing, latency_ms, ...})
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from neo4j import AsyncDriver, AsyncGraphDatabase
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from triageai.domain.models import (
    AuditEvent,
    Classification,
    DraftResponse,
    EvidenceSnippet,
    GroundednessCheck,
    RoutingDecision,
    RoutingResult,
    Ticket,
    TicketCategory,
    TriageResult,
    Urgency,
)

_neo4j_retry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    reraise=True,
)


class Neo4jConnection:
    """Owns the driver lifecycle. One instance shared across both repositories."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[object]:
        async with self._driver.session() as session:
            yield session

    async def verify_connectivity(self) -> None:
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        await self._driver.close()


class Neo4jGraphMemoryRepository:
    """Satisfies GraphMemoryRepository via Neo4j full-text search over past tickets."""

    def __init__(self, connection: Neo4jConnection) -> None:
        self._connection = connection

    @_neo4j_retry
    async def search_similar_tickets(self, ticket: Ticket, top_k: int) -> list[EvidenceSnippet]:
        query_text = f"{ticket.subject} {ticket.body}"
        cypher = """
        CALL db.index.fulltext.queryNodes('ticketFullText', $query_text)
        YIELD node, score
        WHERE node.ticket_id <> $exclude_id
        RETURN node.ticket_id AS ticket_id, node.subject AS subject,
               node.body AS body, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        async with self._connection.session() as session:
            result = await session.run(  # type: ignore[attr-defined]
                cypher, query_text=query_text, exclude_id=ticket.ticket_id, top_k=top_k
            )
            records = [record async for record in result]

        if not records:
            return []
        max_score = max(r["score"] for r in records) or 1.0
        return [
            EvidenceSnippet(
                source_id=r["ticket_id"],
                source_type="similar_ticket",
                title=r["subject"],
                text=r["body"],
                similarity=round(min(1.0, r["score"] / max_score), 4),
            )
            for r in records
        ]

    @_neo4j_retry
    async def record_ticket_relationship(
        self, ticket: Ticket, classification: Classification
    ) -> None:
        cypher = """
        MERGE (t:Ticket {ticket_id: $ticket_id})
        SET t.subject = $subject, t.body = $body, t.created_at = $created_at
        MERGE (c:Category {name: $category})
        MERGE (t)-[r:CLASSIFIED_AS]->(c)
        SET r.confidence = $confidence, r.urgency = $urgency
        """
        async with self._connection.session() as session:
            await session.run(  # type: ignore[attr-defined]
                cypher,
                ticket_id=ticket.ticket_id,
                subject=ticket.subject,
                body=ticket.body,
                created_at=ticket.created_at.isoformat(),
                category=classification.category.value,
                confidence=classification.confidence,
                urgency=classification.urgency.value,
            )


class Neo4jTicketRepository:
    """Satisfies TicketRepository — persists the full triage outcome on the Ticket node."""

    def __init__(self, connection: Neo4jConnection) -> None:
        self._connection = connection

    @_neo4j_retry
    async def save(self, result: TriageResult) -> None:
        cypher = """
        MERGE (t:Ticket {ticket_id: $ticket_id})
        SET t.subject = $subject, t.body = $body, t.requester_email = $requester_email,
            t.created_at = $created_at, t.result_json = $result_json
        """
        async with self._connection.session() as session:
            await session.run(  # type: ignore[attr-defined]
                cypher,
                ticket_id=result.ticket.ticket_id,
                subject=result.ticket.subject,
                body=result.ticket.body,
                requester_email=result.ticket.requester_email,
                created_at=result.ticket.created_at.isoformat(),
                result_json=result.model_dump_json(),
            )

    @_neo4j_retry
    async def get(self, ticket_id: str) -> TriageResult | None:
        cypher = "MATCH (t:Ticket {ticket_id: $ticket_id}) RETURN t.result_json AS result_json"
        async with self._connection.session() as session:
            result = await session.run(cypher, ticket_id=ticket_id)  # type: ignore[attr-defined]
            record = await result.single()

        if record is None or record["result_json"] is None:
            return None
        return _deserialize_triage_result(record["result_json"])


class Neo4jAuditLogRepository:
    """Satisfies AuditLogRepository — one CREATE per event (never MERGE/SET on
    an existing node), so a later ticket reprocessing can never silently
    overwrite an earlier decision's audit record the way Ticket.result_json
    can be overwritten on the Ticket node itself.
    """

    def __init__(self, connection: Neo4jConnection) -> None:
        self._connection = connection

    @_neo4j_retry
    async def record_event(self, event: AuditEvent) -> None:
        cypher = """
        MATCH (t:Ticket {ticket_id: $ticket_id})
        CREATE (a:AuditEvent {
            event_id: $event_id, occurred_at: $occurred_at,
            llm_provider: $llm_provider, routing_strategy: $routing_strategy,
            category: $category, confidence: $confidence,
            routing_decision: $routing_decision, routing_reasons: $routing_reasons,
            latency_ms: $latency_ms
        })
        CREATE (t)-[:AUDITED_BY]->(a)
        """
        async with self._connection.session() as session:
            await session.run(  # type: ignore[attr-defined]
                cypher,
                ticket_id=event.ticket_id,
                event_id=event.event_id,
                occurred_at=event.occurred_at.isoformat(),
                llm_provider=event.llm_provider,
                routing_strategy=event.routing_strategy,
                category=event.category.value,
                confidence=event.confidence,
                routing_decision=event.routing_decision.value,
                routing_reasons=event.routing_reasons,
                latency_ms=event.latency_ms,
            )


def _deserialize_triage_result(payload: str) -> TriageResult:
    data = json.loads(payload)
    return TriageResult(
        ticket=Ticket(**data["ticket"]),
        classification=Classification(
            category=TicketCategory(data["classification"]["category"]),
            urgency=Urgency(data["classification"]["urgency"]),
            confidence=data["classification"]["confidence"],
            rationale=data["classification"]["rationale"],
        ),
        evidence=[EvidenceSnippet(**e) for e in data["evidence"]],
        draft=DraftResponse(
            text=data["draft"]["text"],
            cited_evidence_ids=data["draft"]["cited_evidence_ids"],
            groundedness=GroundednessCheck(**data["draft"]["groundedness"]),
        ),
        routing=RoutingResult(
            decision=RoutingDecision(data["routing"]["decision"]),
            reasons=data["routing"]["reasons"],
        ),
        latency_ms=data["latency_ms"],
    )


async def ensure_indexes(connection: Neo4jConnection) -> None:
    """Create the full-text index used by similar-ticket search. Idempotent."""
    cypher = """
    CREATE FULLTEXT INDEX ticketFullText IF NOT EXISTS
    FOR (t:Ticket) ON EACH [t.subject, t.body]
    """
    async with connection.session() as session:
        await session.run(cypher)  # type: ignore[attr-defined]
