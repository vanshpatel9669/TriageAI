"""FastAPI dependency-injection wiring.

This is the one place that knows about concrete infrastructure classes.
Everything above it (application, domain) only knows about the Protocols in
domain/interfaces.py. Swapping the in-memory graph/ticket repos for the real
Neo4j-backed ones is controlled entirely by TRIAGEAI_USE_IN_MEMORY_GRAPH —
no other file changes.
"""

from __future__ import annotations

from functools import lru_cache

from triageai.application.pipeline import TriagePipeline
from triageai.config import Settings, get_settings
from triageai.domain.interfaces import AuditLogRepository, GraphMemoryRepository, TicketRepository
from triageai.domain.routing_strategies.registry import build_routing_strategy
from triageai.infrastructure.llm.provider_factory import build_llm_provider
from triageai.infrastructure.persistence.chroma_knowledge_repository import (
    ChromaKnowledgeRepository,
)
from triageai.infrastructure.persistence.in_memory_repositories import (
    InMemoryAuditLogRepository,
    InMemoryGraphMemoryRepository,
    InMemoryTicketRepository,
)
from triageai.infrastructure.persistence.neo4j_repositories import (
    Neo4jAuditLogRepository,
    Neo4jConnection,
    Neo4jGraphMemoryRepository,
    Neo4jTicketRepository,
)

_SAMPLE_KB_DOCS = [
    {
        "id": "kb-001",
        "title": "Resetting a locked account password",
        "text": (
            "If a user is locked out after failed login attempts, direct them to the "
            "self-service password reset portal at reset.internal.example. Accounts "
            "unlock automatically after 30 minutes, or an admin can unlock immediately "
            "via the IAM console."
        ),
    },
    {
        "id": "kb-002",
        "title": "VPN connection failures",
        "text": (
            "VPN connection failures are most often caused by an expired client "
            "certificate. Ask the user to open the VPN client, select 'Renew "
            "Certificate', and restart the client. If the issue persists after "
            "renewal, escalate to network engineering."
        ),
    },
    {
        "id": "kb-003",
        "title": "Requesting new hardware",
        "text": (
            "New laptop or monitor requests go through the hardware request form in "
            "the internal portal. Standard turnaround is 5 business days. Urgent "
            "hardware failures (device won't power on) should be flagged as high "
            "urgency for same-day loaner issuance."
        ),
    },
]


@lru_cache
def get_cached_settings() -> Settings:
    return get_settings()


@lru_cache
def get_vector_repo() -> ChromaKnowledgeRepository:
    settings = get_cached_settings()
    repo = ChromaKnowledgeRepository(settings.chroma_persist_dir, settings.chroma_collection)
    repo.seed(_SAMPLE_KB_DOCS)
    return repo


@lru_cache
def get_neo4j_connection() -> Neo4jConnection:
    settings = get_cached_settings()
    return Neo4jConnection(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password or "",
    )


@lru_cache
def get_graph_repo() -> GraphMemoryRepository:
    settings = get_cached_settings()
    if settings.use_in_memory_graph:
        return InMemoryGraphMemoryRepository()
    return Neo4jGraphMemoryRepository(get_neo4j_connection())


@lru_cache
def get_ticket_repo() -> TicketRepository:
    settings = get_cached_settings()
    if settings.use_in_memory_graph:
        return InMemoryTicketRepository()
    return Neo4jTicketRepository(get_neo4j_connection())


@lru_cache
def get_audit_repo() -> AuditLogRepository:
    settings = get_cached_settings()
    if settings.use_in_memory_graph:
        return InMemoryAuditLogRepository()
    return Neo4jAuditLogRepository(get_neo4j_connection())


@lru_cache
def get_pipeline() -> TriagePipeline:
    settings = get_cached_settings()
    llm = build_llm_provider(settings)
    routing_strategy = build_routing_strategy(
        settings.routing_strategy, confidence_threshold=settings.confidence_threshold
    )
    return TriagePipeline(
        llm=llm,
        vector_repo=get_vector_repo(),
        graph_repo=get_graph_repo(),
        ticket_repo=get_ticket_repo(),
        routing_strategy=routing_strategy,
        audit_repo=get_audit_repo(),
    )


async def shutdown_resources() -> None:
    """Called from the FastAPI lifespan handler on shutdown."""
    settings = get_cached_settings()
    if not settings.use_in_memory_graph:
        await get_neo4j_connection().close()
