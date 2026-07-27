"""FastAPI dependency-injection wiring.

This is the one place that knows about concrete infrastructure classes.
Everything above it (application, domain) only knows about the Protocols in
domain/interfaces.py. Swapping the in-memory graph repo for a real Neo4j one
in Phase 3 means changing exactly this file.
"""

from __future__ import annotations

from functools import lru_cache

from triageai.application.pipeline import TriagePipeline
from triageai.config import Settings, get_settings
from triageai.infrastructure.llm.provider_factory import build_llm_provider
from triageai.infrastructure.persistence.chroma_knowledge_repository import (
    ChromaKnowledgeRepository,
)
from triageai.infrastructure.persistence.in_memory_repositories import (
    InMemoryGraphMemoryRepository,
    InMemoryTicketRepository,
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
def get_graph_repo() -> InMemoryGraphMemoryRepository:
    return InMemoryGraphMemoryRepository()


@lru_cache
def get_ticket_repo() -> InMemoryTicketRepository:
    return InMemoryTicketRepository()


@lru_cache
def get_pipeline() -> TriagePipeline:
    settings = get_cached_settings()
    llm = build_llm_provider(settings)
    return TriagePipeline(
        llm=llm,
        vector_repo=get_vector_repo(),
        graph_repo=get_graph_repo(),
        ticket_repo=get_ticket_repo(),
        confidence_threshold=settings.confidence_threshold,
    )
