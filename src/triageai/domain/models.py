"""Core domain models. No framework or infrastructure imports allowed here."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TicketCategory(StrEnum):
    ACCESS = "access"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    ACCOUNT = "account"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingDecision(StrEnum):
    AUTO_RESOLVE = "auto_resolve"
    HUMAN_REVIEW = "human_review"


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TicketInput(BaseModel):
    """What a caller submits to the API."""

    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=10_000)
    requester_email: str
    system: str | None = Field(default=None, max_length=200)


class Ticket(BaseModel):
    """A normalized, persisted ticket."""

    ticket_id: str = Field(default_factory=_new_id)
    subject: str
    body: str
    requester_email: str
    system: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Classification(BaseModel):
    category: TicketCategory
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class EvidenceSnippet(BaseModel):
    source_id: str
    source_type: str  # "knowledge_doc" | "similar_ticket"
    title: str
    text: str
    similarity: float = Field(ge=0.0, le=1.0)


class GroundednessCheck(BaseModel):
    """Result of checking whether a drafted claim is supported by evidence."""

    is_grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    supported_claim_ratio: float = Field(ge=0.0, le=1.0)


class DraftResponse(BaseModel):
    text: str
    cited_evidence_ids: list[str]
    groundedness: GroundednessCheck


class RoutingResult(BaseModel):
    decision: RoutingDecision
    reasons: list[str]


class TriageResult(BaseModel):
    """The final, complete output of one pipeline run."""

    ticket: Ticket
    classification: Classification
    evidence: list[EvidenceSnippet]
    draft: DraftResponse
    routing: RoutingResult
    latency_ms: float
