"""LangGraph state schema threaded through all five pipeline nodes."""

from __future__ import annotations

from typing import TypedDict

from triageai.domain.models import (
    Classification,
    DraftResponse,
    EvidenceSnippet,
    RoutingResult,
    Ticket,
)


class PipelineState(TypedDict, total=False):
    ticket: Ticket
    classification: Classification
    evidence: list[EvidenceSnippet]
    draft: DraftResponse
    routing: RoutingResult
    started_at: float
