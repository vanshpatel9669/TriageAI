"""HTTP routes. Thin — all logic lives in the pipeline/domain layers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from triageai.application.pipeline import TriagePipeline
from triageai.domain.models import Ticket, TicketInput, TriageResult
from triageai.infrastructure.api.dependencies import get_pipeline, get_ticket_repo
from triageai.infrastructure.persistence.in_memory_repositories import InMemoryTicketRepository
from triageai.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

PipelineDep = Annotated[TriagePipeline, Depends(get_pipeline)]
TicketRepoDep = Annotated[InMemoryTicketRepository, Depends(get_ticket_repo)]


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(pipeline: PipelineDep) -> dict:
    return {"status": "ready"}


@router.post("/tickets", response_model=TriageResult)
async def create_ticket(payload: TicketInput, pipeline: PipelineDep) -> TriageResult:
    ticket = Ticket(
        subject=payload.subject,
        body=payload.body,
        requester_email=payload.requester_email,
        system=payload.system,
    )
    logger.info("ticket_received", ticket_id=ticket.ticket_id)
    result = await pipeline.run(ticket)
    logger.info(
        "ticket_triaged",
        ticket_id=ticket.ticket_id,
        category=result.classification.category.value,
        routing=result.routing.decision.value,
        latency_ms=result.latency_ms,
    )
    return result


@router.get("/tickets/{ticket_id}", response_model=TriageResult)
async def get_ticket(ticket_id: str, ticket_repo: TicketRepoDep) -> TriageResult:
    result = await ticket_repo.get(ticket_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return result
