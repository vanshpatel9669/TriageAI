"""Stage 1: ingestion — normalize the ticket before anything else touches it."""

from __future__ import annotations

import time

from triageai.application.state import PipelineState
from triageai.domain.models import Ticket


class IngestionNode:
    async def __call__(self, state: PipelineState) -> dict:
        ticket = state["ticket"]
        normalized = Ticket(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject.strip(),
            body=ticket.body.strip(),
            requester_email=ticket.requester_email.strip().lower(),
            system=ticket.system.strip() if ticket.system else None,
            created_at=ticket.created_at,
        )
        return {"ticket": normalized, "started_at": time.monotonic()}
