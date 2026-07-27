from __future__ import annotations

from triageai.application.nodes.ingestion import IngestionNode
from triageai.domain.models import Ticket


async def test_ingestion_strips_whitespace_and_lowercases_email():
    node = IngestionNode()
    ticket = Ticket(
        subject="  Locked out  ",
        body="  help please  ",
        requester_email="  User@Example.com  ",
    )
    result = await node({"ticket": ticket})

    normalized = result["ticket"]
    assert normalized.subject == "Locked out"
    assert normalized.body == "help please"
    assert normalized.requester_email == "user@example.com"
    assert "started_at" in result


async def test_ingestion_preserves_ticket_id():
    node = IngestionNode()
    ticket = Ticket(subject="a", body="b", requester_email="x@example.com")
    result = await node({"ticket": ticket})
    assert result["ticket"].ticket_id == ticket.ticket_id
