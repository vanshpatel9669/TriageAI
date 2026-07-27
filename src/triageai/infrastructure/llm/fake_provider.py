"""Deterministic fake LLM provider used by unit tests.

Unit tests must never depend on a real model — real or local. This keeps the
suite fast, deterministic, and runnable with zero external services. Real
providers (Ollama, OpenAI) are only exercised by integration tests.
"""

from __future__ import annotations

from triageai.domain.models import Classification, EvidenceSnippet, Ticket, TicketCategory, Urgency


class FakeLLMProvider:
    """Rule-based stand-in satisfying the LLMProvider protocol."""

    name = "fake"

    async def classify(self, ticket: Ticket) -> Classification:
        text = f"{ticket.subject} {ticket.body}".lower()
        if any(w in text for w in ("password", "login", "locked out", "access")):
            category = TicketCategory.ACCESS
        elif any(w in text for w in ("laptop", "monitor", "keyboard", "printer")):
            category = TicketCategory.HARDWARE
        elif any(w in text for w in ("vpn", "wifi", "network", "dns")):
            category = TicketCategory.NETWORK
        elif any(w in text for w in ("crash", "error", "bug", "install")):
            category = TicketCategory.SOFTWARE
        else:
            category = TicketCategory.OTHER

        is_urgent = any(w in text for w in ("urgent", "down", "critical"))
        urgency = Urgency.HIGH if is_urgent else Urgency.MEDIUM
        return Classification(
            category=category,
            urgency=urgency,
            confidence=0.82,
            rationale=f"Keyword match against category '{category.value}'.",
        )

    async def draft_response(
        self, ticket: Ticket, classification: Classification, evidence: list[EvidenceSnippet]
    ) -> str:
        if not evidence:
            return (
                f"Thanks for reaching out about '{ticket.subject}'. "
                "I don't have a matching knowledge-base article yet, so this has been "
                "routed to a human agent for review."
            )
        lead = evidence[0]
        return (
            f"Thanks for reaching out about '{ticket.subject}'. Based on "
            f"[{lead.source_id}] {lead.title}: {lead.text[:200]}"
        )
