from __future__ import annotations

from tests.integration.conftest import OLLAMA_MODEL, requires_ollama
from triageai.domain.models import EvidenceSnippet, Ticket
from triageai.infrastructure.llm.ollama_provider import OllamaLLMProvider


@requires_ollama
async def test_classify_returns_a_valid_classification():
    provider = OllamaLLMProvider(model=OLLAMA_MODEL)
    ticket = Ticket(
        subject="Locked out of my account",
        body="I typed my password wrong too many times and now I'm locked out.",
        requester_email="user@example.com",
    )

    classification = await provider.classify(ticket)

    assert classification.category.value in (
        "access",
        "hardware",
        "software",
        "network",
        "account",
        "other",
    )
    assert 0.0 <= classification.confidence <= 1.0


@requires_ollama
async def test_draft_response_cites_provided_evidence():
    provider = OllamaLLMProvider(model=OLLAMA_MODEL)
    ticket = Ticket(
        subject="Locked out",
        body="Can't log in, account locked after failed attempts.",
        requester_email="user@example.com",
    )
    classification = await provider.classify(ticket)
    evidence = [
        EvidenceSnippet(
            source_id="kb-access-01",
            source_type="knowledge_doc",
            title="Resetting a locked account password",
            text="Direct the user to the self-service password reset portal.",
            similarity=0.9,
        )
    ]

    draft = await provider.draft_response(ticket, classification, evidence)

    assert isinstance(draft, str)
    assert len(draft) > 0
