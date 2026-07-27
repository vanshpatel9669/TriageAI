from __future__ import annotations

from triageai.application.nodes.drafting import DraftingNode
from triageai.domain.models import Classification, EvidenceSnippet, TicketCategory, Urgency
from triageai.infrastructure.llm.fake_provider import FakeLLMProvider


async def test_drafting_produces_grounded_response_with_evidence(sample_ticket):
    node = DraftingNode(llm=FakeLLMProvider())
    classification = Classification(
        category=TicketCategory.ACCESS, urgency=Urgency.HIGH, confidence=0.9, rationale="r"
    )
    evidence = [
        EvidenceSnippet(
            source_id="kb-001",
            source_type="knowledge_doc",
            title="Resetting a locked account password",
            text="Direct the user to the self-service password reset portal.",
            similarity=0.9,
        )
    ]

    result = await node(
        {"ticket": sample_ticket, "classification": classification, "evidence": evidence}
    )

    draft = result["draft"]
    assert "kb-001" in draft.cited_evidence_ids
    assert draft.text


async def test_drafting_with_no_evidence_does_not_fabricate(sample_ticket):
    node = DraftingNode(llm=FakeLLMProvider())
    classification = Classification(
        category=TicketCategory.OTHER, urgency=Urgency.MEDIUM, confidence=0.5, rationale="r"
    )

    result = await node({"ticket": sample_ticket, "classification": classification, "evidence": []})

    draft = result["draft"]
    assert draft.cited_evidence_ids == []
    assert not draft.groundedness.is_grounded
