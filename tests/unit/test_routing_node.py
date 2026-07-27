from __future__ import annotations

from triageai.application.nodes.routing import RoutingNode
from triageai.domain.models import (
    Classification,
    DraftResponse,
    GroundednessCheck,
    RoutingDecision,
    TicketCategory,
    Urgency,
)


def _draft(is_grounded: bool) -> DraftResponse:
    return DraftResponse(
        text="text",
        cited_evidence_ids=[],
        groundedness=GroundednessCheck(
            is_grounded=is_grounded,
            unsupported_claims=[],
            supported_claim_ratio=1.0 if is_grounded else 0.0,
        ),
    )


async def test_high_confidence_grounded_low_risk_auto_resolves():
    node = RoutingNode(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.SOFTWARE, urgency=Urgency.LOW, confidence=0.95, rationale="r"
    )
    result = await node({"classification": classification, "draft": _draft(is_grounded=True)})

    assert result["routing"].decision == RoutingDecision.AUTO_RESOLVE
    assert result["routing"].reasons == []


async def test_low_confidence_routes_to_human():
    node = RoutingNode(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.SOFTWARE, urgency=Urgency.LOW, confidence=0.4, rationale="r"
    )
    result = await node({"classification": classification, "draft": _draft(is_grounded=True)})

    assert result["routing"].decision == RoutingDecision.HUMAN_REVIEW
    assert any("confidence" in reason for reason in result["routing"].reasons)


async def test_ungrounded_draft_routes_to_human_even_with_high_confidence():
    node = RoutingNode(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.SOFTWARE, urgency=Urgency.LOW, confidence=0.99, rationale="r"
    )
    result = await node({"classification": classification, "draft": _draft(is_grounded=False)})

    assert result["routing"].decision == RoutingDecision.HUMAN_REVIEW


async def test_high_risk_category_always_routes_to_human():
    node = RoutingNode(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.ACCESS, urgency=Urgency.LOW, confidence=0.99, rationale="r"
    )
    result = await node({"classification": classification, "draft": _draft(is_grounded=True)})

    assert result["routing"].decision == RoutingDecision.HUMAN_REVIEW
    assert any("policy" in reason for reason in result["routing"].reasons)
