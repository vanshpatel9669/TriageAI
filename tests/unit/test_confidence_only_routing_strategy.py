from __future__ import annotations

from triageai.domain.models import (
    Classification,
    DraftResponse,
    GroundednessCheck,
    RoutingDecision,
    TicketCategory,
    Urgency,
)
from triageai.domain.routing_strategies.confidence_only import ConfidenceOnlyRoutingStrategy


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


def test_high_confidence_grounded_auto_resolves_regardless_of_category():
    strategy = ConfidenceOnlyRoutingStrategy(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.ACCESS, urgency=Urgency.LOW, confidence=0.95, rationale="r"
    )

    result = strategy.evaluate(classification, _draft(is_grounded=True))

    assert result.decision == RoutingDecision.AUTO_RESOLVE


def test_this_is_the_genuine_behavioral_difference_from_confidence_and_risk():
    """The whole point of a second strategy: same inputs, different policy.
    ConfidenceAndRiskRoutingStrategy escalates ACCESS category regardless of
    confidence; this strategy does not — proving the registry swaps real
    behavior, not just a class name.
    """
    strategy = ConfidenceOnlyRoutingStrategy(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.ACCESS, urgency=Urgency.LOW, confidence=0.99, rationale="r"
    )

    result = strategy.evaluate(classification, _draft(is_grounded=True))

    assert result.decision == RoutingDecision.AUTO_RESOLVE
    assert result.reasons == []


def test_low_confidence_still_routes_to_human():
    strategy = ConfidenceOnlyRoutingStrategy(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.SOFTWARE, urgency=Urgency.LOW, confidence=0.4, rationale="r"
    )

    result = strategy.evaluate(classification, _draft(is_grounded=True))

    assert result.decision == RoutingDecision.HUMAN_REVIEW


def test_ungrounded_draft_still_routes_to_human():
    strategy = ConfidenceOnlyRoutingStrategy(confidence_threshold=0.75)
    classification = Classification(
        category=TicketCategory.SOFTWARE, urgency=Urgency.LOW, confidence=0.99, rationale="r"
    )

    result = strategy.evaluate(classification, _draft(is_grounded=False))

    assert result.decision == RoutingDecision.HUMAN_REVIEW
