"""Default routing strategy: confidence threshold + groundedness + a
security-sensitive-category allowlist. This is the same policy that used to
live hardcoded inside RoutingNode itself — extracted here so it's one
interchangeable strategy among several, not the only option the node knows
how to run.
"""

from __future__ import annotations

from triageai.domain.models import (
    Classification,
    DraftResponse,
    RoutingDecision,
    RoutingResult,
    TicketCategory,
)

_HIGH_RISK_CATEGORIES = {TicketCategory.ACCESS, TicketCategory.ACCOUNT}


class ConfidenceAndRiskRoutingStrategy:
    """Escalates on low confidence, any unsupported claim, or a
    security-sensitive category — regardless of confidence.
    """

    name = "confidence_and_risk"

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self._confidence_threshold = confidence_threshold

    def evaluate(self, classification: Classification, draft: DraftResponse) -> RoutingResult:
        reasons: list[str] = []

        if classification.confidence < self._confidence_threshold:
            reasons.append(
                f"classification confidence {classification.confidence:.2f} "
                f"below threshold {self._confidence_threshold:.2f}"
            )

        if not draft.groundedness.is_grounded:
            reasons.append(
                f"{len(draft.groundedness.unsupported_claims)} unsupported claim(s) in draft"
            )

        if classification.category in _HIGH_RISK_CATEGORIES:
            reasons.append(
                f"category '{classification.category.value}' requires human review by policy"
            )

        decision = RoutingDecision.HUMAN_REVIEW if reasons else RoutingDecision.AUTO_RESOLVE
        return RoutingResult(decision=decision, reasons=reasons)
