"""A second, genuinely different routing strategy — demonstrates the
registry is actually pluggable, not just a rename of the one policy that
used to be hardcoded. Escalates on low confidence or an ungrounded draft
only; unlike ConfidenceAndRiskRoutingStrategy, it does NOT treat any
category as automatically high-risk. Not the recommended default (see
docs/architecture.md for why access/account tickets get a category-level
policy override), but a legitimate simpler policy for a deployment that
doesn't need it.
"""

from __future__ import annotations

from triageai.domain.models import Classification, DraftResponse, RoutingDecision, RoutingResult


class ConfidenceOnlyRoutingStrategy:
    """Escalates on low confidence or an ungrounded draft. No category override."""

    name = "confidence_only"

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

        decision = RoutingDecision.HUMAN_REVIEW if reasons else RoutingDecision.AUTO_RESOLVE
        return RoutingResult(decision=decision, reasons=reasons)
