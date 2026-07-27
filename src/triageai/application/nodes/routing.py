"""Stage 5: routing — confidence-based and risk-based escalation.

A ticket is routed to human review if ANY of:
  - classification confidence is below the configured threshold
  - the draft has any unsupported (ungrounded) claim
  - the category is inherently security-sensitive (access/account), regardless
    of confidence — auto-resolving credential/access issues is a deliberate
    policy choice, not a missed edge case, and is documented as such.
"""

from __future__ import annotations

from triageai.application.state import PipelineState
from triageai.domain.models import RoutingDecision, RoutingResult, TicketCategory

_HIGH_RISK_CATEGORIES = {TicketCategory.ACCESS, TicketCategory.ACCOUNT}


class RoutingNode:
    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self._confidence_threshold = confidence_threshold

    async def __call__(self, state: PipelineState) -> dict:
        classification = state["classification"]
        draft = state["draft"]
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
        return {"routing": RoutingResult(decision=decision, reasons=reasons)}
