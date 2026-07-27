"""Stage 2: classification — category, urgency, confidence.

If the LLM is unavailable after retries, fall back to a zero-confidence
"other" classification rather than crashing the request. Zero confidence
guarantees the routing stage sends it to human review (see routing.py) —
the failure degrades safely instead of silently guessing.
"""

from __future__ import annotations

from triageai.application.state import PipelineState
from triageai.domain.exceptions import LLMUnavailableError
from triageai.domain.interfaces import LLMProvider
from triageai.domain.models import Classification, TicketCategory, Urgency


class ClassificationNode:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def __call__(self, state: PipelineState) -> dict:
        try:
            classification = await self._llm.classify(state["ticket"])
        except LLMUnavailableError as exc:
            classification = Classification(
                category=TicketCategory.OTHER,
                urgency=Urgency.MEDIUM,
                confidence=0.0,
                rationale=f"Classification unavailable, defaulted for human review: {exc}",
            )
        return {"classification": classification}
