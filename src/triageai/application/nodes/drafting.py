"""Stage 4: drafting — grounded response generation + hallucination check.

Evidence is required: if retrieval found nothing, we do not let the LLM
freelance an answer — the draft explicitly says so and groundedness is
scored on that basis, which naturally pushes routing toward human review.

If the LLM is unavailable after retries, the draft says so explicitly and is
marked ungrounded, which — same mechanism as ClassificationNode — pushes
routing to human review instead of crashing the request.
"""

from __future__ import annotations

from triageai.application.state import PipelineState
from triageai.domain.exceptions import LLMUnavailableError
from triageai.domain.groundedness import check_groundedness
from triageai.domain.interfaces import LLMProvider
from triageai.domain.models import DraftResponse, GroundednessCheck


class DraftingNode:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def __call__(self, state: PipelineState) -> dict:
        ticket = state["ticket"]
        classification = state["classification"]
        evidence = state.get("evidence", [])

        try:
            text = await self._llm.draft_response(ticket, classification, evidence)
        except LLMUnavailableError as exc:
            draft = DraftResponse(
                text=(
                    "Unable to generate a response automatically right now. "
                    "This ticket has been routed to a human agent."
                ),
                cited_evidence_ids=[],
                groundedness=GroundednessCheck(
                    is_grounded=False,
                    unsupported_claims=[f"drafting unavailable: {exc}"],
                    supported_claim_ratio=0.0,
                ),
            )
            return {"draft": draft}

        groundedness = check_groundedness(text, evidence)
        cited_ids = [e.source_id for e in evidence if f"[{e.source_id}]" in text]

        draft = DraftResponse(text=text, cited_evidence_ids=cited_ids, groundedness=groundedness)
        return {"draft": draft}
