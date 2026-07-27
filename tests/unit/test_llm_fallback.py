"""Verifies nodes degrade safely (route to human review) when the LLM is
unavailable, instead of crashing the request.
"""

from __future__ import annotations

from triageai.application.nodes.classification import ClassificationNode
from triageai.application.nodes.drafting import DraftingNode
from triageai.application.nodes.routing import RoutingNode
from triageai.domain.exceptions import LLMUnavailableError
from triageai.domain.models import Classification, TicketCategory, Urgency


class _AlwaysDownLLM:
    name = "always_down"

    async def classify(self, ticket):
        raise LLMUnavailableError("simulated outage")

    async def draft_response(self, ticket, classification, evidence):
        raise LLMUnavailableError("simulated outage")


async def test_classification_falls_back_to_zero_confidence_other(sample_ticket):
    node = ClassificationNode(llm=_AlwaysDownLLM())
    result = await node({"ticket": sample_ticket})

    classification = result["classification"]
    assert classification.category == TicketCategory.OTHER
    assert classification.confidence == 0.0


async def test_drafting_falls_back_to_ungrounded_notice(sample_ticket):
    node = DraftingNode(llm=_AlwaysDownLLM())
    classification = Classification(
        category=TicketCategory.SOFTWARE, urgency=Urgency.LOW, confidence=0.9, rationale="r"
    )

    result = await node({"ticket": sample_ticket, "classification": classification, "evidence": []})

    draft = result["draft"]
    assert not draft.groundedness.is_grounded
    assert draft.cited_evidence_ids == []


async def test_llm_outage_end_to_end_routes_to_human_review(sample_ticket):
    """Both fallbacks compose correctly with the existing routing safety net."""
    classification_node = ClassificationNode(llm=_AlwaysDownLLM())
    drafting_node = DraftingNode(llm=_AlwaysDownLLM())
    routing_node = RoutingNode(confidence_threshold=0.75)

    classification_result = await classification_node({"ticket": sample_ticket})
    draft_result = await drafting_node(
        {
            "ticket": sample_ticket,
            "classification": classification_result["classification"],
            "evidence": [],
        }
    )
    routing_result = await routing_node(
        {"classification": classification_result["classification"], "draft": draft_result["draft"]}
    )

    assert routing_result["routing"].decision.value == "human_review"
    assert len(routing_result["routing"].reasons) >= 2  # low confidence AND ungrounded
