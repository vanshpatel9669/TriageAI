from __future__ import annotations

from triageai.application.nodes.classification import ClassificationNode
from triageai.domain.models import TicketCategory
from triageai.infrastructure.llm.fake_provider import FakeLLMProvider


async def test_classification_node_returns_classification(sample_ticket):
    node = ClassificationNode(llm=FakeLLMProvider())
    result = await node({"ticket": sample_ticket})

    classification = result["classification"]
    assert classification.category == TicketCategory.ACCESS
    assert 0.0 <= classification.confidence <= 1.0
    assert classification.rationale
