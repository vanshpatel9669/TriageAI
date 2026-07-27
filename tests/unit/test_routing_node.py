"""RoutingNode itself has no policy logic anymore — it just delegates to
whatever RoutingStrategy it's given. Strategy-specific behavior is tested in
test_confidence_and_risk_routing_strategy.py and
test_confidence_only_routing_strategy.py.
"""

from __future__ import annotations

from triageai.application.nodes.routing import RoutingNode
from triageai.domain.models import RoutingDecision, RoutingResult


class _StubStrategy:
    name = "stub"

    def __init__(self, result: RoutingResult) -> None:
        self._result = result
        self.received_args = None

    def evaluate(self, classification, draft):
        self.received_args = (classification, draft)
        return self._result


async def test_routing_node_returns_whatever_the_strategy_decides():
    expected = RoutingResult(decision=RoutingDecision.HUMAN_REVIEW, reasons=["stub reason"])
    strategy = _StubStrategy(expected)
    node = RoutingNode(strategy)

    result = await node({"classification": "some-classification", "draft": "some-draft"})

    assert result["routing"] is expected


async def test_routing_node_passes_classification_and_draft_to_the_strategy():
    strategy = _StubStrategy(RoutingResult(decision=RoutingDecision.AUTO_RESOLVE, reasons=[]))
    node = RoutingNode(strategy)

    await node({"classification": "the-classification", "draft": "the-draft"})

    assert strategy.received_args == ("the-classification", "the-draft")
