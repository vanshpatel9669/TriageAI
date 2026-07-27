"""Stage 5: routing — delegates to a pluggable RoutingStrategy.

The policy itself (confidence threshold, groundedness, category risk) lives
in domain/routing_strategies/ — this node just wires state in and out. See
domain/routing_strategies/registry.py to add a new strategy.
"""

from __future__ import annotations

from triageai.application.state import PipelineState
from triageai.domain.interfaces import RoutingStrategy


class RoutingNode:
    def __init__(self, strategy: RoutingStrategy) -> None:
        self._strategy = strategy

    async def __call__(self, state: PipelineState) -> dict:
        routing = self._strategy.evaluate(state["classification"], state["draft"])
        return {"routing": routing}
