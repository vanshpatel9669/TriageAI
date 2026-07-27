"""Selects a RoutingStrategy implementation by name.

To add a new routing strategy: write one class satisfying
`domain.interfaces.RoutingStrategy` in this package, add one branch here.
`application/nodes/routing.py` never changes.
"""

from __future__ import annotations

from triageai.domain.interfaces import RoutingStrategy


def build_routing_strategy(name: str, confidence_threshold: float = 0.75) -> RoutingStrategy:
    strategy = name.lower()

    if strategy == "confidence_and_risk":
        from triageai.domain.routing_strategies.confidence_and_risk import (
            ConfidenceAndRiskRoutingStrategy,
        )

        return ConfidenceAndRiskRoutingStrategy(confidence_threshold=confidence_threshold)

    if strategy == "confidence_only":
        from triageai.domain.routing_strategies.confidence_only import (
            ConfidenceOnlyRoutingStrategy,
        )

        return ConfidenceOnlyRoutingStrategy(confidence_threshold=confidence_threshold)

    raise ValueError(
        f"Unknown routing strategy {name!r}. Supported: confidence_and_risk, confidence_only."
    )
