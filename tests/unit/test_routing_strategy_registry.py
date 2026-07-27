from __future__ import annotations

import pytest

from triageai.domain.routing_strategies.confidence_and_risk import (
    ConfidenceAndRiskRoutingStrategy,
)
from triageai.domain.routing_strategies.confidence_only import ConfidenceOnlyRoutingStrategy
from triageai.domain.routing_strategies.registry import build_routing_strategy


def test_builds_confidence_and_risk_by_name():
    strategy = build_routing_strategy("confidence_and_risk", confidence_threshold=0.6)
    assert isinstance(strategy, ConfidenceAndRiskRoutingStrategy)


def test_builds_confidence_only_by_name():
    strategy = build_routing_strategy("confidence_only", confidence_threshold=0.6)
    assert isinstance(strategy, ConfidenceOnlyRoutingStrategy)


def test_name_matching_is_case_insensitive():
    strategy = build_routing_strategy("CONFIDENCE_ONLY", confidence_threshold=0.6)
    assert isinstance(strategy, ConfidenceOnlyRoutingStrategy)


def test_unknown_strategy_raises_with_the_supported_list():
    with pytest.raises(ValueError, match="confidence_and_risk, confidence_only"):
        build_routing_strategy("not_a_real_strategy")
