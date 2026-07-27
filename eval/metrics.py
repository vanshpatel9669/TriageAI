"""Pure metric functions over a list of (synthetic_ticket, TriageResult) pairs.

Every metric here operates only on data actually produced by a pipeline run
— nothing here is a target or a claim, only a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from triageai.domain.models import RoutingDecision, TriageResult

# Categories where our own routing policy always escalates regardless of
# confidence (see application/nodes/routing.py). Used to define escalation
# precision/recall honestly against that policy, not an external ground truth
# for "should a human have looked at this" — no such ground truth exists for
# synthetic data, so we measure against the policy we actually implemented.
_ALWAYS_ESCALATE_CATEGORIES = {"access", "account"}


@dataclass
class EvaluationRun:
    ticket: dict
    result: TriageResult


def classification_accuracy(runs: list[EvaluationRun]) -> float:
    if not runs:
        return 0.0
    correct = sum(
        1 for r in runs if r.result.classification.category.value == r.ticket["true_category"]
    )
    return round(correct / len(runs), 4)


def retrieval_recall_at_k(runs: list[EvaluationRun]) -> float:
    """Fraction of tickets where the expected KB doc appears anywhere in
    the retrieved evidence (evidence list length is the retrieval node's
    configured top_k for docs + top_k for similar tickets).
    """
    with_expectation = [r for r in runs if r.ticket.get("expected_kb_id")]
    if not with_expectation:
        return 0.0
    hits = sum(
        1
        for r in with_expectation
        if r.ticket["expected_kb_id"] in {e.source_id for e in r.result.evidence}
    )
    return round(hits / len(with_expectation), 4)


def mean_reciprocal_rank(runs: list[EvaluationRun]) -> float:
    with_expectation = [r for r in runs if r.ticket.get("expected_kb_id")]
    if not with_expectation:
        return 0.0
    reciprocal_ranks = []
    for r in with_expectation:
        ids = [e.source_id for e in r.result.evidence]
        if r.ticket["expected_kb_id"] in ids:
            rank = ids.index(r.ticket["expected_kb_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return round(mean(reciprocal_ranks), 4)


def groundedness_rate(runs: list[EvaluationRun]) -> float:
    if not runs:
        return 0.0
    grounded = sum(1 for r in runs if r.result.draft.groundedness.is_grounded)
    return round(grounded / len(runs), 4)


def hallucination_rate(runs: list[EvaluationRun]) -> float:
    """Fraction of drafts containing at least one unsupported claim, per our
    v1 citation-and-lexical-overlap groundedness heuristic (domain/groundedness.py)
    — not an NLI-model judgment. See docs/architecture.md for that trade-off.
    """
    return round(1.0 - groundedness_rate(runs), 4)


def mean_supported_claim_ratio(runs: list[EvaluationRun]) -> float:
    if not runs:
        return 0.0
    return round(mean(r.result.draft.groundedness.supported_claim_ratio for r in runs), 4)


def escalation_precision(runs: list[EvaluationRun]) -> float:
    """Of tickets routed to human review, what fraction had a genuine reason
    per our own routing policy: misclassified vs. ground truth, OR fell in an
    always-escalate category. Not a claim about real-world correctness.
    """
    escalated = [r for r in runs if r.result.routing.decision == RoutingDecision.HUMAN_REVIEW]
    if not escalated:
        return 0.0
    justified = sum(1 for r in escalated if _escalation_justified(r))
    return round(justified / len(escalated), 4)


def escalation_recall(runs: list[EvaluationRun]) -> float:
    """Of tickets that had a genuine reason to escalate, what fraction were
    actually routed to human review.
    """
    should_escalate = [r for r in runs if _escalation_justified(r)]
    if not should_escalate:
        return 0.0
    caught = sum(
        1 for r in should_escalate if r.result.routing.decision == RoutingDecision.HUMAN_REVIEW
    )
    return round(caught / len(should_escalate), 4)


def _escalation_justified(run: EvaluationRun) -> bool:
    misclassified = run.result.classification.category.value != run.ticket["true_category"]
    high_risk = run.ticket["true_category"] in _ALWAYS_ESCALATE_CATEGORIES
    ungrounded = not run.result.draft.groundedness.is_grounded
    return misclassified or high_risk or ungrounded


def latency_stats(runs: list[EvaluationRun]) -> dict[str, float]:
    if not runs:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    latencies = sorted(r.result.latency_ms for r in runs)
    n = len(latencies)
    return {
        "mean_ms": round(mean(latencies), 2),
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 2),
        "max_ms": round(latencies[-1], 2),
    }


def compute_all_metrics(runs: list[EvaluationRun]) -> dict:
    return {
        "sample_size": len(runs),
        "classification_accuracy": classification_accuracy(runs),
        "retrieval_recall_at_k": retrieval_recall_at_k(runs),
        "retrieval_mrr": mean_reciprocal_rank(runs),
        "groundedness_rate": groundedness_rate(runs),
        "hallucination_rate": hallucination_rate(runs),
        "mean_supported_claim_ratio": mean_supported_claim_ratio(runs),
        "escalation_precision": escalation_precision(runs),
        "escalation_recall": escalation_recall(runs),
        "latency": latency_stats(runs),
    }
