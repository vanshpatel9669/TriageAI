from __future__ import annotations

from metrics import (
    EvaluationRun,
    classification_accuracy,
    escalation_precision,
    escalation_recall,
    groundedness_rate,
    hallucination_rate,
    mean_reciprocal_rank,
    retrieval_recall_at_k,
)

from triageai.domain.models import (
    Classification,
    DraftResponse,
    EvidenceSnippet,
    GroundednessCheck,
    RoutingDecision,
    RoutingResult,
    Ticket,
    TicketCategory,
    TriageResult,
    Urgency,
)

AUTO = RoutingDecision.AUTO_RESOLVE
HUMAN = RoutingDecision.HUMAN_REVIEW


def _run(
    true_category: str,
    predicted_category: TicketCategory,
    expected_kb_id: str | None,
    evidence_ids: list[str],
    is_grounded: bool,
    routing: RoutingDecision,
) -> EvaluationRun:
    ticket_dict = {
        "ticket_id": "t-1",
        "true_category": true_category,
        "expected_kb_id": expected_kb_id,
    }
    evidence = [
        EvidenceSnippet(
            source_id=eid, source_type="knowledge_doc", title="t", text="x", similarity=0.5
        )
        for eid in evidence_ids
    ]
    result = TriageResult(
        ticket=Ticket(subject="s", body="b", requester_email="a@b.com"),
        classification=Classification(
            category=predicted_category, urgency=Urgency.MEDIUM, confidence=0.9, rationale="r"
        ),
        evidence=evidence,
        draft=DraftResponse(
            text="d",
            cited_evidence_ids=[],
            groundedness=GroundednessCheck(
                is_grounded=is_grounded,
                unsupported_claims=[],
                supported_claim_ratio=1.0 if is_grounded else 0.0,
            ),
        ),
        routing=RoutingResult(decision=routing, reasons=[]),
        latency_ms=10.0,
    )
    return EvaluationRun(ticket=ticket_dict, result=result)


def test_classification_accuracy_all_correct():
    runs = [_run("access", TicketCategory.ACCESS, None, [], True, AUTO)]
    assert classification_accuracy(runs) == 1.0


def test_classification_accuracy_all_wrong():
    runs = [_run("access", TicketCategory.SOFTWARE, None, [], True, AUTO)]
    assert classification_accuracy(runs) == 0.0


def test_retrieval_recall_hit_and_miss():
    runs = [
        _run("access", TicketCategory.ACCESS, "kb-1", ["kb-1", "kb-2"], True, AUTO),
        _run("access", TicketCategory.ACCESS, "kb-3", ["kb-1", "kb-2"], True, AUTO),
    ]
    assert retrieval_recall_at_k(runs) == 0.5


def test_mrr_rewards_higher_rank():
    runs = [_run("access", TicketCategory.ACCESS, "kb-1", ["kb-0", "kb-1"], True, AUTO)]
    assert mean_reciprocal_rank(runs) == 0.5  # rank 2 -> 1/2


def test_groundedness_and_hallucination_are_complementary():
    runs = [
        _run("access", TicketCategory.ACCESS, None, [], True, AUTO),
        _run("access", TicketCategory.ACCESS, None, [], False, HUMAN),
    ]
    assert groundedness_rate(runs) == 0.5
    assert hallucination_rate(runs) == 0.5


def test_escalation_precision_counts_justified_escalations():
    runs = [
        # misclassified AND escalated -> justified
        _run("access", TicketCategory.SOFTWARE, None, [], True, HUMAN),
        # correctly classified, low-risk category, grounded, but escalated anyway -> unjustified
        _run("software", TicketCategory.SOFTWARE, None, [], True, HUMAN),
    ]
    assert escalation_precision(runs) == 0.5


def test_escalation_recall_catches_misclassifications():
    runs = [
        # should escalate (misclassified) and did
        _run("access", TicketCategory.SOFTWARE, None, [], True, HUMAN),
        # should escalate (misclassified) but didn't
        _run("network", TicketCategory.SOFTWARE, None, [], True, AUTO),
    ]
    assert escalation_recall(runs) == 0.5


def test_empty_runs_do_not_crash():
    assert classification_accuracy([]) == 0.0
    assert retrieval_recall_at_k([]) == 0.0
    assert escalation_precision([]) == 0.0
