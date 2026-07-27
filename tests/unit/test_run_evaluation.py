from __future__ import annotations

import pytest
from run_evaluation import load_dataset, render_markdown_report


def test_load_dataset_raises_clear_error_when_missing(tmp_path, monkeypatch):
    import run_evaluation

    monkeypatch.setattr(run_evaluation, "DATA_DIR", tmp_path / "does-not-exist")

    with pytest.raises(FileNotFoundError, match="generate_synthetic_tickets"):
        load_dataset()


def test_render_markdown_report_labels_fake_provider_explicitly():
    report = {
        "llm_provider": "fake",
        "dataset": {"source": "synthetic", "ticket_count": 5, "kb_doc_count": 2},
        "runs_completed": 5,
        "runs_failed": 0,
        "errors": [],
        "metrics": {
            "classification_accuracy": 0.8,
            "retrieval_recall_at_k": 0.9,
            "retrieval_mrr": 0.5,
            "groundedness_rate": 0.7,
            "hallucination_rate": 0.3,
            "mean_supported_claim_ratio": 0.75,
            "escalation_precision": 1.0,
            "escalation_recall": 0.6,
            "latency": {"mean_ms": 10.0, "p50_ms": 9.0, "p95_ms": 15.0, "max_ms": 20.0},
        },
    }

    markdown = render_markdown_report(report)

    assert "fake" in markdown
    assert "validates pipeline wiring, not LLM" in markdown
    assert "80.0%" in markdown  # classification accuracy formatted as a percentage


def test_render_markdown_report_omits_fake_disclaimer_for_real_provider():
    report = {
        "llm_provider": "ollama",
        "dataset": {"source": "synthetic", "ticket_count": 5, "kb_doc_count": 2},
        "runs_completed": 5,
        "runs_failed": 0,
        "errors": [],
        "metrics": {
            "classification_accuracy": 0.8,
            "retrieval_recall_at_k": 0.9,
            "retrieval_mrr": 0.5,
            "groundedness_rate": 0.7,
            "hallucination_rate": 0.3,
            "mean_supported_claim_ratio": 0.75,
            "escalation_precision": 1.0,
            "escalation_recall": 0.6,
            "latency": {"mean_ms": 10.0, "p50_ms": 9.0, "p95_ms": 15.0, "max_ms": 20.0},
        },
    }

    markdown = render_markdown_report(report)

    assert "validates pipeline wiring, not LLM" not in markdown


def test_render_markdown_report_lists_errors_when_present():
    report = {
        "llm_provider": "fake",
        "dataset": {"source": "synthetic", "ticket_count": 1, "kb_doc_count": 1},
        "runs_completed": 0,
        "runs_failed": 1,
        "errors": [{"ticket_id": "t-1", "error": "boom"}],
        "metrics": {
            "classification_accuracy": 0.0,
            "retrieval_recall_at_k": 0.0,
            "retrieval_mrr": 0.0,
            "groundedness_rate": 0.0,
            "hallucination_rate": 0.0,
            "mean_supported_claim_ratio": 0.0,
            "escalation_precision": 0.0,
            "escalation_recall": 0.0,
            "latency": {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0},
        },
    }

    markdown = render_markdown_report(report)

    assert "t-1" in markdown
    assert "boom" in markdown
