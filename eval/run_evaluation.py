#!/usr/bin/env python3
"""Runs the full pipeline against data/synthetic/tickets.jsonl and writes a
measured evaluation report. Every number in the output report comes from
this run — nothing is copied from cv.md, a resume, or any prior claim.

Usage:
  python eval/run_evaluation.py
    # uses TRIAGEAI_LLM_PROVIDER env (default: fake)
  TRIAGEAI_LLM_PROVIDER=ollama python eval/run_evaluation.py
    # requires Ollama running + model pulled

Writes, per provider (so a fake-provider run and a real-provider run never
clobber each other — an earlier version of this script wrote to a single
`latest.json`/`latest_report.md` regardless of provider, and a real Ollama
run silently overwrote the fake-provider report):
  eval/results/{provider}.json
  eval/results/{provider}_report.md
  eval/results/{provider}_raw.jsonl   — one line per ticket, for post-hoc
    investigation (e.g. "did any ticket in this run actually hit the
    LLM-unavailable fallback path rather than a genuine model response?" —
    a question the aggregate metrics alone can't answer)
  eval/results/latest.json, latest_report.md — copy of the most recent run,
    for tooling that wants "whatever ran last" without knowing the provider
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import EvaluationRun, compute_all_metrics  # noqa: E402

from triageai.application.pipeline import TriagePipeline  # noqa: E402
from triageai.config import get_settings  # noqa: E402
from triageai.domain.models import Ticket  # noqa: E402
from triageai.infrastructure.llm.provider_factory import build_llm_provider  # noqa: E402
from triageai.infrastructure.persistence.chroma_knowledge_repository import (  # noqa: E402
    ChromaKnowledgeRepository,
)
from triageai.infrastructure.persistence.in_memory_repositories import (  # noqa: E402
    InMemoryGraphMemoryRepository,
    InMemoryTicketRepository,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Markers that identify a result as having gone through the safe-degradation
# fallback path (application/nodes/classification.py, drafting.py) rather
# than a genuine LLM response — used only to report this honestly, not to
# change any metric's definition.
_CLASSIFICATION_FALLBACK_MARKER = "Classification unavailable"
_DRAFT_FALLBACK_MARKER = "Unable to generate a response automatically"


def load_dataset() -> tuple[list[dict], list[dict]]:
    kb_path = DATA_DIR / "knowledge_base.json"
    tickets_path = DATA_DIR / "tickets.jsonl"
    if not kb_path.exists() or not tickets_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found at {DATA_DIR}. "
            "Run `python scripts/generate_synthetic_tickets.py` first."
        )
    knowledge_base = json.loads(kb_path.read_text())
    tickets = [json.loads(line) for line in tickets_path.read_text().splitlines() if line.strip()]
    return knowledge_base, tickets


def _is_fallback_result(run: EvaluationRun) -> bool:
    return _CLASSIFICATION_FALLBACK_MARKER in run.result.classification.rationale or (
        _DRAFT_FALLBACK_MARKER in run.result.draft.text
    )


async def run_evaluation() -> tuple[dict, list[dict]]:
    knowledge_base, tickets = load_dataset()
    settings = get_settings()
    llm = build_llm_provider(settings)

    with tempfile.TemporaryDirectory() as tmp_dir:
        vector_repo = ChromaKnowledgeRepository(tmp_dir, "eval_docs")
        vector_repo.seed(
            [{"id": d["id"], "title": d["title"], "text": d["text"]} for d in knowledge_base]
        )

        pipeline = TriagePipeline(
            llm=llm,
            vector_repo=vector_repo,
            graph_repo=InMemoryGraphMemoryRepository(),
            ticket_repo=InMemoryTicketRepository(),
            confidence_threshold=settings.confidence_threshold,
        )

        runs: list[EvaluationRun] = []
        errors: list[dict] = []
        for ticket_dict in tickets:
            ticket = Ticket(
                ticket_id=ticket_dict["ticket_id"],
                subject=ticket_dict["subject"],
                body=ticket_dict["body"],
                requester_email=ticket_dict["requester_email"],
            )
            try:
                result = await pipeline.run(ticket)
                runs.append(EvaluationRun(ticket=ticket_dict, result=result))
            except Exception as exc:  # noqa: BLE001 — record and continue, don't abort the whole eval run
                errors.append({"ticket_id": ticket_dict["ticket_id"], "error": str(exc)})

    raw_records = [
        {
            "ticket_id": run.ticket["ticket_id"],
            "true_category": run.ticket["true_category"],
            "predicted_category": run.result.classification.category.value,
            "confidence": run.result.classification.confidence,
            "is_grounded": run.result.draft.groundedness.is_grounded,
            "routing_decision": run.result.routing.decision.value,
            "latency_ms": run.result.latency_ms,
            "hit_fallback_path": _is_fallback_result(run),
        }
        for run in runs
    ]

    fallback_count = sum(1 for r in raw_records if r["hit_fallback_path"])
    metrics = compute_all_metrics(runs)
    report = {
        "llm_provider": llm.name,
        "dataset": {
            "source": (
                "synthetic (generated by scripts/generate_synthetic_tickets.py, not real user data)"
            ),
            "ticket_count": len(tickets),
            "kb_doc_count": len(knowledge_base),
        },
        "runs_completed": len(runs),
        "runs_failed": len(errors),
        "runs_hit_fallback_path": fallback_count,
        "errors": errors,
        "metrics": metrics,
    }
    return report, raw_records


def render_markdown_report(report: dict) -> str:
    m = report["metrics"]
    lat = m["latency"]
    lines = [
        "# TriageAI Evaluation Report",
        "",
        "**This report is machine-generated by `eval/run_evaluation.py` and reflects "
        "only what that run actually measured. No number here is a target, a claim, "
        "or copied from elsewhere.**",
        "",
        f"- LLM provider used for this run: `{report['llm_provider']}`",
        f"- Dataset: {report['dataset']['ticket_count']} synthetic tickets, "
        f"{report['dataset']['kb_doc_count']} synthetic KB docs "
        f"({report['dataset']['source']})",
        f"- Runs completed: {report['runs_completed']} / "
        f"{report['runs_completed'] + report['runs_failed']}",
        f"- Runs that hit the LLM-unavailable fallback path (not a genuine model "
        f"response — see application/nodes/classification.py, drafting.py): "
        f"{report['runs_hit_fallback_path']} / {report['runs_completed']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Definition |",
        "|---|---|---|",
        f"| Classification accuracy | {m['classification_accuracy']:.1%} | "
        "predicted category == synthetic ground-truth category |",
        f"| Retrieval recall@k | {m['retrieval_recall_at_k']:.1%} | "
        "expected KB doc appears anywhere in retrieved evidence |",
        f"| Retrieval MRR | {m['retrieval_mrr']:.3f} | "
        "mean reciprocal rank of the expected KB doc in retrieved evidence |",
        f"| Groundedness rate | {m['groundedness_rate']:.1%} | "
        "fraction of drafts with zero unsupported claims (v1 citation+lexical-overlap heuristic) |",
        f"| Hallucination rate | {m['hallucination_rate']:.1%} | "
        "1 - groundedness rate, by the same heuristic |",
        f"| Mean supported-claim ratio | {m['mean_supported_claim_ratio']:.1%} | "
        "average fraction of a draft's sentences that were grounded |",
        f"| Escalation precision | {m['escalation_precision']:.1%} | "
        "of tickets routed to human review, fraction with a genuine reason "
        "per our own routing policy (misclassified, high-risk category, or ungrounded) |",
        f"| Escalation recall | {m['escalation_recall']:.1%} | "
        "of tickets with a genuine reason to escalate, fraction actually routed to human review |",
        f"| Latency (mean / p50 / p95 / max) | {lat['mean_ms']:.1f}ms / "
        f"{lat['p50_ms']:.1f}ms / {lat['p95_ms']:.1f}ms / {lat['max_ms']:.1f}ms | "
        "end-to-end pipeline.run() wall-clock time |",
        "",
    ]
    if report["errors"]:
        lines.append(f"## Errors ({len(report['errors'])})")
        lines.append("")
        for err in report["errors"]:
            lines.append(f"- `{err['ticket_id']}`: {err['error']}")
        lines.append("")

    lines.append("## Known limitations of this measurement")
    lines.append("")
    lines.append(
        "- Dataset is synthetic and template-generated (see "
        "`scripts/generate_synthetic_tickets.py`), not real support tickets — "
        "these numbers describe pipeline correctness on a controlled dataset, "
        "not production performance."
    )
    if report["llm_provider"] == "fake":
        lines.append(
            "- Classification/drafting used the deterministic **fake** provider "
            "(keyword rules), not a real LLM — this run validates pipeline wiring, "
            "not LLM classification/generation quality. Re-run with "
            "`TRIAGEAI_LLM_PROVIDER=ollama` once Ollama is installed for LLM-quality numbers."
        )
    if report["runs_hit_fallback_path"] > 0:
        lines.append(
            f"- {report['runs_hit_fallback_path']} of {report['runs_completed']} runs hit the "
            "LLM-unavailable fallback (a real timeout/retry exhaustion, not a metrics artifact) "
            "and used a safe default rather than a genuine model response. See "
            f"`{report['llm_provider']}_raw.jsonl` for which tickets."
        )
    lines.append(
        "- Groundedness/hallucination detection uses a citation-and-lexical-overlap "
        "heuristic (`domain/groundedness.py`), not an NLI model — see "
        "`docs/architecture.md` for that trade-off."
    )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report, raw_records = asyncio.run(run_evaluation())
    provider = report["llm_provider"]

    json_path = RESULTS_DIR / f"{provider}.json"
    report_path = RESULTS_DIR / f"{provider}_report.md"
    raw_path = RESULTS_DIR / f"{provider}_raw.jsonl"

    json_path.write_text(json.dumps(report, indent=2))
    markdown = render_markdown_report(report)
    report_path.write_text(markdown)
    raw_path.write_text("\n".join(json.dumps(r) for r in raw_records) + "\n")

    # Convenience copies for tooling that wants "whatever ran last".
    (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2))
    (RESULTS_DIR / "latest_report.md").write_text(markdown)

    print(markdown)
    print(f"\nWrote {json_path}, {report_path}, {raw_path}")


if __name__ == "__main__":
    os.environ.setdefault("TRIAGEAI_LLM_PROVIDER", "fake")
    main()
