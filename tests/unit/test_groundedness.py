from __future__ import annotations

from triageai.domain.groundedness import check_groundedness
from triageai.domain.models import EvidenceSnippet


def _evidence(source_id: str, text: str) -> EvidenceSnippet:
    return EvidenceSnippet(
        source_id=source_id, source_type="knowledge_doc", title="t", text=text, similarity=1.0
    )


def test_cited_and_supported_claim_is_grounded():
    evidence = [_evidence("kb-1", "Reset your password using the self-service portal.")]
    draft = "Use the self-service portal to reset your password [kb-1]."

    result = check_groundedness(draft, evidence)

    assert result.is_grounded
    assert result.unsupported_claims == []
    assert result.supported_claim_ratio == 1.0


def test_uncited_sentence_is_unsupported():
    evidence = [_evidence("kb-1", "Reset your password using the self-service portal.")]
    draft = "Your account will be permanently deleted in 24 hours."

    result = check_groundedness(draft, evidence)

    assert not result.is_grounded
    assert len(result.unsupported_claims) == 1


def test_cited_but_unrelated_content_is_unsupported():
    evidence = [_evidence("kb-1", "Reset your password using the self-service portal.")]
    draft = "Your laptop warranty has expired and needs replacement [kb-1]."

    result = check_groundedness(draft, evidence)

    assert not result.is_grounded


def test_empty_draft_is_not_grounded():
    result = check_groundedness("", [])
    assert not result.is_grounded
    assert result.supported_claim_ratio == 0.0
