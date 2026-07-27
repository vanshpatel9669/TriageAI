"""Groundedness / hallucination checking — pure domain logic, no I/O.

v1 approach: citation-and-lexical-overlap heuristic. Every sentence in a
draft must (a) cite an evidence source_id in square brackets and (b) share
enough content words with that cited evidence's text. Sentences that fail
either check are flagged as unsupported.

This is a deliberately simple, fully explainable, zero-cost baseline — not
an NLI/entailment model. That trade-off is documented in
docs/architecture.md as a named limitation with a concrete upgrade path
(swap this function for an NLI-model-backed one behind the same signature),
rather than silently presented as more rigorous than it is.
"""

from __future__ import annotations

import re

from triageai.domain.models import EvidenceSnippet, GroundednessCheck

_CITATION_RE = re.compile(r"\[([\w.-]+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "for", "in", "on", "with", "this", "that", "it", "be", "as", "by", "your",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def check_groundedness(
    draft: str,
    evidence: list[EvidenceSnippet],
    min_overlap_ratio: float = 0.15,
) -> GroundednessCheck:
    evidence_by_id = {e.source_id: e for e in evidence}
    sentences = _split_sentences(draft)

    if not sentences:
        return GroundednessCheck(
            is_grounded=False, unsupported_claims=[], supported_claim_ratio=0.0
        )

    unsupported: list[str] = []
    supported_count = 0

    for sentence in sentences:
        cited_ids = _CITATION_RE.findall(sentence)
        cited_evidence = [evidence_by_id[cid] for cid in cited_ids if cid in evidence_by_id]

        if not cited_evidence:
            unsupported.append(sentence)
            continue

        sentence_words = _content_words(sentence)
        evidence_words: set[str] = set()
        for snippet in cited_evidence:
            evidence_words |= _content_words(snippet.text) | _content_words(snippet.title)

        if not sentence_words:
            unsupported.append(sentence)
            continue

        overlap_ratio = len(sentence_words & evidence_words) / len(sentence_words)
        if overlap_ratio >= min_overlap_ratio:
            supported_count += 1
        else:
            unsupported.append(sentence)

    ratio = supported_count / len(sentences)
    return GroundednessCheck(
        is_grounded=not unsupported,
        unsupported_claims=unsupported,
        supported_claim_ratio=round(ratio, 4),
    )
