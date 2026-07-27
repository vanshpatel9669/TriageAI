"""Ollama-backed LLM provider (default local/reproducible provider).

Requires a running Ollama daemon (`ollama serve`) with the configured model
pulled locally. Not exercised by unit tests — see tests/integration/.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from triageai.domain.models import Classification, EvidenceSnippet, Ticket, TicketCategory, Urgency

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

_CLASSIFY_PROMPT = """You are an IT support ticket classifier. Given the ticket below, \
respond with ONLY a JSON object with keys "category" (one of: access, hardware, software, \
network, account, other), "urgency" (one of: low, medium, high, critical), "confidence" \
(0.0-1.0), and "rationale" (one sentence).

Subject: {subject}
Body: {body}
"""

_DRAFT_PROMPT = """You are an IT support agent drafting a reply. Use ONLY the evidence below \
to answer. Cite each evidence item you use by its source_id in square brackets, e.g. [doc-1]. \
Do not state anything that isn't backed by the evidence.

Ticket subject: {subject}
Ticket body: {body}
Classification: {category} / {urgency}

Evidence:
{evidence}

Write a concise, grounded reply:"""


class OllamaLLMProvider:
    """Adapter satisfying the LLMProvider protocol via langchain-ollama."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client: ChatOllama | None = None  # lazy: importing this module needs no dependency

    def _get_client(self) -> ChatOllama:
        if self._client is None:
            from langchain_ollama import ChatOllama

            self._client = ChatOllama(model=self._model, base_url=self._base_url, temperature=0.0)
        return self._client

    async def classify(self, ticket: Ticket) -> Classification:
        client = self._get_client()
        prompt = _CLASSIFY_PROMPT.format(subject=ticket.subject, body=ticket.body)
        response = await client.ainvoke(prompt)
        payload = json.loads(_extract_json(_message_text(response.content)))
        return Classification(
            category=TicketCategory(payload["category"]),
            urgency=Urgency(payload["urgency"]),
            confidence=float(payload["confidence"]),
            rationale=payload["rationale"],
        )

    async def draft_response(
        self, ticket: Ticket, classification: Classification, evidence: list[EvidenceSnippet]
    ) -> str:
        client = self._get_client()
        evidence_block = (
            "\n".join(f"[{e.source_id}] {e.title}: {e.text}" for e in evidence) or "(none)"
        )
        prompt = _DRAFT_PROMPT.format(
            subject=ticket.subject,
            body=ticket.body,
            category=classification.category.value,
            urgency=classification.urgency.value,
            evidence=evidence_block,
        )
        response = await client.ainvoke(prompt)
        return str(response.content)


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in prose or code fences; pull the object out."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return text[start : end + 1]


def _message_text(content: str | list[str | dict]) -> str:
    """LangChain message content is str for most models but can be a list of
    content blocks for some providers. Normalize to plain text either way.
    """
    if isinstance(content, str):
        return content
    parts = [block if isinstance(block, str) else block.get("text", "") for block in content]
    return "".join(parts)
