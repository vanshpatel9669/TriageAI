"""OpenAI-backed LLM provider (optional cloud provider).

Requires OPENAI_API_KEY in the environment (see .env.example). Never commit
a real key. Not exercised by unit tests — see tests/integration/.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from triageai.domain.models import Classification, EvidenceSnippet, Ticket, TicketCategory, Urgency
from triageai.infrastructure.llm.ollama_provider import (
    _CLASSIFY_PROMPT,
    _DRAFT_PROMPT,
    _extract_json,
    _message_text,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


class OpenAILLMProvider:
    """Adapter satisfying the LLMProvider protocol via langchain-openai.

    Deliberately mirrors OllamaLLMProvider's prompt templates and JSON
    contract so swapping providers is a config change, not a rewrite.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client: ChatOpenAI | None = None

    def _get_client(self) -> ChatOpenAI:
        if self._client is None:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr

            key = SecretStr(self._api_key) if self._api_key else None
            self._client = ChatOpenAI(model=self._model, api_key=key, temperature=0.0)
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
