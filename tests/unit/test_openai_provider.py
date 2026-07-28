"""Mock-based unit tests for OpenAILLMProvider — no real API key or network
call needed. This provider previously had 0% test coverage entirely
(the real key requirement made integration testing it impossible without
paying for API calls, and no mock-based unit test existed either, which
*was* possible and just hadn't been written).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from triageai.domain.models import EvidenceSnippet, Ticket, TicketCategory, Urgency
from triageai.infrastructure.llm.openai_provider import OpenAILLMProvider


@dataclass
class _FakeResponse:
    content: str


@pytest.fixture
def ticket() -> Ticket:
    return Ticket(
        subject="Locked out of my account",
        body="I typed my password wrong too many times.",
        requester_email="user@example.com",
    )


async def test_classify_parses_a_clean_json_response(ticket, monkeypatch):
    provider = OpenAILLMProvider(api_key="sk-test-not-real")
    fake_client = AsyncMock()
    fake_client.ainvoke.return_value = _FakeResponse(
        content='{"category": "access", "urgency": "high", '
        '"confidence": 0.9, "rationale": "locked out"}'
    )
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)

    classification = await provider.classify(ticket)

    assert classification.category == TicketCategory.ACCESS
    assert classification.urgency == Urgency.HIGH
    assert classification.confidence == 0.9
    assert classification.rationale == "locked out"
    fake_client.ainvoke.assert_awaited_once()


async def test_classify_extracts_json_from_markdown_code_fence(ticket, monkeypatch):
    provider = OpenAILLMProvider(api_key="sk-test-not-real")
    fake_client = AsyncMock()
    fake_client.ainvoke.return_value = _FakeResponse(
        content='Here is my answer:\n```json\n{"category": "network", '
        '"urgency": "low", "confidence": 0.6, "rationale": "vpn issue"}\n```'
    )
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)

    classification = await provider.classify(ticket)

    assert classification.category == TicketCategory.NETWORK
    assert classification.confidence == 0.6


async def test_draft_response_includes_evidence_in_the_prompt(ticket, monkeypatch):
    provider = OpenAILLMProvider(api_key="sk-test-not-real")
    fake_client = AsyncMock()
    fake_client.ainvoke.return_value = _FakeResponse(content="[kb-1] Reset your password.")
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)

    from triageai.domain.models import Classification

    classification = Classification(
        category=TicketCategory.ACCESS, urgency=Urgency.HIGH, confidence=0.9, rationale="r"
    )
    evidence = [
        EvidenceSnippet(
            source_id="kb-1",
            source_type="knowledge_doc",
            title="Password reset",
            text="Use the self-service portal.",
            similarity=0.9,
        )
    ]

    draft = await provider.draft_response(ticket, classification, evidence)

    assert draft == "[kb-1] Reset your password."
    sent_prompt = fake_client.ainvoke.call_args[0][0]
    assert "kb-1" in sent_prompt
    assert "self-service portal" in sent_prompt


async def test_draft_response_with_no_evidence_says_none(ticket, monkeypatch):
    provider = OpenAILLMProvider(api_key="sk-test-not-real")
    fake_client = AsyncMock()
    fake_client.ainvoke.return_value = _FakeResponse(content="No matching info found.")
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)

    from triageai.domain.models import Classification

    classification = Classification(
        category=TicketCategory.OTHER, urgency=Urgency.LOW, confidence=0.5, rationale="r"
    )

    await provider.draft_response(ticket, classification, [])

    sent_prompt = fake_client.ainvoke.call_args[0][0]
    assert "(none)" in sent_prompt


def test_get_client_is_constructed_lazily_and_cached():
    provider = OpenAILLMProvider(model="gpt-4o-mini", api_key="sk-test-not-real")
    assert provider._client is None

    client = provider._get_client()

    assert provider._client is client
    assert provider._get_client() is client  # same instance, not rebuilt


def test_get_client_wraps_the_api_key_in_secret_str():
    from pydantic import SecretStr

    provider = OpenAILLMProvider(api_key="sk-test-not-real")
    client = provider._get_client()

    assert isinstance(client.openai_api_key, SecretStr)
    assert client.openai_api_key.get_secret_value() == "sk-test-not-real"
