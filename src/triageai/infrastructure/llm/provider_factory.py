"""Selects an LLMProvider implementation from configuration.

To add a new provider (Anthropic, Azure OpenAI, ...): write one adapter class
satisfying `triageai.domain.interfaces.LLMProvider` in this package, then add
one branch here. Nothing in `application/` (the pipeline) ever changes.
"""

from __future__ import annotations

from triageai.config import Settings
from triageai.domain.interfaces import LLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "fake":
        from triageai.infrastructure.llm.fake_provider import FakeLLMProvider

        return FakeLLMProvider()

    if provider == "ollama":
        from triageai.infrastructure.llm.ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(model=settings.ollama_model, base_url=settings.ollama_base_url)

    if provider == "openai":
        from triageai.infrastructure.llm.openai_provider import OpenAILLMProvider

        if not settings.openai_api_key:
            raise ValueError(
                "TRIAGEAI_OPENAI_API_KEY must be set when TRIAGEAI_LLM_PROVIDER=openai"
            )
        return OpenAILLMProvider(model=settings.openai_model, api_key=settings.openai_api_key)

    raise ValueError(
        f"Unknown LLM provider {provider!r}. Supported: fake, ollama, openai "
        "(anthropic, azure_openai planned — see docs/architecture.md)."
    )
