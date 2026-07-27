"""Stage 2: classification — category, urgency, confidence."""

from __future__ import annotations

from triageai.application.state import PipelineState
from triageai.domain.interfaces import LLMProvider


class ClassificationNode:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def __call__(self, state: PipelineState) -> dict:
        classification = await self._llm.classify(state["ticket"])
        return {"classification": classification}
