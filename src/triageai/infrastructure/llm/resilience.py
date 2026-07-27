"""Retry/timeout wrapper shared by every LLM provider adapter.

Not used by domain/application code directly — providers call this
internally so ClassificationNode/DraftingNode never need to know which
provider they're talking to or how it fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from triageai.domain.exceptions import LLMUnavailableError

T = TypeVar("T")


async def call_with_resilience(
    fn: Callable[[], Awaitable[T]],
    timeout_s: float = 15.0,
    max_attempts: int = 3,
) -> T:
    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    async def _attempt() -> T:
        return await asyncio.wait_for(fn(), timeout=timeout_s)

    try:
        return await _attempt()
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise LLMUnavailableError(f"LLM call failed after {max_attempts} attempts: {exc}") from exc
