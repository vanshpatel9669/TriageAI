from __future__ import annotations

import asyncio

import pytest

from triageai.domain.exceptions import LLMUnavailableError
from triageai.infrastructure.llm.resilience import call_with_resilience


async def test_succeeds_on_first_try():
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await call_with_resilience(fn, timeout_s=1, max_attempts=3)

    assert result == "ok"
    assert calls == 1


async def test_retries_then_succeeds():
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await call_with_resilience(fn, timeout_s=1, max_attempts=5)

    assert result == "ok"
    assert calls == 3


async def test_raises_llm_unavailable_after_exhausting_retries():
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise ConnectionError("down")

    with pytest.raises(LLMUnavailableError):
        await call_with_resilience(fn, timeout_s=1, max_attempts=3)

    assert calls == 3


async def test_timeout_raises_llm_unavailable():
    async def fn() -> str:
        await asyncio.sleep(10)
        return "too slow"

    with pytest.raises(LLMUnavailableError):
        await call_with_resilience(fn, timeout_s=0.05, max_attempts=1)


async def test_non_retryable_exception_propagates_unchanged():
    async def fn() -> str:
        raise ValueError("not a transient failure")

    with pytest.raises(ValueError):
        await call_with_resilience(fn, timeout_s=1, max_attempts=3)
