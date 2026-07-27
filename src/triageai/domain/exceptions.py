"""Domain-level exceptions. Any infrastructure adapter may raise these;
application-layer nodes catch them without depending on a concrete adapter.
"""

from __future__ import annotations


class LLMUnavailableError(Exception):
    """An LLM provider failed after retries/timeout. Callers should apply a
    safe fallback (route to human review) rather than let the request crash.
    """
