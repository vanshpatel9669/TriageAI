"""Verifies PII redaction actually redacts something, not just that
configure_logging() runs without error. logging_config.py previously showed
100% line coverage from other tests exercising logging incidentally, but no
test ever fed it a string containing a real email/phone number and checked
the output — the redaction regexes' actual correctness was unverified.
"""

from __future__ import annotations

import io
import json
import logging
import sys

from triageai.logging_config import _redact_pii, configure_logging, get_logger


def test_redacts_an_email_address():
    event_dict = {"event": "contact us at jane.doe@example.com for help"}
    result = _redact_pii(None, "info", event_dict)
    assert "jane.doe@example.com" not in result["event"]
    assert "[REDACTED_EMAIL]" in result["event"]


def test_redacts_a_phone_number_with_dashes():
    event_dict = {"event": "call 555-123-4567 now"}
    result = _redact_pii(None, "info", event_dict)
    assert "555-123-4567" not in result["event"]
    assert "[REDACTED_PHONE]" in result["event"]


def test_redacts_a_phone_number_with_no_separators():
    event_dict = {"event": "call 5551234567 now"}
    result = _redact_pii(None, "info", event_dict)
    assert "5551234567" not in result["event"]
    assert "[REDACTED_PHONE]" in result["event"]


def test_redacts_across_multiple_fields():
    event_dict = {
        "requester_email": "user@example.com",
        "note": "backup contact 555.987.6543",
        "unrelated": "no pii here",
    }
    result = _redact_pii(None, "info", event_dict)
    assert result["requester_email"] == "[REDACTED_EMAIL]"
    assert "555.987.6543" not in result["note"]
    assert result["unrelated"] == "no pii here"


def test_non_string_values_pass_through_unchanged():
    event_dict = {"count": 5, "flag": None, "items": ["a", "b"]}
    result = _redact_pii(None, "info", event_dict)
    assert result == {"count": 5, "flag": None, "items": ["a", "b"]}


def test_text_with_no_pii_is_unchanged():
    event_dict = {"event": "ticket_triaged", "category": "access"}
    result = _redact_pii(None, "info", event_dict)
    assert result == {"event": "ticket_triaged", "category": "access"}


def test_configure_logging_applies_redaction_end_to_end(monkeypatch):
    """Exercises the real wiring (configure_logging + get_logger), not just
    the processor function in isolation — proves _redact_pii is actually in
    the active processor chain, not just correct if it were.
    """
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    logging.getLogger().handlers.clear()

    configure_logging(log_level="INFO")
    logger = get_logger("test")
    logger.info("ticket_received", requester_email="leaked@example.com")

    output = captured.getvalue()
    assert "leaked@example.com" not in output
    assert "[REDACTED_EMAIL]" in output
    # Confirm it's genuinely valid JSON with the redacted value, not just a
    # substring match against unstructured text.
    payload = json.loads(output.strip().splitlines()[-1])
    assert payload["requester_email"] == "[REDACTED_EMAIL]"
