from __future__ import annotations

import pytest

from triageai.domain.models import Ticket


@pytest.fixture
def sample_ticket() -> Ticket:
    return Ticket(
        subject="Locked out of my account",
        body="I typed my password wrong too many times and now I'm locked out. This is urgent.",
        requester_email="user@example.com",
    )
