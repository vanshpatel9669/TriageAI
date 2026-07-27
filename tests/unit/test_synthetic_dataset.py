from __future__ import annotations

from generate_synthetic_tickets import KNOWLEDGE_BASE, generate_tickets


def test_generation_is_deterministic():
    first = generate_tickets(count=60)
    second = generate_tickets(count=60)
    assert first == second


def test_every_ticket_is_labeled_synthetic():
    tickets = generate_tickets(count=60)
    assert all(t["synthetic"] is True for t in tickets)


def test_every_ticket_has_a_valid_expected_kb_id():
    kb_ids = {doc["id"] for doc in KNOWLEDGE_BASE}
    tickets = generate_tickets(count=60)
    assert all(t["expected_kb_id"] in kb_ids for t in tickets)


def test_ticket_ids_are_unique():
    tickets = generate_tickets(count=60)
    ids = [t["ticket_id"] for t in tickets]
    assert len(ids) == len(set(ids))


def test_knowledge_base_categories_match_ticket_taxonomy():
    valid_categories = {"access", "hardware", "software", "network", "account", "other"}
    assert all(doc["category"] in valid_categories for doc in KNOWLEDGE_BASE)
