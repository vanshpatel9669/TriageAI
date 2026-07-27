#!/usr/bin/env python3
"""Generates a synthetic (clearly labeled, not real user data) IT-ticket
dataset plus a matching knowledge-base doc set, for use by eval/run_evaluation.py.

Deterministic: fixed random seed, so the dataset is reproducible byte-for-byte
across runs — required for reproducible evaluation numbers.

Usage: python scripts/generate_synthetic_tickets.py
Writes: data/synthetic/knowledge_base.json, data/synthetic/tickets.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"

KNOWLEDGE_BASE = [
    {
        "id": "kb-access-01",
        "title": "Resetting a locked account password",
        "text": (
            "If a user is locked out after failed login attempts, direct them to the "
            "self-service password reset portal at reset.internal.example. Accounts "
            "unlock automatically after 30 minutes, or an admin can unlock immediately "
            "via the IAM console."
        ),
        "category": "access",
    },
    {
        "id": "kb-access-02",
        "title": "Requesting elevated system permissions",
        "text": (
            "Elevated or admin permissions require a manager-approved access request "
            "submitted through the IAM portal. Standard approval turnaround is 1 "
            "business day; emergency access can be granted by the on-call security "
            "engineer with a documented justification."
        ),
        "category": "access",
    },
    {
        "id": "kb-hardware-01",
        "title": "Requesting new hardware",
        "text": (
            "New laptop or monitor requests go through the hardware request form in "
            "the internal portal. Standard turnaround is 5 business days. Urgent "
            "hardware failures (device won't power on) should be flagged as high "
            "urgency for same-day loaner issuance."
        ),
        "category": "hardware",
    },
    {
        "id": "kb-hardware-02",
        "title": "Laptop won't power on",
        "text": (
            "First confirm the charger and outlet both work with another device. If "
            "the laptop still won't power on, hold the power button for 15 seconds to "
            "force a hardware reset. If that fails, issue a loaner from the IT closet "
            "and file a hardware repair ticket with asset tag."
        ),
        "category": "hardware",
    },
    {
        "id": "kb-software-01",
        "title": "Application crashes on startup",
        "text": (
            "Application crashes on launch are usually resolved by clearing the local "
            "app cache (Settings > Storage > Clear Cache) and reinstalling the latest "
            "version from the internal app catalog. If the crash persists, collect the "
            "crash log from %LOCALAPPDATA%/CrashLogs and attach it to the ticket."
        ),
        "category": "software",
    },
    {
        "id": "kb-software-02",
        "title": "Software installation request",
        "text": (
            "New software installations must be requested through the software "
            "catalog. Pre-approved tools install automatically within 15 minutes; "
            "anything outside the catalog requires a security review before "
            "installation, typically 2-3 business days."
        ),
        "category": "software",
    },
    {
        "id": "kb-network-01",
        "title": "VPN connection failures",
        "text": (
            "VPN connection failures are most often caused by an expired client "
            "certificate. Ask the user to open the VPN client, select 'Renew "
            "Certificate', and restart the client. If the issue persists after "
            "renewal, escalate to network engineering."
        ),
        "category": "network",
    },
    {
        "id": "kb-network-02",
        "title": "Wi-Fi keeps disconnecting",
        "text": (
            "Intermittent Wi-Fi drops in office buildings are usually caused by AP "
            "roaming near floor boundaries. Ask the user to forget the network and "
            "reconnect. If the issue is isolated to one area, file a facilities ticket "
            "to check access point signal strength in that zone."
        ),
        "category": "network",
    },
    {
        "id": "kb-account-01",
        "title": "Updating account email or display name",
        "text": (
            "Legal name or display name changes require an HR-verified request routed "
            "through the identity management team, since it affects badge records and "
            "payroll systems. Email alias changes for preferred name can be "
            "self-serviced immediately from the profile settings page."
        ),
        "category": "account",
    },
    {
        "id": "kb-account-02",
        "title": "Merging duplicate user accounts",
        "text": (
            "Duplicate accounts (e.g. from a rehire) must be merged by the identity "
            "team, never deleted directly, to preserve audit history. Flag the "
            "duplicate account IDs and submit an account-merge request; expect 2-3 "
            "business days for completion."
        ),
        "category": "account",
    },
]

# (category, urgency, subject_template, body_template, expected_kb_id)
TICKET_TEMPLATES = [
    (
        "access",
        "high",
        "Locked out of my account",
        "I typed my password wrong too many times on {system} and now I'm locked out. "
        "I need to get back in urgently, I have a deadline today.",
        "kb-access-01",
    ),
    (
        "access",
        "medium",
        "Need admin access to {system}",
        "Can I get elevated permissions on {system}? My manager approved it in Slack "
        "but I'm not sure how to formally request it.",
        "kb-access-02",
    ),
    (
        "hardware",
        "medium",
        "Requesting a new monitor",
        "My current monitor on {system} has a flickering screen. Can I request a "
        "replacement through the normal process?",
        "kb-hardware-01",
    ),
    (
        "hardware",
        "high",
        "Laptop completely dead, won't turn on",
        "My laptop ({system}) won't power on at all, tried the charger on another "
        "outlet and it still won't turn on. This is urgent, I have back to back "
        "meetings today.",
        "kb-hardware-02",
    ),
    (
        "software",
        "medium",
        "{system} crashes every time I open it",
        "Every time I try to launch {system} it crashes immediately with no error "
        "message. I've restarted my computer twice already.",
        "kb-software-01",
    ),
    (
        "software",
        "low",
        "Requesting install of {system}",
        "Could someone install {system} on my machine? I need it for a project "
        "starting next week, no rush.",
        "kb-software-02",
    ),
    (
        "network",
        "high",
        "VPN won't connect at all",
        "My VPN client fails every time I try to connect from home, and I can't reach "
        "{system} at all. This is blocking my whole workday.",
        "kb-network-01",
    ),
    (
        "network",
        "medium",
        "Wi-Fi keeps dropping near my desk",
        "My Wi-Fi disconnects every 20 minutes or so while I'm working near {system} "
        "on the 4th floor. Reconnecting fixes it temporarily.",
        "kb-network-02",
    ),
    (
        "account",
        "low",
        "Need to update my display name",
        "I recently changed my legal name and would like my account on {system} "
        "updated to match. No urgency, whenever it's convenient.",
        "kb-account-01",
    ),
    (
        "account",
        "medium",
        "Duplicate account from rehire",
        "I was rehired and now have two accounts showing up in {system}. Can these "
        "be merged? I don't want to lose my old history.",
        "kb-account-02",
    ),
]

SYSTEMS = [
    "the CRM",
    "Okta",
    "the finance portal",
    "Slack",
    "the VPN client",
    "SharePoint",
    "the design tool",
    "Jira",
    "the internal wiki",
    "Zoom",
]

REQUESTERS = [
    "alex.chen",
    "jordan.smith",
    "priya.raman",
    "sam.oconnor",
    "morgan.lee",
    "taylor.nguyen",
    "casey.brooks",
    "riley.patel",
    "jamie.diaz",
    "avery.kim",
]


def generate_tickets(count: int) -> list[dict]:
    rng = random.Random(SEED)
    tickets = []
    for i in range(count):
        category, urgency, subject_tpl, body_tpl, expected_kb_id = rng.choice(TICKET_TEMPLATES)
        system = rng.choice(SYSTEMS)
        requester = rng.choice(REQUESTERS)
        tickets.append(
            {
                "ticket_id": f"synthetic-{i:04d}",
                "subject": subject_tpl.format(system=system),
                "body": body_tpl.format(system=system),
                "requester_email": f"{requester}@example.com",
                "true_category": category,
                "true_urgency": urgency,
                "expected_kb_id": expected_kb_id,
                "synthetic": True,
            }
        )
    return tickets


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kb_path = OUTPUT_DIR / "knowledge_base.json"
    kb_path.write_text(json.dumps(KNOWLEDGE_BASE, indent=2))

    tickets = generate_tickets(count=60)
    tickets_path = OUTPUT_DIR / "tickets.jsonl"
    with tickets_path.open("w") as f:
        for ticket in tickets:
            f.write(json.dumps(ticket) + "\n")

    print(f"Wrote {len(KNOWLEDGE_BASE)} KB docs to {kb_path}")
    print(f"Wrote {len(tickets)} synthetic tickets to {tickets_path}")


if __name__ == "__main__":
    main()
