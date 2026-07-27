# API Reference

Base URL (local): `http://localhost:8000`

## `GET /healthz`

Liveness check. Always returns 200 if the process is up.

```
curl http://localhost:8000/healthz
```

```json
{"status": "ok"}
```

## `GET /readyz`

Readiness check. Resolves the full dependency graph (LLM provider, vector
store, graph repository) — if `TRIAGEAI_USE_IN_MEMORY_GRAPH=false`, this
also implicitly depends on Neo4j being reachable, since the pipeline
singleton can't be built otherwise.

```
curl http://localhost:8000/readyz
```

```json
{"status": "ready"}
```

## `POST /tickets`

Submits a ticket for triage. Runs the full 5-stage pipeline synchronously
and returns the complete result.

**Request body:**

```json
{
  "subject": "Locked out of my account",
  "body": "I typed my password wrong too many times and now I'm locked out.",
  "requester_email": "user@example.com",
  "system": "Okta"
}
```

`system` is optional. `subject` (1-300 chars) and `body` (1-10,000 chars)
are required.

**Response — `200 OK`:**

```json
{
  "ticket": {
    "ticket_id": "98af7b31-46b8-4751-9887-18b678e186ad",
    "subject": "Locked out of my account",
    "body": "I typed my password wrong too many times and now I'm locked out.",
    "requester_email": "user@example.com",
    "system": "Okta",
    "created_at": "2026-07-27T18:22:56.698563Z"
  },
  "classification": {
    "category": "access",
    "urgency": "high",
    "confidence": 0.82,
    "rationale": "Keyword match against category 'access'."
  },
  "evidence": [
    {
      "source_id": "kb-access-01",
      "source_type": "knowledge_doc",
      "title": "Resetting a locked account password",
      "text": "...",
      "similarity": 0.87
    }
  ],
  "draft": {
    "text": "[kb-access-01] If a user is locked out after failed login attempts...",
    "cited_evidence_ids": ["kb-access-01"],
    "groundedness": {
      "is_grounded": true,
      "unsupported_claims": [],
      "supported_claim_ratio": 1.0
    }
  },
  "routing": {
    "decision": "human_review",
    "reasons": ["category 'access' requires human review by policy"]
  },
  "latency_ms": 104.3
}
```

`routing.decision` is one of `auto_resolve` or `human_review`.
`routing.reasons` is always present (empty array when auto-resolved) and
explains exactly which rule(s) triggered escalation — see
`application/nodes/routing.py`.

## `GET /tickets/{ticket_id}`

Fetches a previously triaged ticket's full result.

```
curl http://localhost:8000/tickets/98af7b31-46b8-4751-9887-18b678e186ad
```

Returns the same shape as `POST /tickets`'s response, or `404` with
`{"detail": "Ticket not found"}` if the ID doesn't exist.

## Errors

| Status | Meaning |
|---|---|
| 422 | Request body failed Pydantic validation (e.g. empty `subject`) |
| 404 | `GET /tickets/{id}` for an ID that doesn't exist |
| 503 | `readyz` when a required dependency (e.g. Neo4j) isn't reachable |

An LLM outage does **not** produce a 5xx from `POST /tickets` — see
`docs/architecture.md`'s "safe degradation" section. The request still
returns 200, routed to human review.
