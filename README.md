# TriageAI

A production-oriented agent harness for IT support ticket triage: classify,
retrieve grounded evidence, draft a cited response, and route low-confidence
or high-risk tickets to a human — instead of a single opaque LLM call.

Built with Clean Architecture, a Repository Pattern over pluggable storage
(Chroma + Neo4j), dependency injection via FastAPI, and a provider-agnostic
LLM layer (Ollama by default, OpenAI as a config-driven alternative).

## Why this exists

Most "LLM does the ticket" demos skip the part that actually matters in
production: knowing when *not* to trust the model. This project's core bet
is a routing policy that sends a ticket to a human whenever the model is
unconfident, the draft cites nothing it can back up, or the category is
inherently security-sensitive — regardless of how fluent the response
sounds. See [`docs/architecture.md`](docs/architecture.md) for the full
design rationale and every trade-off made along the way.

## What's actually measured, not claimed

Every number below comes from `eval/run_evaluation.py`, is reproducible by
running it yourself, and is labeled with which LLM provider produced it —
see [`eval/results/latest_report.md`](eval/results/latest_report.md) for the
full report including known limitations of the measurement itself.

| Metric | Value (fake/keyword-rule provider, 60 synthetic tickets) |
|---|---|
| Classification accuracy | 76.7% |
| Retrieval recall@k | 100.0% |
| Groundedness rate | 76.7% |
| Hallucination rate | 23.3% |
| Escalation recall | 61.8% |
| Latency (p50 / p95) | 96.9ms / 124.5ms |

**These numbers describe pipeline wiring correctness on a controlled,
synthetic, template-generated dataset — not LLM classification/generation
quality, and not production performance.** Re-run with
`TRIAGEAI_LLM_PROVIDER=ollama` (see Quickstart) once Ollama is installed to
measure real LLM-quality numbers instead.

## Architecture

```mermaid
flowchart LR
    A[Ingestion] --> B[Classification]
    B --> C[Retrieval]
    C --> D[Drafting]
    D --> E[Routing]
    E -->|auto_resolve| F[Response returned]
    E -->|human_review| G[Queued for a human]
```

Full diagrams (layer dependency graph, request sequence diagram) and every
design decision behind them are in
[`docs/architecture.md`](docs/architecture.md).

## Stack

Python 3.11 · FastAPI (async) · LangGraph · LangChain · Neo4j (graph memory)
· Chroma (vector search) · Pydantic v2 · pytest · Docker Compose ·
GitHub Actions · structured JSON logging (`structlog`) · OpenTelemetry-ready
request IDs and latency instrumentation

## Quickstart

### Option A — Docker Compose (Neo4j + Ollama + the app)

```bash
cp .env.example .env
docker compose up -d
docker compose exec ollama ollama pull llama3.1:8b
curl http://localhost:8000/healthz
```

### Option B — local Python, no Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ollama,openai]"
python scripts/generate_synthetic_tickets.py
TRIAGEAI_LLM_PROVIDER=fake uvicorn triageai.infrastructure.api.main:app --reload
```

This runs with the deterministic fake provider and an in-memory graph
repository by default — no external services required to see it work.

### Try it

```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Locked out of my account",
    "body": "I typed my password wrong too many times and now Im locked out.",
    "requester_email": "user@example.com"
  }'
```

Full request/response examples and every endpoint: [`docs/api.md`](docs/api.md).

## Development

```bash
pip install -e ".[dev,ollama,openai]"
pre-commit install

ruff check src tests scripts eval       # lint
ruff format src tests scripts eval      # format
mypy src                                # type check
pytest tests/ -v                        # unit + integration (integration tests
                                         # skip with an explicit reason if
                                         # Neo4j/Ollama aren't reachable)
python eval/run_evaluation.py           # regenerate the evaluation report
```

## Testing philosophy

Integration tests for Neo4j and Ollama **skip with an explicit reason**
when those services aren't reachable — they are never mocked into a false
green. A skipped suite is an honest, visible signal; a green suite that
quietly tested nothing is not. Run `docker compose up` first to exercise
the full integration suite for real.

## Known limitations

- **Groundedness detection is a v1 heuristic** (citation + lexical overlap),
  not an NLI model — see the trade-off writeup in `docs/architecture.md`.
- **Escalation precision/recall are measured against this project's own
  routing policy**, not an external ground truth — there isn't one for
  synthetic data. The definition is stated explicitly in `eval/metrics.py`.
- **JWT auth, RBAC, rate limiting, Redis caching, and background workers are
  documented as future work, not built** — seeing a table of exactly where
  each would plug in and why it wasn't needed for this project's scope is in
  `docs/architecture.md`'s "Production-readiness" section.
- **Docker build/compose and the Neo4j/Ollama integration suite are verified
  in CI, not necessarily on every contributor's machine** — see
  `.github/workflows/ci.yml`.

## License

MIT — see [LICENSE](LICENSE).
