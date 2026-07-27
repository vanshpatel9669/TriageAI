# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ollama,openai]"
pre-commit install
python scripts/generate_synthetic_tickets.py
```

## Before opening a PR

```bash
ruff check src tests scripts eval
ruff format --check src tests scripts eval
mypy src
pytest tests/ -v
```

`pre-commit` runs the lint/format/mypy checks automatically on commit.
Integration tests (Neo4j, Ollama) skip automatically if those services
aren't running locally — see the Testing philosophy section in the README.
To run them for real: `docker compose up -d` and, for Ollama coverage,
`docker compose exec ollama ollama pull llama3.1:8b`.

## Code organization

This project follows Clean Architecture — see `docs/architecture.md` for the
full rationale. The short version:

- `domain/` — models, Protocol interfaces, pure business logic. No imports
  from `application/` or `infrastructure/`, ever.
- `application/` — the LangGraph pipeline and its five nodes. Depends only
  on `domain/` Protocols, never a concrete infrastructure class.
- `infrastructure/` — FastAPI routes, Chroma/Neo4j adapters, LLM provider
  adapters. This is the only layer allowed to import a concrete third-party
  client.

If a change needs to reach from `domain/` or `application/` into a concrete
`infrastructure/` class, that's a sign the abstraction (a `Protocol` in
`domain/interfaces.py`) needs to grow instead.

## Adding a new LLM provider

1. Write an adapter class in `infrastructure/llm/` satisfying the
   `LLMProvider` Protocol (`domain/interfaces.py`) — see
   `openai_provider.py` for the shape to follow.
2. Add one branch to `infrastructure/llm/provider_factory.py`.
3. Nothing in `application/nodes/` changes.

## Commit messages

State what changed and, if it's not obvious from the diff, why. If you found
and fixed a real bug while building something else, say so — that's more
useful to a reviewer than pretending the code was right the first time.
