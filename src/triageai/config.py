"""Environment-based configuration. Single source of truth for all settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGEAI_", env_file=".env", extra="ignore")

    # LLM provider selection
    llm_provider: str = "fake"  # fake | ollama | openai
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    # Measured: a single classify() call against llama3.1:8b on CPU-only
    # (no GPU) local inference took ~41s end to end. 15s (a reasonable
    # default for a cloud API) was not enough and made every real call
    # exhaust its retries — see the timeout postmortem in
    # docs/architecture.md. 90s gives real margin above the measured value.
    ollama_timeout_s: float = 90.0
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    openai_timeout_s: float = 20.0

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "knowledge_docs"

    # Graph memory
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    # Default true so the app runs with zero external services out of the
    # box; set false (and run `docker compose up neo4j`) for real graph
    # memory — see infrastructure/persistence/neo4j_repositories.py.
    use_in_memory_graph: bool = True

    # Routing
    confidence_threshold: float = 0.75
    # confidence_and_risk (default) | confidence_only — see
    # domain/routing_strategies/registry.py to add another.
    routing_strategy: str = "confidence_and_risk"

    # Observability
    log_level: str = "INFO"
    service_name: str = "triageai"
    # Console exporter by default — real spans, zero external collector
    # needed. Point at a real backend by setting an OTLP endpoint instead;
    # see docs/architecture.md.
    otel_console_export: bool = True
    otel_exporter_otlp_endpoint: str | None = None


def get_settings() -> Settings:
    return Settings()
