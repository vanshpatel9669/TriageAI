"""Environment-based configuration. Single source of truth for all settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGEAI_", env_file=".env", extra="ignore")

    # LLM provider selection
    llm_provider: str = "fake"  # fake | ollama | openai
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "knowledge_docs"

    # Graph memory
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    use_in_memory_graph: bool = True  # True until Neo4j is wired in Phase 3

    # Routing
    confidence_threshold: float = 0.75

    # Observability
    log_level: str = "INFO"
    service_name: str = "triageai"


def get_settings() -> Settings:
    return Settings()
