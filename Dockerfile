# syntax=docker/dockerfile:1
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[ollama,openai]"

RUN useradd --create-home --uid 1000 triageai \
    && mkdir -p /app/data/chroma \
    && chown -R triageai:triageai /app
USER triageai

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "triageai.infrastructure.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
