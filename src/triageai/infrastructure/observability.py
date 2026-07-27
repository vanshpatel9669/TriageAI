"""Prometheus metrics + OpenTelemetry tracing wiring for the FastAPI app.

Kept separate from main.py (like logging_config.py) so the app factory
stays about app construction, not observability plumbing.
"""

from __future__ import annotations

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_fastapi_instrumentator import Instrumentator


def build_tracer_provider(
    service_name: str, console_export: bool = True, otlp_endpoint: str | None = None
) -> TracerProvider:
    """Pure construction, no global state — kept separate from
    configure_tracing() so the exporter-selection logic (OTLP if an endpoint
    is set, else console if enabled, else nothing) is unit-testable by
    inspecting the returned provider directly, without touching the
    process-global TracerProvider singleton (see configure_tracing()'s
    docstring for why that singleton makes this path hard to test any other
    way).
    """
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    elif console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    return provider


def configure_tracing(
    service_name: str, console_export: bool = True, otlp_endpoint: str | None = None
) -> None:
    """Registers a global TracerProvider. Console exporter by default — real
    spans, no external collector required for local dev. Point at a real
    backend (Jaeger, Tempo, ...) by setting TRIAGEAI_OTEL_EXPORTER_OTLP_ENDPOINT
    instead; swapping the exporter is the only change needed.

    The OTel SDK only allows the global TracerProvider to be set once per
    process — a second call elsewhere (e.g. a test suite that calls
    create_app() many times) is a silent no-op with a logged warning, not an
    error. Guarded here so repeated calls in the same process don't spam
    that warning; the first caller's config is what actually takes effect
    for the rest of the process, which is expected OTel behavior, not a bug.
    """
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return

    trace.set_tracer_provider(build_tracer_provider(service_name, console_export, otlp_endpoint))


def instrument_app(app: FastAPI) -> None:
    """Auto-instruments every route with a span (OTel) and exposes
    Prometheus-format metrics at /metrics — request count, latency
    histograms, and in-progress requests, all from
    prometheus-fastapi-instrumentator's defaults.
    """
    FastAPIInstrumentor.instrument_app(app)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)
