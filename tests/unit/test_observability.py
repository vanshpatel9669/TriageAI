"""Verifies Prometheus metrics and OTel tracing are genuinely wired up —
not just that create_app() doesn't crash with them enabled.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from triageai.infrastructure.observability import build_tracer_provider, instrument_app


def test_build_tracer_provider_sets_the_service_name():
    provider = build_tracer_provider("triageai-test", console_export=False)
    assert provider.resource.attributes["service.name"] == "triageai-test"


def test_build_tracer_provider_with_console_export_can_start_and_end_a_span():
    provider = build_tracer_provider("triageai-test", console_export=True)
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.attribute", "value")
    # shutdown() forces the BatchSpanProcessor to flush synchronously before
    # this test function returns. Without it, the export happens on a
    # background thread that can still be running when pytest closes its
    # stdout capture, producing a real (if harmless) "I/O operation on
    # closed file" exception in the background thread — found by actually
    # reading the test's own stderr output, not by assuming a passing
    # assertion meant the test was clean.
    provider.shutdown()


def test_build_tracer_provider_with_no_exporters_still_works_safely():
    provider = build_tracer_provider("triageai-test", console_export=False, otlp_endpoint=None)
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span("test-span"):
        pass
    # Spans can be created even with no configured exporter — they're just
    # not sent anywhere. Should not raise.


def test_metrics_endpoint_reflects_real_request_counts():
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    instrument_app(app)

    with TestClient(app) as client:
        client.get("/ping")
        client.get("/ping")
        metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    assert "text/plain" in metrics_response.headers["content-type"]
    body = metrics_response.text
    assert 'handler="/ping"' in body
    # 2 pings + the /metrics request itself in flight isn't counted yet at
    # scrape time, so assert on the ping count specifically, not a total.
    ping_lines = [
        line
        for line in body.splitlines()
        if line.startswith("http_requests_total") and "/ping" in line
    ]
    assert any(line.endswith("2.0") for line in ping_lines), ping_lines


def test_fastapi_instrumentation_produces_real_spans_with_http_attributes():
    """Uses an isolated TracerProvider + InMemorySpanExporter (not the
    process-global one configure_tracing() manages) so this test can inspect
    exact span content, independent of whatever other tests already claimed
    the global provider — see observability.py's docstring on why the
    global provider can only be set once per process.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = FastAPI()

    @app.get("/traced")
    def traced() -> dict:
        return {"ok": True}

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    with TestClient(app) as client:
        response = client.get("/traced")
    assert response.status_code == 200

    spans = exporter.get_finished_spans()
    server_spans = [s for s in spans if s.kind.name == "SERVER"]
    assert len(server_spans) == 1
    span = server_spans[0]
    assert span.name == "GET /traced"
    assert span.attributes["http.method"] == "GET"
    assert span.attributes["http.status_code"] == 200
    assert span.attributes["http.route"] == "/traced"
