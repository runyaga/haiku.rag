from importlib import metadata

import logfire
import pytest

from haiku.rag import telemetry


@pytest.fixture
def captured_telemetry(monkeypatch):
    """Capture Logfire setup calls without touching a real exporter."""
    captured_configure: dict = {}
    captured_instrumentation: dict = {}

    def _fake_configure(**kwargs):
        captured_configure.update(kwargs)

    monkeypatch.setattr(logfire, "configure", _fake_configure)
    monkeypatch.setattr(
        logfire,
        "instrument_pydantic_ai",
        lambda **kwargs: captured_instrumentation.update(kwargs),
    )
    return captured_configure, captured_instrumentation


def test_default_service_name_used_when_env_unset(captured_telemetry, monkeypatch):
    captured_configure, _ = captured_telemetry
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("LOGFIRE_SERVICE_NAME", raising=False)

    telemetry.configure(service_name="haiku-ingester")

    assert captured_configure["service_name"] == "haiku-ingester"


def test_otel_service_name_overrides_default(captured_telemetry, monkeypatch):
    captured_configure, _ = captured_telemetry
    monkeypatch.setenv("OTEL_SERVICE_NAME", "customer-ingester")

    telemetry.configure(service_name="haiku-ingester")

    # Deferring to logfire (service_name=None) lets it read the env var,
    # so the customer's OTEL_SERVICE_NAME wins over our default.
    assert captured_configure["service_name"] is None


def test_logfire_service_name_overrides_default(captured_telemetry, monkeypatch):
    captured_configure, _ = captured_telemetry
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.setenv("LOGFIRE_SERVICE_NAME", "customer-ingester")

    telemetry.configure(service_name="haiku-ingester")

    assert captured_configure["service_name"] is None


def test_service_version_is_package_version(captured_telemetry, monkeypatch):
    captured_configure, _ = captured_telemetry
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("LOGFIRE_SERVICE_NAME", raising=False)

    telemetry.configure(service_name="haiku-rag")

    assert captured_configure["service_version"] == metadata.version("haiku.rag-slim")


def test_scrubbing_defaults_to_enabled(captured_telemetry):
    captured_configure, _ = captured_telemetry
    telemetry.configure(service_name="haiku-rag")

    # None is logfire's "scrubbing enabled" default.
    assert captured_configure["scrubbing"] is None


def test_scrubbing_can_be_disabled(captured_telemetry):
    captured_configure, _ = captured_telemetry
    telemetry.configure(service_name="evals", scrubbing=False)

    assert captured_configure["scrubbing"] is False


def test_content_is_excluded_from_instrumented_spans_by_default(captured_telemetry):
    _, captured_instrumentation = captured_telemetry

    telemetry.configure(service_name="haiku-rag")

    assert captured_instrumentation["include_content"] is False


def test_content_can_be_included_explicitly(captured_telemetry):
    _, captured_instrumentation = captured_telemetry

    telemetry.configure(service_name="haiku-rag", include_content=True)

    assert captured_instrumentation["include_content"] is True


def test_app_config_disables_content_export_by_default():
    from haiku.rag.config.models import AppConfig

    assert AppConfig().telemetry.include_content is False


def test_app_config_can_enable_content_export():
    from haiku.rag.config.models import AppConfig

    config = AppConfig.model_validate({"telemetry": {"include_content": True}})

    assert config.telemetry.include_content is True
