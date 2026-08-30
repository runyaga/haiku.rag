"""Telemetry configuration. Pure: the configurer is injected, nothing is sent."""

import pytest

from fusionlab.observability import (
    DEFAULT_ENVIRONMENT,
    SERVICE_NAME,
    cell_span,
    configure,
    environment_name,
)


class TestEnvironmentName:
    def test_it_defaults_when_unset(self):
        assert environment_name({}) == DEFAULT_ENVIRONMENT

    def test_an_override_wins(self):
        assert environment_name({"FUSIONLAB_ENV": "one-off"}) == "one-off"

    def test_an_empty_override_falls_back(self):
        """An exported-but-blank var must not produce a nameless environment."""
        assert environment_name({"FUSIONLAB_ENV": ""}) == DEFAULT_ENVIRONMENT


class TestConfigure:
    def test_no_token_means_telemetry_stays_off(self):
        """So the fast gate runs offline and in CI with no network."""
        calls = []
        assert configure({}, configurer=lambda **kw: calls.append(kw)) is None
        assert calls == []

    def test_a_blank_token_also_stays_off(self):
        calls = []
        assert (
            configure({"LOGFIRE_TOKEN": ""}, configurer=lambda **kw: calls.append(kw))
            is None
        )
        assert calls == []

    def test_a_token_turns_it_on_and_returns_the_environment(self):
        calls = []
        result = configure(
            {"LOGFIRE_TOKEN": "tok"}, configurer=lambda **kw: calls.append(kw)
        )
        assert result == DEFAULT_ENVIRONMENT
        assert len(calls) == 1

    def test_it_passes_the_token_environment_and_service(self):
        calls = []
        configure(
            {"LOGFIRE_TOKEN": "tok", "FUSIONLAB_ENV": "experiment-7"},
            configurer=lambda **kw: calls.append(kw),
        )
        assert calls[0] == {
            "token": "tok",
            "environment": "experiment-7",
            "service_name": SERVICE_NAME,
            "console": False,
        }

    def test_the_console_exporter_is_off(self):
        """Otherwise every span would print into the test output."""
        calls = []
        configure({"LOGFIRE_TOKEN": "t"}, configurer=lambda **kw: calls.append(kw))
        assert calls[0]["console"] is False


class TestCellSpan:
    def test_it_yields_none_when_telemetry_is_off(self, monkeypatch):
        monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
        with cell_span("x", a=1) as span:
            assert span is None

    def test_the_body_still_runs_with_telemetry_off(self, monkeypatch):
        """Instrumentation must never change whether a test executes."""
        monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
        ran = False
        with cell_span("x"):
            ran = True
        assert ran

    def test_an_exception_in_the_body_propagates(self, monkeypatch):
        monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
        with pytest.raises(ValueError, match="boom"), cell_span("x"):
            raise ValueError("boom")
