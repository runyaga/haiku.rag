"""Optional Logfire instrumentation for the conformance matrices.

Off unless `LOGFIRE_TOKEN` is set, so `make check` runs offline and in CI with
no telemetry and no network. When on, every matrix cell emits one span carrying
its identity and outcome, so a run can be sliced by attribute, operation,
search type, fusion mode and source count in the Logfire UI.

`environment` is what you filter on: it defaults to `fusionlab` and is
overridable with `FUSIONLAB_ENV`, so a one-off run can be isolated from the
baseline without touching code.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

DEFAULT_ENVIRONMENT = "fusionlab"
SERVICE_NAME = "fusionlab"


def environment_name(env: dict[str, str] | None = None) -> str:
    """The Logfire environment to report under."""
    source = os.environ if env is None else env
    return source.get("FUSIONLAB_ENV") or DEFAULT_ENVIRONMENT


def configure(
    env: dict[str, str] | None = None,
    configurer: Callable[..., Any] | None = None,
) -> str | None:
    """Turn telemetry on if a token is present. Returns the environment, or None.

    `configurer` is injected so the decision can be tested without a network
    call; production passes None and gets `logfire.configure`.
    """
    source = os.environ if env is None else env
    token = source.get("LOGFIRE_TOKEN")
    if not token:
        return None
    name = environment_name(source)
    if configurer is None:  # pragma: no cover - the real call needs a network
        import logfire

        configurer = logfire.configure
    configurer(
        token=token,
        environment=name,
        service_name=SERVICE_NAME,
        console=False,
    )
    return name


@contextmanager
def cell_span(name: str, **attributes: Any) -> Iterator[Any]:
    """One span per matrix cell, or nothing at all when telemetry is off.

    Import is deferred so the module stays importable without logfire, and the
    no-token path never touches it.
    """
    if not os.environ.get("LOGFIRE_TOKEN"):
        yield None
        return
    import logfire  # pragma: no cover - only on the instrumented path

    with logfire.span(name, **attributes) as span:  # pragma: no cover
        yield span
