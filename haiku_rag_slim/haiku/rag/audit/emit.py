"""Where an audit record goes, and the decorator that records an outcome.

The sink is a dedicated logger, `haiku.rag.audit`, deliberately NOT the
`haiku.rag` logger the 118 diagnostic calls use. That logger is configured at
`logging.py:27` and `:52` with `propagate = False`, and `configure_cli_logging`
detaches every root handler at `logging.py:41-43` -- which is why a stdlib
record cannot reach an OTel or syslog handler at all, measured as 0 captured
spans. An audit stream that inherited those settings would inherit that defect,
so this logger propagates and nothing here silences the root.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from haiku.rag.audit.record import (
    LOCAL_PROCESS,
    ActorSource,
    AuditEvent,
    Component,
    Event,
    Outcome,
    serialize,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

#: Its own logger so a deployment can route audit records without routing
#: diagnostics, and so turning tracing off cannot turn auditing off.
LOGGER_NAME = "haiku.rag.audit"

_P = ParamSpec("_P")
_R = TypeVar("_R")


def logger() -> logging.Logger:
    """The audit logger. Propagates on purpose, unlike `haiku.rag`."""
    return logging.getLogger(LOGGER_NAME)


def emit(record: AuditEvent) -> None:
    """Write one record as one line.

    Never raises: an audit sink that takes the application down when it is
    misconfigured is worse than one that reports. Failing loudly on audit
    failure is V-222485/V-222486 and belongs with the spool that can tell a
    write failure from a forwarding backlog, not here.
    """
    try:
        logger().info(serialize(record))
    except Exception:  # noqa: BLE001 - see the docstring
        logging.getLogger(LOGGER_NAME).debug("audit emit failed", exc_info=True)


def audited(
    event: Event,
    component: Component,
    *,
    target: Callable[..., str | None] | None = None,
    detail_of: Callable[[Any], dict[str, Any]] | None = None,
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Record the outcome of an async call, then let the call's result stand.

    Wrapping rather than hand-editing each site is deliberate: it makes "no
    handler discards an outcome" a property of the code's shape instead of a
    rule someone has to keep following. A bare `except Exception: return None`
    cannot coexist with this decorator, because the decorator is what observes
    the exception.

    The failure path re-raises. Returning a benign value on failure is the
    defect itself -- `[]` from a store error is indistinguishable from "nothing
    matched" -- so the caller is told, and the record says the same thing the
    caller was told.

    `detail_of` maps a successful result onto extra detail. It exists because
    an operation can succeed and still not do what was asked: deleting an id
    that is not there returns False without raising. That is NOT a failure --
    nothing went wrong -- but it must be distinguishable from a delete that
    removed something, and from a store error. Three states, recorded as
    outcome=success/deleted=true, outcome=success/deleted=false, and
    outcome=failure.
    """

    def decorate(call: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        @functools.wraps(call)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            detail: dict[str, Any] = {"tool": call.__name__}
            try:
                result = await call(*args, **kwargs)
            except Exception as exc:
                emit(
                    AuditEvent(
                        event=event,
                        component=component,
                        outcome=Outcome.FAILURE,
                        actor=LOCAL_PROCESS,
                        actor_source=ActorSource.NO_AUTHENTICATION_SURFACE,
                        target=_resolve(target, kwargs),
                        detail={**detail, "error": type(exc).__name__},
                    )
                )
                raise
            if detail_of is not None:
                try:
                    detail |= detail_of(result)
                except Exception:  # noqa: BLE001 - detail is evidence, not control flow
                    detail |= {"detail_of": "raised"}
            emit(
                AuditEvent(
                    event=event,
                    component=component,
                    outcome=Outcome.SUCCESS,
                    actor=LOCAL_PROCESS,
                    actor_source=ActorSource.NO_AUTHENTICATION_SURFACE,
                    target=_resolve(target, kwargs),
                    detail=detail,
                )
            )
            return result

        return wrapper

    return decorate


def _resolve(
    target: Callable[..., str | None] | None, kwargs: dict[str, Any]
) -> str | None:
    """The record's `target`, from the call's own arguments."""
    if target is None:
        return None
    try:
        return target(**kwargs)
    except Exception:  # noqa: BLE001 - a target is evidence, not control flow
        return None
