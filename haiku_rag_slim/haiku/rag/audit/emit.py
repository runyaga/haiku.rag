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

import atexit
import contextlib
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
from haiku.rag.audit.spool import Segment

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

#: Its own logger so a deployment can route audit records without routing
#: diagnostics, and so turning tracing off cannot turn auditing off.
LOGGER_NAME = "haiku.rag.audit"

_P = ParamSpec("_P")
_R = TypeVar("_R")

#: This process's segment. One per process, opened lazily: a CLI invocation
#: that audits nothing should not create a file.
_SEGMENT: Segment | None = None


def logger() -> logging.Logger:
    """The audit logger. Propagates on purpose, unlike `haiku.rag`."""
    return logging.getLogger(LOGGER_NAME)


class AuditWriteError(Exception):
    """The audit path failed and the configured policy is to halt.

    Its own type so a caller can tell "auditing failed" from any other error.
    ASD STIG V-222486 asks the application to shut down upon audit failure
    unless availability overrides it, and a deployment that chose `continue`
    gets a logged failure instead of this.
    """


def _spool() -> Segment | None:
    """This process's spool segment, opened once, or None when audit is off."""
    global _SEGMENT
    if _SEGMENT is not None:
        return _SEGMENT
    try:
        from haiku.rag.config import get_config

        settings = get_config().audit
    except Exception:  # noqa: BLE001 - no config is not an audit failure
        return None
    if not settings.enabled:
        return None
    _SEGMENT = Segment(settings.spool_path)
    atexit.register(_seal)
    return _SEGMENT


def _seal() -> None:
    """Seal this process's segment so a forwarder knows the writer finished."""
    global _SEGMENT
    if _SEGMENT is not None:
        with contextlib.suppress(OSError):
            _SEGMENT.close()
        _SEGMENT = None


def emit(record: AuditEvent) -> None:
    """Write one record: to the spool when configured, and to the logger.

    The logger always, because a deployment with no spool configured still
    wants its records somewhere. The spool additionally, because stderr is not
    durable and cannot be off-loaded -- which is the whole of ASD STIG
    V-222481/V-222482.

    On failure the configured policy decides. `halt` raises `AuditWriteError`
    (V-222486's default); `continue` logs and returns, which is the rule's own
    "unless availability is an overriding concern" exception taken explicitly
    by a deployment rather than silently by this function.
    """
    payload = serialize(record)
    try:
        logger().info(payload)
    except Exception:  # noqa: BLE001 - the spool is the durable path
        logging.getLogger(LOGGER_NAME).debug("audit log emit failed", exc_info=True)

    segment = _spool()
    if segment is None:
        return
    try:
        segment.append(payload)
    except Exception as exc:
        from haiku.rag.config import get_config
        from haiku.rag.config.models import OnAuditFailure

        logging.getLogger(LOGGER_NAME).error(
            "audit spool write failed: %s", exc, exc_info=True
        )
        if get_config().audit.on_failure is OnAuditFailure.HALT:
            raise AuditWriteError(
                f"could not write an audit record to the spool: {exc}"
            ) from exc


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
