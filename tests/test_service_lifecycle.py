"""A SIGTERM must leave a service.stop record behind.

`systemctl stop` sends SIGTERM. Python's default handler terminates the
interpreter without unwinding, so a `finally` that emits the shutdown record
never runs, and the ordinary stop path -- the one an operator uses every day --
produces no audit record at all. ASD STIG V-222469 asks for shutdown to be
recorded; a record that only appears when the process crashes does not answer
it.

These tests assert the mechanism rather than the wiring: that the context
manager turns the signal into cancellation of the awaiting task, and that a
cancellation reaching `serve`'s `try` still leaves the record.
"""

import asyncio
import json
import os
import signal

import pytest

from haiku.rag.app import _terminate_gracefully


@pytest.mark.asyncio
async def test_sigterm_cancels_the_awaiting_task_instead_of_killing_the_process():
    """Without this, the process dies here and nothing below the yield runs."""
    unwound = False

    async def serve_forever() -> None:
        nonlocal unwound
        try:
            with _terminate_gracefully():
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass
        finally:
            unwound = True

    task = asyncio.create_task(serve_forever())
    await asyncio.sleep(0)  # let the handler install before signalling
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, timeout=5)

    assert unwound, "SIGTERM did not unwind the awaiting task"


@pytest.mark.asyncio
async def test_handlers_are_removed_on_exit():
    """The context manager must not leave the loop's SIGTERM rewired.

    A leaked handler would cancel whatever task happened to be current the next
    time the signal arrived, which is a worse failure than the one it fixes.
    """
    loop = asyncio.get_running_loop()

    async def briefly() -> None:
        with _terminate_gracefully():
            await asyncio.sleep(0)

    await briefly()

    # Reinstalling must succeed and must be ours to remove: if the manager had
    # leaked its own, this would be replacing a live handler.
    assert loop.remove_signal_handler(signal.SIGTERM) is False, (
        "a SIGTERM handler survived the context manager"
    )


@pytest.mark.asyncio
async def test_cancellation_still_records_service_stop(caplog):
    """The `except` must swallow CancelledError so the `finally` can record.

    Mirrors `HaikuRAGApp.serve`'s structure: if `asyncio.CancelledError` is not
    caught alongside KeyboardInterrupt, it propagates out of the coroutine and
    the stop record is emitted but the shutdown reads as a crash.
    """
    from haiku.rag import audit

    def _service(event: audit.Event, outcome: audit.Outcome) -> None:
        audit.emit(
            audit.AuditEvent(
                event=event,
                component=audit.Component.MCP,
                outcome=outcome,
                actor=audit.LOCAL_PROCESS,
                actor_source=audit.ActorSource.NO_AUTHENTICATION_SURFACE,
                target="stdio",
            )
        )

    async def serve_like() -> None:
        _service(audit.Event.SERVICE_START, audit.Outcome.SUCCESS)
        outcome = audit.Outcome.SUCCESS
        try:
            with _terminate_gracefully():
                await asyncio.sleep(30)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except Exception:
            outcome = audit.Outcome.FAILURE
            raise
        finally:
            _service(audit.Event.SERVICE_STOP, outcome)

    with caplog.at_level("INFO", logger="haiku.rag.audit"):
        task = asyncio.create_task(serve_like())
        await asyncio.sleep(0)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(task, timeout=5)

    records = [json.loads(r.message) for r in caplog.records]
    stops = [r for r in records if r["event"] == "service.stop"]
    assert stops, "SIGTERM produced no service.stop record"
    assert stops[-1]["outcome"] == "success", (
        "an ordinary stop was recorded as a failure"
    )
