"""Drains the spool to its sinks, and deletes only what was accepted.

A separate long-lived process owns the spool DIRECTORY. It is not an in-process
task: haiku.rag runs as short-lived CLI invocations, so an in-process forwarder
loses whatever has not drained when the command exits -- and a worker pool would
have several of them writing one file at once.

Ownership is the load-bearing idea. The writer owns its own segment and nothing
else; the forwarder owns every segment, including one whose writer died without
sealing it. A segment nobody owns is a segment nobody drains.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from haiku.rag.audit.spool import (
    OPEN_SUFFIX,
    SEALED_SUFFIX,
    BrokenChain,
    head_of,
    read_segment,
    segments,
)

if TYPE_CHECKING:
    from pathlib import Path

    from haiku.rag.audit.sinks import Sink

logger = logging.getLogger("haiku.rag.audit.forward")

#: How long a `.open` segment may go untouched before the forwarder treats its
#: writer as gone. Generous: draining a live writer's segment early is
#: harmless (records are re-read, not moved), but declaring a busy process dead
#: and racing it is not.
STALE_SECONDS = 60.0


class ForwardResult:
    """What one drain pass did, so a caller can act on a backlog."""

    def __init__(self) -> None:
        self.forwarded = 0
        self.segments_drained = 0
        self.pending = 0
        self.broken: list[str] = []
        self.failures: list[str] = []

    def __repr__(self) -> str:
        return (
            f"ForwardResult(forwarded={self.forwarded}, "
            f"segments={self.segments_drained}, pending={self.pending}, "
            f"broken={len(self.broken)}, failures={len(self.failures)})"
        )


def writer_is_live(path: Path) -> bool:
    """Whether the process that owns an open segment still exists.

    The segment name starts with the pid that created it. Not authoritative --
    a pid can be reused, and a container may not share a pid namespace -- so it
    is used to WITHHOLD draining rather than to permit it: an unknown answer
    leaves the segment alone, which costs a delay, where the opposite error
    costs the chain.
    """
    pid_text = path.name.split("-", 1)[0]
    if not pid_text.isdigit():
        return False
    try:
        os.kill(int(pid_text), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _drainable(path: Path, now: float) -> bool:
    """Whether this segment may be drained on this pass.

    A sealed segment always may: its writer is finished. An open one needs BOTH
    to have gone quiet AND its writer to be gone.

    Staleness alone is not enough, and the first version of this used only
    that. A live writer idle for the staleness window would have had its open
    segment drained and unlinked underneath it, and its next append would start
    a fresh file whose `prev` pointed at a digest no longer on disk --
    permanently breaking the chain for everything after. An idle process is not
    a dead one, and the fix for a busy service must not be to audit it less.
    """
    if path.name.endswith(SEALED_SUFFIX):
        return True
    try:
        quiet = (now - path.stat().st_mtime) > STALE_SECONDS
    except OSError:
        return False
    return quiet and not writer_is_live(path)


def drain(spool: Path, sinks: list[Sink], *, now: float | None = None) -> ForwardResult:
    """Send every drainable segment, deleting only what every sink accepted.

    Deletion is gated on acknowledgement from ALL sinks. Deleting after a
    partial success would lose the records the failed sink never got, and the
    spool exists precisely so that a sink being down costs nothing.
    """
    result = ForwardResult()
    moment = time.time() if now is None else now

    for path in segments(spool):
        if not _drainable(path, moment):
            result.pending += 1
            continue
        try:
            records = read_segment(path)
        except BrokenChain as exc:
            # Left in place, never deleted: a segment whose chain is broken is
            # the evidence that something tampered with it.
            result.broken.append(f"{path.name}: {exc}")
            logger.error("audit spool segment failed verification: %s", exc)
            continue
        if not records:
            path.unlink(missing_ok=True)
            continue

        accepted = True
        head = head_of(path)
        for sink in sinks:
            try:
                sink.send(records, head)
            except Exception as exc:  # noqa: BLE001 - a sink failing is expected
                accepted = False
                result.failures.append(f"{type(sink).__name__}: {exc}")
                logger.warning("audit sink %s rejected a batch: %s", type(sink), exc)
        if not accepted:
            result.pending += 1
            continue

        path.unlink(missing_ok=True)
        result.forwarded += len(records)
        result.segments_drained += 1
    return result


def backlog(spool: Path) -> int:
    """Records written and not yet forwarded, for V-222483's threshold."""
    total = 0
    for path in segments(spool):
        try:
            total += len(read_segment(path))
        except BrokenChain:
            # Counted as pending rather than skipped: a broken segment is still
            # un-forwarded, and hiding it would make the backlog look smaller
            # than it is.
            total += sum(1 for line in path.read_text().splitlines() if line.strip())
    return total


def orphaned(spool: Path, *, now: float | None = None) -> list[Path]:
    """Open segments whose writer is gone, named so an operator can see them."""
    moment = time.time() if now is None else now
    return [
        path
        for path in segments(spool)
        if path.name.endswith(OPEN_SUFFIX) and _drainable(path, moment)
    ]
