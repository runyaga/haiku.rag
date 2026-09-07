"""A durable, tamper-evident spool of audit records.

Records are appended to a per-process SEGMENT inside a spool directory, never
to one shared file. haiku.rag runs as short-lived CLI invocations and as a
concurrent worker pool that already identifies its processes as
``f"{os.getpid()}-{uuid4().hex[:8]}"`` (``ingester/workers/pool.py``), so a
single append-only file would be written by unrelated processes at once.

Each line carries the hash of the line before it. A record cannot be removed,
reordered or edited without breaking the chain, which is what makes the spool
evidence rather than a list -- ASD STIG V-222507 asks for cryptographic
mechanisms protecting the integrity of audit information.

Writing is separated from forwarding on purpose. A short-lived process must be
able to finish and exit the moment its record is durable; whether that record
has reached a SIEM is a different question with a different lifetime, and
answering it in-process would tie a CLI invocation to a network round trip.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Opens a chain. Distinguishable from a real digest at a glance.
GENESIS = "0" * 64

#: A segment is claimed by the process that created it. `.open` while its
#: writer may still be running; renamed to `.sealed` when it exits cleanly.
OPEN_SUFFIX = ".open"
SEALED_SUFFIX = ".sealed"


def chain(previous: str, payload: str) -> str:
    """The digest binding one record to the one before it."""
    return hashlib.sha256(f"{previous}\n{payload}".encode()).hexdigest()


class Segment:
    """One process's append-only slice of the spool.

    Not a context manager by accident of style: `close()` seals the segment so
    a forwarder can tell "this writer finished" from "this writer may still be
    running", and a process that dies without sealing leaves a `.open` segment
    that must still be drained rather than one that is silently skipped.
    """

    def __init__(self, directory: Path, name: str | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.name = name or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.path = directory / f"{self.name}{OPEN_SUFFIX}"
        self._previous = GENESIS
        self._count = 0

    @property
    def head(self) -> str:
        """The chain head: the digest of the last record written."""
        return self._previous

    @property
    def count(self) -> int:
        return self._count

    def append(self, payload: str) -> str:
        """Write one record, durably, and return the new chain head.

        `fsync` on every record rather than on close. A record that is only in
        the page cache when the machine loses power is a record that did not
        survive, and the whole reason to spool before forwarding is to survive.
        The cost is one sync per audited event, which is the price of the
        guarantee rather than an oversight.
        """
        digest = chain(self._previous, payload)
        line = json.dumps(
            {"prev": self._previous, "digest": digest, "record": payload},
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous = digest
        self._count += 1
        return digest

    def close(self) -> Path:
        """Seal the segment, so a forwarder knows its writer finished."""
        if not self.path.exists():
            return self.path
        sealed = self.path.with_suffix(SEALED_SUFFIX)
        self.path.rename(sealed)
        self.path = sealed
        return sealed


class BrokenChain(Exception):
    """A spool segment was edited, truncated or reordered."""


def read_segment(path: Path) -> list[str]:
    """Every record in a segment, verifying the chain as it goes.

    Raises rather than skipping a bad line: a spool that quietly drops the
    record someone tampered with defeats the point of chaining it.
    """
    records: list[str] = []
    previous = GENESIS
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BrokenChain(f"{path.name}:{number} is not valid JSON") from exc
        if entry.get("prev") != previous:
            raise BrokenChain(
                f"{path.name}:{number} claims to follow {entry.get('prev')!r}, "
                f"but the previous record hashes to {previous!r} -- a record "
                f"was removed, reordered or inserted"
            )
        expected = chain(previous, entry["record"])
        if entry.get("digest") != expected:
            raise BrokenChain(
                f"{path.name}:{number} has digest {entry.get('digest')!r}, "
                f"expected {expected!r} -- the record was edited"
            )
        records.append(entry["record"])
        previous = expected
    return records


def head_of(path: Path) -> str:
    """The chain head a segment ends at, for a forwarder to send onward."""
    previous = GENESIS
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            previous = json.loads(line)["digest"]
        except (json.JSONDecodeError, KeyError):
            break
    return previous


def segments(directory: Path) -> list[Path]:
    """Every segment awaiting a forwarder, sealed ones first.

    Sealed first because their writer is known to have finished, so they can be
    drained whole. An `.open` segment may still be growing -- or may belong to a
    process that died, which is why it is returned at all rather than skipped.
    """
    if not directory.is_dir():
        return []
    sealed = sorted(directory.glob(f"*{SEALED_SUFFIX}"))
    still_open = sorted(directory.glob(f"*{OPEN_SUFFIX}"))
    return sealed + still_open
