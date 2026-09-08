"""A durable, tamper-evident spool of audit records.

Records are appended to a per-process SEGMENT inside a spool directory, never
to one shared file. haiku.rag runs as short-lived CLI invocations and as a
concurrent worker pool that already identifies its processes as
``f"{os.getpid()}-{uuid4().hex[:8]}"`` (``ingester/workers/pool.py``), so a
single append-only file would be written by unrelated processes at once.

Each line carries the hash of the line before it, and sealing writes a footer
naming the record count and the final head. Editing, reordering, duplicating or
removing a record from the middle breaks the chain; removing records from the
END breaks the footer, which a chain alone does not catch -- a truncated
prefix is still a valid chain from GENESIS, and erasing your own last actions
is the obvious attack. Found by review, after the first version claimed
otherwise (ASD STIG V-222507).

**What this does and does not buy.** Against a process that crashes, or a
transport that garbles, it is conclusive. Against an attacker with write access
to the spool it raises the cost -- they must rewrite every subsequent digest
and the footer -- but it cannot be conclusive locally, because anything this
file can compute that attacker can recompute. The guarantee that survives them
is continuity at the RECEIVER: each forwarded batch carries its chain head, so
a SIEM that records the previous head can see a gap it never received. That
check lives on the far side and is not implemented here.

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

#: Written by `close()`. Its presence is what makes a sealed segment's LENGTH
#: attested rather than merely its contents.
FOOTER_KEY = "sealed"

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
        """Seal the segment: attest its length, then mark the writer finished.

        The footer is what makes truncation detectable. Without it a segment
        with its last N lines deleted is still a valid chain, so an attacker
        could erase the record of whatever they did last and leave no trace.
        """
        if not self.path.exists():
            return self.path
        footer = json.dumps(
            {FOOTER_KEY: True, "count": self._count, "head": self._previous},
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(footer + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        sealed = self.path.with_suffix(SEALED_SUFFIX)
        self.path.rename(sealed)
        self.path = sealed
        return sealed


class BrokenChain(Exception):
    """A spool segment was edited, truncated or reordered."""


def read_segment(path: Path) -> list[str]:
    """Every record in a segment, verifying the chain and the seal.

    A torn LAST line is tolerated and the valid prefix returned: a process
    killed mid-write leaves one, and discarding every record before it would
    lose exactly the audit trail of the crash. A break anywhere EARLIER is
    tampering and raises -- a spool that quietly drops the record someone
    edited defeats the point of chaining it.
    """
    lines = [
        (number, line)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    records: list[str] = []
    previous = GENESIS
    footer: dict[str, object] | None = None

    for position, (number, line) in enumerate(lines):
        is_last = position == len(lines) - 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if is_last:
                # A torn tail. Everything before it is intact and forwardable.
                break
            raise BrokenChain(f"{path.name}:{number} is not valid JSON") from exc
        if entry.get(FOOTER_KEY):
            footer = entry
            continue
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

    if path.name.endswith(SEALED_SUFFIX):
        if footer is None:
            raise BrokenChain(
                f"{path.name} is sealed but carries no footer -- it was "
                f"truncated past its own seal"
            )
        if footer.get("count") != len(records) or footer.get("head") != previous:
            raise BrokenChain(
                f"{path.name} attests {footer.get('count')} records ending at "
                f"{str(footer.get('head'))[:16]}..., but {len(records)} were "
                f"found ending at {previous[:16]}... -- records were removed "
                f"from the end"
            )
    return records


def head_of(path: Path) -> str:
    """The chain head a segment ends at, for a forwarder to send onward."""
    previous = GENESIS
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break
        if entry.get(FOOTER_KEY):
            continue
        if "digest" not in entry:
            break
        previous = entry["digest"]
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
