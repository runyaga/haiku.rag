"""Where a forwarder sends audit records.

A sink either accepts a whole batch or does not. There is no partial success:
a forwarder that deleted the records a sink half-took would lose exactly the
ones it could not account for.
"""

from __future__ import annotations

import socket
import ssl
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from haiku.rag.config.models import AuditSinkConfig

#: RFC 5424 fixed parts. facility 13 (log audit) * 8 + severity 5 (notice).
SYSLOG_PRI = 13 * 8 + 5
SYSLOG_VERSION = 1
NILVALUE = "-"


class Sink(Protocol):
    """Accepts a batch of serialized records, or raises.

    `chain_head` is the spool digest the batch ends at. It travels with the
    batch so a receiver can verify the sender's integrity claim without
    parsing our record format -- carrying it is the difference between hashing
    the spool for ourselves and offering the guarantee to whoever holds the
    records afterwards (ASD STIG V-222507).
    """

    def send(self, records: Sequence[str], chain_head: str) -> None: ...


def rfc5424(
    record: str, *, hostname: str, app_name: str, chain_head: str, sd_id: str
) -> str:
    """One record as an RFC 5424 frame.

    The chain head rides in a structured-data element rather than in the
    message, because SD is RFC 5424's own extension point (section 6.3) and a
    conforming receiver must accept an SD-ID it does not know -- so the head
    survives a pipeline that rewrites the message.

    What this does NOT establish is that a given SIEM EXTRACTS the head. That
    is a property of the receiver, it is not knowable from here, and an earlier
    version of this claimed it. The frame is conformant and puts the head where
    a receiver CAN find it; whether one does is a per-deployment question, which
    is why `sd_id` is configuration rather than a constant.
    """
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    structured = f'[{sd_id} chainHead="{chain_head}"]'
    return (
        f"<{SYSLOG_PRI}>{SYSLOG_VERSION} {stamp} {hostname or NILVALUE} "
        f"{app_name} {NILVALUE} {NILVALUE} {structured} {record}"
    )


class FileSink:
    """Append records to a file. The honest floor.

    Present because "off-load to a different system" is often done by a log
    shipper reading a file, and a library that offered only network sinks would
    push those deployments into configuring nothing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, records: Sequence[str], chain_head: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record + "\n")
            handle.write(f'{{"chain_head":"{chain_head}"}}\n')
            handle.flush()


class SyslogTLSSink:
    """RFC 5424 over TLS, octet-counted framing (RFC 6587 section 3.4.1).

    Octet counting rather than newline termination: a record containing a
    newline would otherwise be read as two messages, and an audit record whose
    content can split its own frame is a record an attacker can forge around.
    """

    def __init__(
        self, address: str, config: AuditSinkConfig | None = None, *, sd_id: str = ""
    ) -> None:
        host, _, port = address.rpartition(":")
        self.host = host or "localhost"
        self.port = int(port)
        self.config = config
        self.sd_id = (sd_id or (config.syslog_sd_id if config else "") or "").strip()
        # Stripped before the check: an SD-ID of spaces satisfies every type
        # annotation and names nothing, which is the same defect as an actor
        # of whitespace.
        if not self.sd_id:
            raise ValueError(
                "a syslog sink needs an SD-ID to carry the chain head in; see "
                "AuditSinkConfig.syslog_sd_id"
            )
        self.hostname = socket.gethostname()

    def _connect(self) -> socket.socket:
        raw = socket.create_connection((self.host, self.port), timeout=10)
        if self.config is None or self.config.tls_cert is None:
            return raw
        context = ssl.create_default_context(
            cafile=str(self.config.tls_ca or "") or None
        )
        context.load_cert_chain(str(self.config.tls_cert), str(self.config.tls_key))
        return context.wrap_socket(raw, server_hostname=self.host)

    def send(self, records: Sequence[str], chain_head: str) -> None:
        with self._connect() as connection:
            for record in records:
                frame = rfc5424(
                    record,
                    hostname=self.hostname,
                    app_name="haiku.rag",
                    chain_head=chain_head,
                    sd_id=self.sd_id,
                ).encode()
                connection.sendall(f"{len(frame)} ".encode() + frame)
