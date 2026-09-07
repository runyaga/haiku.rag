"""Audit records, separate from diagnostic logging."""

from haiku.rag.audit.emit import (
    LOGGER_NAME,
    AuditWriteError,
    audited,
    emit,
    logger,
)
from haiku.rag.audit.forward import backlog, drain, orphaned
from haiku.rag.audit.record import (
    LOCAL_PROCESS,
    UNAUTHENTICATED,
    ActorSource,
    AuditEvent,
    Component,
    Event,
    Outcome,
    serialize,
)
from haiku.rag.audit.sinks import FileSink, Sink, SyslogTLSSink, rfc5424
from haiku.rag.audit.spool import BrokenChain, Segment, read_segment, segments

__all__ = [
    "LOCAL_PROCESS",
    "AuditWriteError",
    "BrokenChain",
    "Segment",
    "FileSink",
    "Sink",
    "SyslogTLSSink",
    "backlog",
    "drain",
    "orphaned",
    "read_segment",
    "rfc5424",
    "segments",
    "LOGGER_NAME",
    "UNAUTHENTICATED",
    "ActorSource",
    "AuditEvent",
    "Component",
    "Event",
    "Outcome",
    "audited",
    "emit",
    "logger",
    "serialize",
]
