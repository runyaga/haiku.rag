"""One audit record, and the single function that serializes it.

Separate from the `logging` module on purpose. The 118 existing `logger.*` calls
are diagnostics: they may be silenced, reformatted or dropped without anyone
being wronged. An audit record carries an obligation instead -- it must state
who did what, to what, and whether it worked -- so the two cannot share a type
or a sink.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

#: The value `actor` takes when no identity was established. A sentinel rather
#: than None or "": a record whose actor is absent is indistinguishable from a
#: record whose actor was never looked for, and an assessor cannot tell them
#: apart. Paired with an `ActorSource` that says how it came to be unknown.
UNAUTHENTICATED = "unauthenticated"

#: The value for a process with no caller to authenticate -- a CLI invocation,
#: the stdio MCP transport, the local TUI. Not the same as "unauthenticated",
#: which means a caller existed and was not identified.
LOCAL_PROCESS = "local_process"


class Event(StrEnum):
    """What happened. A closed vocabulary, so an event cannot be invented.

    Open-ended event names defeat the purpose: an assessor filtering for
    document deletions cannot know that a later release renamed the string.
    """

    DOCUMENT_CREATE = "document.create"
    DOCUMENT_UPDATE = "document.update"
    DOCUMENT_DELETE = "document.delete"
    DOCUMENT_READ = "document.read"
    CHUNK_DELETE_ALL = "chunk.delete_all"
    SEARCH = "search"
    ASK = "ask"
    ANALYZE = "analyze"
    TOOL_INVOKE = "tool.invoke"
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    CONFIG_LOAD = "config.load"
    SERVICE_START = "service.start"
    SERVICE_STOP = "service.stop"
    AUDIT_FAILURE = "audit.failure"


class Component(StrEnum):
    """Which part of the application emitted the record (V-222474)."""

    MCP = "mcp"
    INGESTER_API = "ingester.api"
    STORE = "store"
    CLIENT = "client"
    CLI = "cli"
    TELEMETRY = "telemetry"


class Outcome(StrEnum):
    """Whether it worked (V-222476). There is deliberately no default.

    An audit record cannot state an outcome the code discarded, so the type
    refuses to guess one: every call site must decide.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"


class ActorSource(StrEnum):
    """How `actor` was established.

    This field exists because both failure directions are defects. Recording an
    invented actor is a lie about attribution; recording `unauthenticated` for a
    request that WAS authenticated is a lie in the other direction, and
    `ingester/api/auth.py` does authenticate a bearer token. The source makes
    the difference recordable instead of guessable.

    None of these satisfies V-222449 or V-222477 on their own: a shared
    credential identifies a credential, not a person, which is the case
    V-222478 covers when it asks for the individual identities of group account
    users.
    """

    #: A bearer token matched the configured one. Identifies the credential.
    SHARED_BEARER_TOKEN = "shared_bearer_token"
    #: The control plane is running with no `auth_token`, so every request is
    #: allowed and no caller is identified.
    AUTH_DISABLED = "auth_disabled"
    #: There is no caller to authenticate: CLI, stdio MCP, local TUI.
    NO_AUTHENTICATION_SURFACE = "no_authentication_surface"
    #: A credential WAS required and the caller did not present a matching one.
    #: Distinct from AUTH_DISABLED, which says the opposite about the server's
    #: configuration -- conflating them records "the control plane was open" for
    #: a request that was in fact refused, which is the kind of false statement
    #: this whole field exists to prevent.
    CREDENTIAL_REJECTED = "credential_rejected"


def _now() -> datetime:
    """The current instant, timezone-aware and in UTC."""
    return datetime.now(UTC)


class AuditEvent(BaseModel):
    """A single audit record.

    Frozen and `extra="forbid"`: a record is evidence, so it may not be mutated
    after the fact and may not carry a field nothing declared.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    #: Always present and always UTC. Records after the first in a second used
    #: to carry no time at all, because rich's handler omits repeated ones --
    #: which is why this is a required field on a model rather than a formatter
    #: option (V-222446, V-222473, V-222498, V-222499).
    timestamp: datetime = Field(default_factory=_now)
    event: Event
    component: Component
    outcome: Outcome
    actor: str = Field(min_length=1)
    actor_source: ActorSource
    target: str | None = None
    client_addr: str | None = None
    user_agent: str | None = None
    app_id: str = "haiku.rag"
    host: str = Field(default_factory=socket.gethostname)
    pid: int = Field(default_factory=os.getpid)
    trace_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @field_validator("actor")
    @classmethod
    def _actor_is_not_blank(cls, value: str) -> str:
        """An actor of whitespace is not an identity.

        `min_length=1` accepts a single space, which produces a record
        asserting an authenticated caller whose identity is `" "` -- consistent
        with its source and useless for attribution, which is the shape of
        defect this model exists to refuse.
        """
        if not value.strip():
            raise ValueError("actor must not be blank")
        return value

    @model_validator(mode="after")
    def _actor_agrees_with_its_source(self) -> AuditEvent:
        """The actor and how it was established must not contradict each other.

        Catches the two lies this record exists to prevent: claiming no
        authentication surface on a request that carried a credential, and
        claiming a named principal on one that did not.
        """
        if self.actor_source is ActorSource.SHARED_BEARER_TOKEN and self.actor in {
            UNAUTHENTICATED,
            LOCAL_PROCESS,
        }:
            raise ValueError(
                f"actor_source={self.actor_source.value} names an authenticated "
                f"caller, so actor may not be {self.actor!r}"
            )
        if (
            self.actor_source is ActorSource.NO_AUTHENTICATION_SURFACE
            and self.actor != LOCAL_PROCESS
        ):
            raise ValueError(
                f"actor_source={self.actor_source.value} means there was no "
                f"caller to authenticate, so actor must be {LOCAL_PROCESS!r}"
            )
        if (
            self.actor_source
            in {ActorSource.AUTH_DISABLED, ActorSource.CREDENTIAL_REJECTED}
            and self.actor != UNAUTHENTICATED
        ):
            raise ValueError(
                f"actor_source={self.actor_source.value} means the caller was "
                f"not identified, so actor must be {UNAUTHENTICATED!r}"
            )
        return self

    @model_validator(mode="after")
    def _timestamp_is_utc(self) -> AuditEvent:
        """A naive timestamp cannot be mapped to UTC, which V-222498 requires."""
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return self


def serialize(record: AuditEvent) -> str:
    """One record as one JSON line, with a millisecond RFC 3339 timestamp.

    A line rather than a nested structure so a forwarder can treat the stream as
    append-only and a SIEM can parse it without a schema.
    """
    payload = record.model_dump(mode="json")
    payload["timestamp"] = (
        record.timestamp.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
