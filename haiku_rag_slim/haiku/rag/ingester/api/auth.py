import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from haiku.rag.audit import (
    UNAUTHENTICATED,
    ActorSource,
    AuditEvent,
    Component,
    Event,
    Outcome,
    emit,
)

_bearer = HTTPBearer(auto_error=False)

#: What `actor` says when a bearer token matched. The token itself is never
#: recorded -- a credential in an audit record is a credential in a log file,
#: which is what ASD STIG V-222444 forbids. This names the credential's ROLE,
#: which is as much as a shared token can honestly identify.
BEARER_ACTOR = "ingester-api-token"


def _record(
    request: Request,
    event: Event,
    outcome: Outcome,
    actor: str,
    source: ActorSource,
) -> None:
    """Record one authentication decision.

    Both directions are recorded. An authentication surface that logs only its
    failures cannot answer "who was in here", which is the question V-222462
    exists to make answerable.

    `source` is passed by the caller, never inferred from `actor`. An earlier
    version derived it -- anything that was not the bearer actor became
    AUTH_DISABLED -- so a REJECTED credential was recorded as "the control
    plane had no token configured". That is a false statement about the
    server's configuration, and it is the exact class of lie `actor_source`
    exists to prevent. Found by an independent review.
    """
    emit(
        AuditEvent(
            event=event,
            component=Component.INGESTER_API,
            outcome=outcome,
            actor=actor,
            actor_source=source,
            target=request.url.path,
            client_addr=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    )


async def require_auth(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Bearer-token gate. When the app has no auth_token configured, all
    requests are allowed (and a warning was logged at startup). Otherwise
    the Authorization header's bearer must match exactly."""
    expected = getattr(request.app.state, "auth_token", None)
    if expected is None:
        # Allowed, and recorded as allowed-without-authentication rather than
        # not recorded: "the control plane was open" is the fact an assessor
        # needs, and a missing record does not state it.
        _record(
            request,
            Event.AUTH_SUCCESS,
            Outcome.SUCCESS,
            UNAUTHENTICATED,
            ActorSource.AUTH_DISABLED,
        )
        return
    if creds is None or not secrets.compare_digest(creds.credentials, expected):
        _record(
            request,
            Event.AUTH_FAILURE,
            Outcome.DENIED,
            UNAUTHENTICATED,
            ActorSource.CREDENTIAL_REJECTED,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _record(
        request,
        Event.AUTH_SUCCESS,
        Outcome.SUCCESS,
        BEARER_ACTOR,
        ActorSource.SHARED_BEARER_TOKEN,
    )
