"""LangGraph Platform authentication and per-user thread authorization.

Validates the product JWT (same checks as Django API auth) and stamps
``metadata.owner`` so each customer only sees their own conversations.
"""

from __future__ import annotations

import os

from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from langgraph_sdk import Auth

from ai_agent.conf import resolve_authenticate_token, resolve_studio_django_settings_module
from ai_agent.http_auth import parse_bearer_token

auth = Auth()

HEALTHCHECK_IDENTITY = "__health__"
_FALLBACK_DISPLAY_NAME = "Authenticated user"


def _ensure_django():
    import django
    from django.conf import settings

    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        os.environ["DJANGO_SETTINGS_MODULE"] = resolve_studio_django_settings_module()
    if not settings.configured:
        django.setup()


def _is_healthcheck(path: str, method: str) -> bool:
    if (method or "GET").upper() != "GET":
        return False
    return (path or "").rstrip("/") == "/ok"


def _require_user(ctx: Auth.types.AuthContext) -> str:
    identity = getattr(ctx.user, "identity", None)
    if identity in (None, "", HEALTHCHECK_IDENTITY):
        raise Auth.exceptions.HTTPException(status_code=403, detail="Forbidden")
    return str(identity)


def _owner_filter(ctx: Auth.types.AuthContext) -> dict:
    return {"owner": _require_user(ctx)}


@auth.authenticate
async def authenticate(
    authorization: str | None = None,
    path: str = "",
    method: str = "GET",
) -> Auth.types.MinimalUserDict:
    if _is_healthcheck(path, method):
        return {"identity": HEALTHCHECK_IDENTITY, "is_authenticated": True}

    try:
        token = parse_bearer_token(authorization)
    except ValueError as exc:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Unauthorized"
        ) from exc

    _ensure_django()
    authenticate_token = resolve_authenticate_token()

    def _authenticate_sync(token):
        from ai_agent.db import refresh_db_connections

        refresh_db_connections()
        return authenticate_token(token)

    try:
        user = await sync_to_async(_authenticate_sync, thread_sensitive=True)(token)
    except ImproperlyConfigured:
        raise
    except Exception as exc:
        from django.db import DatabaseError

        if isinstance(exc, DatabaseError):
            raise
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Unauthorized"
        ) from exc

    display = _display_name(user)
    return {
        "identity": str(user.pk),
        "display_name": display,
        "is_authenticated": True,
    }


def _display_name(user) -> str:
    name = " ".join(
        part
        for part in (getattr(user, "first_name", None), getattr(user, "last_name", None))
        if part
    )
    if name:
        return name
    getter = getattr(user, "get_username", None)
    if callable(getter):
        username = str(getter() or "").strip()
        if username:
            return username
    return _FALLBACK_DISPLAY_NAME


@auth.on.threads
async def on_threads(ctx: Auth.types.AuthContext, value: dict):
    """Stamp and filter threads (and runs) by the JWT user. Conversations stay private."""
    filters = _owner_filter(ctx)
    metadata = value.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.update(filters)
    return filters


@auth.on.assistants
async def on_assistants(ctx: Auth.types.AuthContext, value: dict):
    """Let authenticated users read the built-in graph. Do not filter by owner.

    AG-UI clients and LangGraph Studio/SDK call ``assistants.search({ graphId })``.
    The graph registered in langgraph.json has no ``owner`` metadata, so an
    owner filter would return an empty list and surface
    ``No agent found with graph ID assistant``.
    """
    _require_user(ctx)
    action = getattr(ctx, "action", None)
    if action in ("create", "update", "delete"):
        raise Auth.exceptions.HTTPException(status_code=403, detail="Forbidden")
    return {}
