"""FastAPI Bearer gate for the CopilotKit AG-UI mount."""

from __future__ import annotations

from typing import Optional

from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from fastapi import Header, HTTPException, Request

_PERIOD_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_bearer_token(authorization: str | None) -> str:
    """Return the raw JWT from ``Authorization: Bearer <token>``."""
    if not authorization:
        raise ValueError("Missing authorization header")
    scheme, _, remainder = authorization.partition(" ")
    if scheme.lower() != "bearer" or not remainder.strip():
        raise ValueError("Invalid authorization header")
    return remainder.strip()


async def require_bearer_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Reject CopilotKit requests that do not carry a valid product JWT.

    OPTIONS is skipped so browser CORS preflight can succeed without a token.
    """
    if request.method == "OPTIONS":
        return None
    _enforce_http_throttle(request)
    try:
        token = parse_bearer_token(authorization)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Unauthorized") from exc

    from ai_agent.conf import resolve_authenticate_token

    authenticate_token = resolve_authenticate_token()
    try:
        return await sync_to_async(authenticate_token, thread_sensitive=True)(token)
    except (ImproperlyConfigured, HTTPException):
        raise
    except Exception as exc:
        from django.db import DatabaseError

        if isinstance(exc, DatabaseError):
            raise
        raise HTTPException(status_code=401, detail="Unauthorized") from exc


def _enforce_http_throttle(request: Request) -> None:
    from django.core.cache import cache

    from ai_agent.conf import get_agent_settings

    parsed = _parse_http_throttle(get_agent_settings().http_throttle)
    if parsed is None:
        return
    num_requests, duration = parsed
    ident = _client_ip(request)
    key = f"throttle_ai_agent_http_{ident}"
    try:
        count = cache.incr(key)
    except ValueError:
        if cache.add(key, 1, duration):
            count = 1
        else:
            try:
                count = cache.incr(key)
            except ValueError:
                cache.set(key, 1, duration)
                count = 1
    if count > num_requests:
        raise HTTPException(status_code=429, detail="Too many requests")


def _parse_http_throttle(rate: str) -> tuple[int, int] | None:
    text = str(rate or "").strip().lower()
    if not text:
        return None
    num, _, period = text.partition("/")
    try:
        num_requests = int(num)
    except ValueError:
        num_requests = 60
        period = "minute"
    if num_requests < 1:
        return None
    duration = _PERIOD_SECONDS.get((period or "m")[:1], 60)
    return num_requests, duration


def _client_ip(request: Request) -> str:
    from ai_agent.conf import get_agent_settings

    peer = ""
    if request.client and request.client.host:
        peer = request.client.host
    count = get_agent_settings().http_trusted_proxy_count
    if count <= 0:
        return peer or "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return peer or "unknown"
    parts = [part.strip() for part in forwarded.split(",") if part.strip()]
    if not parts:
        return peer or "unknown"
    if len(parts) < count:
        return parts[0]
    return parts[-count]
