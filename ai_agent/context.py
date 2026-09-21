"""Per-request user identity for in-process tool execution.

The model never receives a user id argument; tools always run as this user.
"""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
import logging

from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

_current_user: ContextVar = ContextVar("ai_agent_user", default=None)


class AgentUserContext:
    def __init__(self, user):
        self.user = user
        self._token = None

    def __enter__(self):
        self._token = _current_user.set(self.user)
        return self.user

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _current_user.reset(self._token)
        return False


def peek_current_user():
    return _current_user.get()


def get_current_user():
    user = peek_current_user()
    if user is None:
        raise RuntimeError("No authenticated user in AI agent context")
    return user


def _configurable(config) -> dict:
    if not config:
        return {}
    if not isinstance(config, dict):
        config = dict(config)
    merged = dict(config.get("configurable") or {})
    extra = config.get("context")
    if isinstance(extra, dict):
        merged = {**merged, **extra}
    return merged


def _auth_user_id(cfg: dict):
    """Trusted identity injected by LangGraph ``@auth.authenticate``."""
    user_id = cfg.get("langgraph_auth_user_id")
    if user_id not in (None, ""):
        return user_id
    auth_user = cfg.get("langgraph_auth_user")
    if isinstance(auth_user, dict):
        return auth_user.get("identity")
    identity = getattr(auth_user, "identity", None)
    if identity not in (None, ""):
        return identity
    return None


def _lookup_user(*, user_id=None, phone=None):
    from django.contrib.auth import get_user_model
    from django.db.utils import OperationalError

    User = get_user_model()
    try:
        if user_id not in (None, ""):
            return User.objects.get(pk=user_id)
        if phone:
            field = User.USERNAME_FIELD
            return User.objects.get(**{field: str(phone).strip()})
    except (User.DoesNotExist, ValueError, TypeError) as exc:
        logger.warning("AI agent user lookup failed", exc_info=exc)
        raise RuntimeError(_user_not_found_message(user_id=user_id, phone=phone)) from exc
    except OperationalError as exc:
        if _is_unreachable_db_error(exc):
            logger.exception("AI agent database unreachable")
            raise RuntimeError(_db_unreachable_message(exc)) from exc
        raise
    return None


def resolve_agent_user(config=None):
    """Resolve the acting user from LangGraph auth, then test-only fallbacks.

    Production Agent Server identity comes from ``langgraph_auth_user`` after
    JWT validation. Client-supplied ``user_id`` / ``phone_number`` are ignored
    when that auth user is present. Configurable and Studio-phone fallbacks
    run only under Django ``TESTING``.
    """
    from django.conf import settings

    from ai_agent.conf import get_agent_settings
    from ai_agent.db import refresh_db_connections

    refresh_db_connections()
    cfg = _configurable(config)
    auth_user_id = _auth_user_id(cfg)
    if auth_user_id not in (None, ""):
        return _lookup_user(user_id=auth_user_id)

    if not getattr(settings, "TESTING", False):
        return None

    user_id = cfg.get("user_id")
    if user_id not in (None, ""):
        return _lookup_user(user_id=user_id)

    login = (
        cfg.get("phone_number")
        or cfg.get("phone")
        or cfg.get("username")
        or get_agent_settings().studio_user_phone
    )
    if login:
        return _lookup_user(phone=login)
    return None


def agent_user_from_config(config=None, *, required: bool = False):
    """Context manager: keep an existing chat user, else bind from Studio config."""
    existing = peek_current_user()
    if existing is not None:
        return nullcontext(existing)
    user = resolve_agent_user(config)
    if user is None:
        if required:
            raise RuntimeError(_missing_user_message())
        return nullcontext(None)
    return AgentUserContext(user)


async def agent_user_from_config_async(config=None, *, required: bool = False):
    """Async variant: ORM lookup runs in a sync thread (Studio uses astream)."""
    existing = peek_current_user()
    if existing is not None:
        return nullcontext(existing)
    user = await sync_to_async(resolve_agent_user, thread_sensitive=True)(config)
    if user is None:
        if required:
            raise RuntimeError(_missing_user_message())
        return nullcontext(None)
    return AgentUserContext(user)


def _missing_user_message() -> str:
    from django.conf import settings

    if getattr(settings, "TESTING", False):
        return (
            "No AI agent user. In Studio set configurable.user_id or "
            "configurable.phone_number, or AI_AGENT_STUDIO_USER_PHONE."
        )
    return (
        "No authenticated AI agent user. Send Authorization: Bearer <access_token>."
    )


def _user_not_found_message(*, user_id=None, phone=None) -> str:
    from django.conf import settings

    if getattr(settings, "TESTING", False):
        if user_id not in (None, ""):
            return f"AI agent user_id={user_id!r} was not found"
        return f"AI agent studio user phone={phone!r} was not found"
    return "AI agent user was not found."


def _db_unreachable_message(exc=None) -> str:
    from django.conf import settings

    if getattr(settings, "TESTING", False):
        detail = f": {exc}" if exc else ""
        return f"AI agent database error{detail}"
    return "AI agent cannot reach the database."


def _is_unreachable_db_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "database table is locked" in msg:
        return False
    return any(
        token in msg
        for token in (
            "could not connect",
            "connection refused",
            "could not translate host",
            "name or service not known",
            "timeout expired",
            "server closed the connection",
            "could not resolve",
        )
    )
