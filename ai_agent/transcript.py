"""Persist user-visible supervisor turns, independent of checkpoint compaction."""

from __future__ import annotations

import hashlib
import logging

from asgiref.sync import sync_to_async
from django.db import IntegrityError, close_old_connections
from langchain.agents.middleware import AgentMiddleware

from ai_agent.memory.namespaces import user_id_from_config
from ai_agent.messages import message_text
from ai_agent.models import AgentMessage

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "Here is a summary of the conversation to date:"


class TranscriptMiddleware(AgentMiddleware):
    """Write human/assistant text from the supervisor turn to AgentMessage."""

    def before_agent(self, state, runtime):
        persist_transcript(state, runtime)
        return None

    async def abefore_agent(self, state, runtime):
        await _apersist_transcript(state, runtime)
        return None

    def after_agent(self, state, runtime):
        persist_transcript(state, runtime)
        return None

    async def aafter_agent(self, state, runtime):
        await _apersist_transcript(state, runtime)
        return None


def persist_transcript(state, runtime) -> int:
    """Save newly visible supervisor messages. Returns how many rows were created."""
    if not isinstance(state, dict):
        return 0
    if state.get("__interrupt__"):
        return 0
    config = _config_from_runtime(runtime)
    thread_id = _thread_id_from_config(config)
    user_id = user_id_from_config(config)
    if not thread_id or not user_id or "::" in thread_id:
        return 0
    rows = visible_transcript_rows(state.get("messages") or [])
    if not rows:
        return 0
    return _save_rows(user_id=user_id, thread_id=thread_id, rows=rows)


_apersist_transcript = sync_to_async(persist_transcript, thread_sensitive=True)


def visible_transcript_rows(messages) -> list[dict]:
    rows = []
    for message in messages or []:
        if _is_summary(message) or _has_tool_calls(message):
            continue
        role = _role(message)
        if role is None:
            continue
        content = message_text(message).strip()
        if not content:
            continue
        rows.append(
            {
                "role": role,
                "content": content,
                "external_id": _external_id(message, role, content),
            }
        )
    return rows


def _save_rows(*, user_id, thread_id: str, rows: list[dict]) -> int:
    from django.contrib.auth import get_user_model

    close_old_connections()
    User = get_user_model()
    try:
        user = User.objects.filter(pk=user_id).first()
    except (TypeError, ValueError, OverflowError):
        logger.warning("Transcript skip: invalid user_id=%r", user_id)
        return 0
    if user is None:
        return 0
    created = 0
    for row in rows:
        try:
            _, was_created = AgentMessage.objects.get_or_create(
                thread_id=thread_id,
                external_id=row["external_id"],
                defaults={
                    "user": user,
                    "role": row["role"],
                    "content": row["content"],
                },
            )
        except IntegrityError:
            logger.debug(
                "Transcript row race on thread_id=%s external_id=%s",
                thread_id,
                row["external_id"],
            )
            continue
        if was_created:
            created += 1
    return created


def _is_summary(message) -> bool:
    extra = getattr(message, "additional_kwargs", None) or {}
    if isinstance(extra, dict) and extra.get("lc_source") == "summarization":
        return True
    return message_text(message).lstrip().startswith(_SUMMARY_PREFIX)


def _has_tool_calls(message) -> bool:
    calls = getattr(message, "tool_calls", None)
    return bool(calls)


def _role(message) -> str | None:
    name = type(message).__name__.lower()
    if "human" in name:
        return AgentMessage.ROLE_USER
    if name.startswith("ai") or name == "aimessage":
        return AgentMessage.ROLE_ASSISTANT
    kind = str(getattr(message, "type", None) or getattr(message, "role", "")).lower()
    if kind in {"human", "user"}:
        return AgentMessage.ROLE_USER
    if kind in {"ai", "assistant"}:
        return AgentMessage.ROLE_ASSISTANT
    return None


def _external_id(message, role: str, content: str) -> str:
    msg_id = getattr(message, "id", None)
    if msg_id not in (None, ""):
        return str(msg_id)[:128]
    digest = hashlib.sha256(f"{role}:{content}".encode()).hexdigest()
    return digest[:64]


def _thread_id_from_config(config) -> str | None:
    if not isinstance(config, dict):
        return None
    cfg = dict(config.get("configurable") or {})
    extra = config.get("context")
    if isinstance(extra, dict):
        cfg = {**cfg, **extra}
    thread_id = cfg.get("thread_id")
    if thread_id in (None, ""):
        return None
    return str(thread_id)


def _config_from_runtime(runtime) -> dict:
    config = getattr(runtime, "config", None) if runtime is not None else None
    if isinstance(config, dict):
        return config
    try:
        from langgraph.config import get_config

        live = get_config()
        if isinstance(live, dict):
            return live
    except Exception:
        logger.debug("No LangGraph config available for transcript")
    return {}
