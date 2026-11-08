"""Persist user-visible supervisor turns, independent of checkpoint compaction."""

from __future__ import annotations

import hashlib
import json
import logging

from asgiref.sync import sync_to_async
from django.db import IntegrityError
from langchain.agents.middleware import AgentMiddleware

from ai_agent.memory.namespaces import user_id_from_config
from ai_agent.messages import message_text
from ai_agent.models import AgentMessage

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "Here is a summary of the conversation to date:"


class TranscriptMiddleware(AgentMiddleware):
    """Write the supervisor's visible turns, tool calls included, to AgentMessage."""

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
        if _is_summary(message):
            continue
        role = _role(message)
        if role is None:
            continue
        content = message_text(message).strip()
        tool_calls = _tool_calls(message) if role == AgentMessage.ROLE_ASSISTANT else []
        tool_call_id = (
            str(getattr(message, "tool_call_id", None) or "")[:128]
            if role == AgentMessage.ROLE_TOOL
            else ""
        )
        # A tool result marks its card complete even when it is empty.
        if not content and not tool_calls and role != AgentMessage.ROLE_TOOL:
            continue
        rows.append(
            {
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "tool_call_id": tool_call_id,
                "external_id": _external_id(
                    message, role, content, tool_calls, tool_call_id
                ),
            }
        )
    return rows


def _save_rows(*, user_id, thread_id: str, rows: list[dict]) -> int:
    from django.contrib.auth import get_user_model

    from ai_agent.db import refresh_db_connections

    refresh_db_connections()
    User = get_user_model()
    try:
        user = User.objects.filter(pk=user_id).first()
    except (TypeError, ValueError, OverflowError):
        logger.warning("Transcript skip: invalid user_id=%r", user_id)
        return 0
    if user is None:
        return 0
    saved = set(
        AgentMessage.objects.filter(
            thread_id=thread_id,
            external_id__in=[row["external_id"] for row in rows],
        ).values_list("external_id", flat=True)
    )
    last_saved = max(
        (index for index, row in enumerate(rows) if row["external_id"] in saved),
        default=-1,
    )
    created = 0
    for index, row in enumerate(rows):
        if row["external_id"] in saved:
            continue
        # History orders by created_at, so a row older than one already saved
        # would show at the bottom. Happens for tool rows of threads saved
        # before tool calls were stored.
        if index < last_saved:
            continue
        try:
            _, was_created = AgentMessage.objects.get_or_create(
                thread_id=thread_id,
                external_id=row["external_id"],
                defaults={
                    "user": user,
                    "role": row["role"],
                    "content": row["content"],
                    "tool_calls": row["tool_calls"],
                    "tool_call_id": row["tool_call_id"],
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


def _tool_calls(message) -> list[dict]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        if not isinstance(call, dict):
            continue
        calls.append(
            {
                "id": str(call.get("id") or ""),
                "name": str(call.get("name") or ""),
                # Args may hold datetimes or Decimals; JSONField needs plain JSON.
                "args": json.loads(json.dumps(call.get("args") or {}, default=str)),
            }
        )
    return calls


def _role(message) -> str | None:
    name = type(message).__name__.lower()
    if "human" in name:
        return AgentMessage.ROLE_USER
    if name.startswith("ai") or name == "aimessage":
        return AgentMessage.ROLE_ASSISTANT
    if name.startswith("tool"):
        return AgentMessage.ROLE_TOOL
    kind = str(getattr(message, "type", None) or getattr(message, "role", "")).lower()
    if kind in {"human", "user"}:
        return AgentMessage.ROLE_USER
    if kind in {"ai", "assistant"}:
        return AgentMessage.ROLE_ASSISTANT
    if kind == "tool":
        return AgentMessage.ROLE_TOOL
    return None


def _external_id(
    message, role: str, content: str, tool_calls: list[dict], tool_call_id: str
) -> str:
    msg_id = getattr(message, "id", None)
    if msg_id not in (None, ""):
        return str(msg_id)[:128]
    key = f"{role}:{content}"
    # Text-only keys stay as before so id-less rows already saved still match.
    if tool_calls:
        key += ":" + ",".join(call["id"] for call in tool_calls)
    if tool_call_id:
        key += f":{tool_call_id}"
    digest = hashlib.sha256(key.encode()).hexdigest()
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
