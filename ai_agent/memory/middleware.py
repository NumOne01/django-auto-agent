"""Recall long-term memory into the system prompt; debounce background writes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from ai_agent.conf import AgentSettings, get_agent_settings
from ai_agent.messages import message_text
from ai_agent.memory.namespaces import (
    bind_namespace,
    memory_namespaces,
    user_id_from_config,
)

logger = logging.getLogger(__name__)


class MemoryRecallMiddleware(AgentMiddleware):
    """Inject this layer's profile, facts, and episodes before each model call."""

    def __init__(self, layer: str, *, settings: AgentSettings | None = None):
        super().__init__()
        self.layer = layer
        self._settings = settings

    def before_agent(self, state, runtime):
        from ai_agent.memory.tools import reset_created_memory_ids

        reset_created_memory_ids()
        return None

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    def wrap_model_call(self, request, handler):
        return handler(self._with_memories(request))

    async def awrap_model_call(self, request, handler):
        return await handler(await self._awith_memories(request))

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)

    def _memory_on(self) -> bool:
        agent_settings = self._settings or get_agent_settings()
        return agent_settings.memory_enabled

    def _with_memories(self, request):
        if not self._memory_on():
            return request
        block = recall_memory_block(
            self.layer,
            request.messages,
            runtime=getattr(request, "runtime", None),
        )
        return _apply_memory_block(request, block)

    async def _awith_memories(self, request):
        if not self._memory_on():
            return request
        block = await arecall_memory_block(
            self.layer,
            request.messages,
            runtime=getattr(request, "runtime", None),
        )
        return _apply_memory_block(request, block)


class MemoryWriteMiddleware(AgentMiddleware):
    """Submit the transcript to the background memory agent after a turn."""

    def __init__(self, layer: str, *, agent_runtime=None):
        super().__init__()
        self.layer = layer
        self._agent_runtime = agent_runtime

    def after_agent(self, state, runtime):
        self._submit(state, runtime)
        return None

    async def aafter_agent(self, state, runtime):
        self._submit(state, runtime)
        return None

    def _submit(self, state, runtime):
        agent_settings = (
            self._agent_runtime.settings
            if self._agent_runtime is not None
            else get_agent_settings()
        )
        if not agent_settings.is_memory_background():
            return
        if not isinstance(state, dict):
            return
        if state.get("__interrupt__"):
            return
        messages = state.get("messages") or []
        if not messages:
            return
        config = _config_from_runtime(runtime)
        store = getattr(runtime, "store", None) if runtime is not None else None
        if self._agent_runtime is not None:
            try:
                self._agent_runtime.submit_memory(
                    self.layer, messages, config, store=store
                )
            except Exception:
                logger.exception(
                    "Background memory submit failed for %s", self.layer
                )
            try:
                self._agent_runtime.submit_curator(
                    self.layer, config, store=store
                )
            except Exception:
                logger.exception(
                    "Background curator submit failed for %s", self.layer
                )
            return
        from ai_agent.memory.managers import submit_memory

        try:
            submit_memory(
                self.layer, messages, config, store=store, settings=agent_settings
            )
        except Exception:
            logger.exception("Background memory submit failed for %s", self.layer)
        from ai_agent.memory.curator import submit_curator

        try:
            submit_curator(self.layer, config, store=store, settings=agent_settings)
        except Exception:
            logger.exception("Background curator submit failed for %s", self.layer)


class MemoryCuratorMiddleware(AgentMiddleware):
    """Schedule the per-user curator after a hot-path turn."""

    def __init__(self, layer: str, *, agent_runtime=None):
        super().__init__()
        self.layer = layer
        self._agent_runtime = agent_runtime

    def after_agent(self, state, runtime):
        self._submit(state, runtime)
        return None

    async def aafter_agent(self, state, runtime):
        self._submit(state, runtime)
        return None

    def _submit(self, state, runtime):
        agent_settings = (
            self._agent_runtime.settings
            if self._agent_runtime is not None
            else get_agent_settings()
        )
        if not agent_settings.memory_curator_enabled:
            return
        if not isinstance(state, dict):
            return
        if state.get("__interrupt__"):
            return
        config = _config_from_runtime(runtime)
        store = getattr(runtime, "store", None) if runtime is not None else None
        if self._agent_runtime is not None:
            try:
                self._agent_runtime.submit_curator(
                    self.layer, config, store=store
                )
            except Exception:
                logger.exception(
                    "Hot-path curator submit failed for %s", self.layer
                )
            return
        from ai_agent.memory.curator import submit_curator

        try:
            submit_curator(self.layer, config, store=store, settings=agent_settings)
        except Exception:
            logger.exception("Hot-path curator submit failed for %s", self.layer)


def recall_memory_block(layer: str, messages, *, runtime=None) -> str:
    store = _store_from_runtime(runtime)
    if store is None:
        return ""
    query, namespaces, user_id, limit = _recall_args(layer, messages, runtime)
    if not user_id:
        return ""
    profile = _load_items(
        store, bind_namespace(namespaces["profile"], user_id=user_id), query=""
    )
    facts = _load_items(
        store,
        bind_namespace(namespaces["semantic"], user_id=user_id),
        query=query,
        limit=limit,
    )
    episodes = _load_items(
        store,
        bind_namespace(namespaces["episodes"], user_id=user_id),
        query=query,
        limit=limit,
    )
    playbook = _load_items(
        store, bind_namespace(namespaces["playbook"], user_id=user_id), query=""
    )
    global_prompt, user_prompt = _recall_overlays(store, user_id=user_id, layer=layer)
    return format_memory_block(
        profile,
        facts,
        episodes,
        playbook,
        global_prompt=global_prompt,
        user_prompt=user_prompt,
    )


async def arecall_memory_block(layer: str, messages, *, runtime=None) -> str:
    store = _store_from_runtime(runtime)
    if store is None:
        return ""
    query, namespaces, user_id, limit = _recall_args(layer, messages, runtime)
    if not user_id:
        return ""
    profile = await _aload_items(
        store, bind_namespace(namespaces["profile"], user_id=user_id), query=""
    )
    facts = await _aload_items(
        store,
        bind_namespace(namespaces["semantic"], user_id=user_id),
        query=query,
        limit=limit,
    )
    episodes = await _aload_items(
        store,
        bind_namespace(namespaces["episodes"], user_id=user_id),
        query=query,
        limit=limit,
    )
    playbook = await _aload_items(
        store, bind_namespace(namespaces["playbook"], user_id=user_id), query=""
    )
    global_prompt, user_prompt = await _arecall_overlays(
        store, user_id=user_id, layer=layer
    )
    return format_memory_block(
        profile,
        facts,
        episodes,
        playbook,
        global_prompt=global_prompt,
        user_prompt=user_prompt,
    )


def format_memory_block(
    profile,
    facts,
    episodes,
    playbook=None,
    *,
    global_prompt: str = "",
    user_prompt: str = "",
) -> str:
    overlay_parts = []
    if global_prompt:
        overlay_parts.append(
            "<global_prompt>\n" + global_prompt.strip() + "\n</global_prompt>"
        )
    if user_prompt:
        overlay_parts.append(
            "<user_prompt>\n" + user_prompt.strip() + "\n</user_prompt>"
        )
    parts = []
    if profile:
        parts.append("<profile>\n" + _format_items(profile) + "\n</profile>")
    if facts:
        parts.append("<facts>\n" + _format_items(facts) + "\n</facts>")
    if episodes:
        parts.append("<episodes>\n" + _format_items(episodes) + "\n</episodes>")
    if playbook:
        parts.append("<playbook>\n" + _format_items(playbook) + "\n</playbook>")
    chunks = []
    if overlay_parts:
        chunks.append("Prompt overlays:\n" + "\n".join(overlay_parts))
    if parts:
        chunks.append("Long-term memory:\n" + "\n".join(parts))
    return "\n\n".join(chunks)


def _apply_memory_block(request, block: str):
    if not block:
        return request
    if request.system_message is not None:
        content = [
            *request.system_message.content_blocks,
            {"type": "text", "text": f"\n\n{block}"},
        ]
        system_message = SystemMessage(content=content)
    else:
        system_message = SystemMessage(content=block)
    return request.override(system_message=system_message)


def _recall_overlays(store, *, user_id: str, layer: str) -> tuple[str, str]:
    try:
        from ai_agent.memory.prompt_optimize import recall_overlays

        return recall_overlays(store, user_id=user_id, layer=layer)
    except Exception:
        logger.exception("Prompt overlay recall failed for %s/%s", user_id, layer)
        return "", ""


async def _arecall_overlays(store, *, user_id: str, layer: str) -> tuple[str, str]:
    try:
        from ai_agent.memory.prompt_optimize import arecall_overlays

        return await arecall_overlays(store, user_id=user_id, layer=layer)
    except Exception:
        logger.exception("Prompt overlay recall failed for %s/%s", user_id, layer)
        return "", ""


def _recall_args(layer: str, messages, runtime):
    config = _config_from_runtime(runtime)
    user_id = user_id_from_config(config)
    query = _latest_text(messages)
    return query, memory_namespaces(layer), user_id, get_agent_settings().memory_search_limit


def _load_items(store, namespace: tuple[str, ...], *, query: str, limit: int = 10):
    if _on_store_event_loop(store):
        logger.debug(
            "Skipping sync memory search on the store event loop for %s", namespace
        )
        return []
    try:
        kwargs: dict[str, Any] = {"limit": limit}
        if query:
            kwargs["query"] = query
        return list(store.search(namespace, **kwargs) or [])
    except asyncio.InvalidStateError:
        logger.debug("Sync memory search blocked on event loop for %s", namespace)
        return []
    except Exception:
        logger.exception("Memory search failed for %s", namespace)
        return []


async def _aload_items(store, namespace: tuple[str, ...], *, query: str, limit: int = 10):
    try:
        kwargs: dict[str, Any] = {"limit": limit}
        if query:
            kwargs["query"] = query
        asearch = getattr(store, "asearch", None)
        if asearch is not None:
            return list(await asearch(namespace, **kwargs) or [])
        return list(store.search(namespace, **kwargs) or [])
    except Exception:
        logger.exception("Memory search failed for %s", namespace)
        return []


def _on_store_event_loop(store) -> bool:
    loop = getattr(store, "_loop", None)
    if loop is None:
        return False
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _format_items(items) -> str:
    lines = []
    for item in items:
        value = getattr(item, "value", item)
        if not isinstance(value, (str, int, float, bool)):
            try:
                value = json.dumps(value, ensure_ascii=False, default=str)
            except TypeError:
                value = str(value)
        lines.append(str(value))
    return "\n".join(lines)


def _latest_text(messages) -> str:
    for message in reversed(list(messages or [])):
        text = message_text(message)
        if text and text.strip():
            return text.strip()
    return ""


def _store_from_runtime(runtime):
    store = getattr(runtime, "store", None) if runtime is not None else None
    if store is not None:
        return store
    try:
        from langgraph.config import get_store

        return get_store()
    except Exception:
        return None


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
        logger.debug(
            "langgraph.config.get_config unavailable for memory middleware",
            exc_info=True,
        )
    return {}
