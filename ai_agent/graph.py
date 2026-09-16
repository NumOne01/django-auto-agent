"""Supervisor + per-app LangGraph subagents built from discovered API tools."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from typing import Optional

from django.apps import apps
from django.conf import settings
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolErrorMiddleware
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphBubbleUp, GraphInterrupt

from ai_agent.conf import AgentSettings, get_agent_settings, resolve_agent_middleware
from ai_agent.messages import message_text
from ai_agent.safety import (
    DOMAIN_SAFETY_POLICY,
    SafetyMiddleware,
    supervisor_safety_policy,
)
from ai_agent.limits import (
    apply_run_limits,
    bind_recursion_limit,
    build_limit_middleware,
)
from ai_agent.memory.namespaces import SUPERVISOR_LAYER
from ai_agent.context import agent_user_from_config, agent_user_from_config_async
from ai_agent.discovery import discover_endpoints
from ai_agent.tools import build_tool

logger = logging.getLogger(__name__)

TOOL_FAILURE_MESSAGE = (
    "This request could not be completed. "
    "Please try again with a narrower question."
)

WrapSubagent = Callable[[str, str, object], StructuredTool]


def build_supervisor(
    *,
    settings: Optional[AgentSettings] = None,
    endpoints: Optional[Sequence] = None,
    checkpointer=None,
    store=None,
    model: Optional[str] = None,
    supervisor_model: Optional[str] = None,
    subagent_model: Optional[str] = None,
    extra_middleware: Optional[Sequence] = None,
    stub_subagents: bool = False,
    wrap_subagent: Optional[WrapSubagent] = None,
    agent_runtime=None,
):
    agent_settings = _resolve_settings(settings, agent_runtime)
    supervisor_model = supervisor_model or model or agent_settings.supervisor_model
    subagent_model = subagent_model or model or agent_settings.subagent_model
    grouped_endpoints = _endpoints_by_app(
        _resolve_endpoints(endpoints, agent_runtime)
    )
    wrap = wrap_subagent or _wrap_subagent
    resolved_store = _bound_store(store, agent_settings)
    extra = extra_middleware or (
        agent_runtime.extra_middleware if agent_runtime is not None else None
    )

    supervisor_tools: list[StructuredTool] = []
    domain_lines: list[str] = []

    for app_label, app_endpoints in grouped_endpoints.items():
        app_tools = [build_tool(endpoint) for endpoint in app_endpoints]
        blurb = _app_description(app_label)
        if stub_subagents:
            supervisor_tools.append(_stub_subagent(app_label, blurb))
        else:
            subagent = _create_domain_agent(
                app_label,
                blurb,
                app_tools,
                model=subagent_model,
                middleware=_agent_middleware(
                    extra,
                    layer=app_label,
                    settings=agent_settings,
                    agent_runtime=agent_runtime,
                ),
                checkpointer=checkpointer,
                store=resolved_store,
                settings=agent_settings,
            )
            supervisor_tools.append(wrap(app_label, blurb, subagent))
        domain_lines.append(
            f"- {subagent_tool_name(app_label)}: {blurb} Delegate {app_label} tasks here."
        )

    if not supervisor_tools:
        raise RuntimeError("No API endpoints are exposed to the AI agent")

    if agent_settings.is_memory_hot():
        supervisor_tools.extend(_memory_tools("supervisor"))

    agent_kwargs = {
        "model": supervisor_model,
        "tools": supervisor_tools,
        "system_prompt": _supervisor_prompt(domain_lines, agent_settings),
        "middleware": _agent_middleware(
            extra,
            layer="supervisor",
            settings=agent_settings,
            agent_runtime=agent_runtime,
        ),
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    if resolved_store is not None:
        agent_kwargs["store"] = resolved_store
    return apply_run_limits(create_agent(**agent_kwargs), agent_settings)


def build_domain_agent(
    app_label: str,
    *,
    settings: Optional[AgentSettings] = None,
    endpoints: Optional[Sequence] = None,
    model: Optional[str] = None,
    extra_middleware: Optional[Sequence] = None,
    checkpointer=None,
    store=None,
    agent_runtime=None,
):
    """Compiled specialist for one Django app's exposed API tools."""
    agent_settings = _resolve_settings(settings, agent_runtime)
    extra = extra_middleware or (
        agent_runtime.extra_middleware if agent_runtime is not None else None
    )
    app_endpoints = _endpoints_by_app(
        _resolve_endpoints(endpoints, agent_runtime)
    ).get(app_label) or []
    if not app_endpoints:
        raise LookupError(f"No AI agent endpoints for app {app_label!r}")
    app_tools = [build_tool(endpoint) for endpoint in app_endpoints]
    blurb = _app_description(app_label)
    model = model or agent_settings.subagent_model
    return _create_domain_agent(
        app_label,
        blurb,
        app_tools,
        model=model,
        middleware=_agent_middleware(
            extra,
            layer=app_label,
            settings=agent_settings,
            agent_runtime=agent_runtime,
        ),
        checkpointer=checkpointer,
        store=_bound_store(store, agent_settings),
        settings=agent_settings,
    )


def subagent_tool_name(app_label: str) -> str:
    return f"call_{app_label}_agent"


def _resolve_settings(settings, agent_runtime) -> AgentSettings:
    if settings is not None:
        return settings
    if agent_runtime is not None:
        return agent_runtime.settings
    return get_agent_settings()


def _resolve_endpoints(endpoints, agent_runtime) -> Sequence:
    if endpoints is not None:
        return endpoints
    if agent_runtime is not None:
        return agent_runtime.endpoints
    return discover_endpoints()


def _endpoints_by_app(endpoints: Sequence) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for endpoint in endpoints:
        grouped.setdefault(endpoint.app_label, []).append(endpoint)
    return grouped


def _create_domain_agent(
    app_label: str,
    blurb: str,
    app_tools: list[StructuredTool],
    *,
    model,
    middleware,
    checkpointer=None,
    store=None,
    settings: Optional[AgentSettings] = None,
):
    agent_settings = settings or get_agent_settings()
    tools = list(app_tools)
    if agent_settings.is_memory_hot():
        tools.extend(_memory_tools(app_label))
    agent_kwargs = {
        "model": model,
        "tools": tools,
        "system_prompt": _subagent_prompt(app_label, blurb, app_tools, agent_settings),
        "middleware": middleware,
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    if store is not None:
        agent_kwargs["store"] = store
    return apply_run_limits(create_agent(**agent_kwargs), agent_settings)


def build_studio_graph(*, model: Optional[str] = None, agent_runtime=None):
    """Compiled supervisor for LangGraph Studio. The Agent Server owns persistence."""
    return build_supervisor(
        model=model,
        checkpointer=None,
        store=None,
        agent_runtime=agent_runtime,
        settings=None if agent_runtime is None else agent_runtime.settings,
        endpoints=None if agent_runtime is None else agent_runtime.endpoints,
    )


def build_memory_graph(*, store=None, settings: Optional[AgentSettings] = None):
    """Memory manager graph for background LangMem writes (graph id ``memory``)."""
    from ai_agent.memory.managers import build_memory_graph as _build

    return _build(store=store, settings=settings)


def build_curator_graph(*, store=None):
    """Per-user curator graph (graph id ``memory_curator``)."""
    from ai_agent.memory.curator import build_curator_graph as _build

    return _build(store=store)


def build_agui_http_app(compiled_graph):
    """AG-UI FastAPI app mounted next to Agent Server via ``langgraph.json`` ``http.app``.

    Studio keeps the default LangGraph routes. AG-UI clients should use
    ``http://127.0.0.1:2024/agui`` with ``Authorization: Bearer <access_token>``.
    """
    from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
    from fastapi import Depends, FastAPI

    from ai_agent.http_auth import require_bearer_user

    app = FastAPI(dependencies=[Depends(require_bearer_user)])
    add_langgraph_fastapi_endpoint(
        app=app,
        agent=LangGraphAgent(
            name="assistant",
            description=(
                f"Assistant for {get_agent_settings().platform.platform_name}. "
                "Operates as the authenticated user."
            ),
            graph=compiled_graph,
        ),
        path="/agui",
    )
    return app


def _agent_middleware(
    extra: Optional[Sequence] = None,
    *,
    layer: Optional[str] = None,
    settings: Optional[AgentSettings] = None,
    agent_runtime=None,
):
    agent_settings = _resolve_settings(settings, agent_runtime)
    items = [
        ToolErrorMiddleware(on_error=_on_tool_error),
        AgentUserMiddleware(),
        SafetyMiddleware(),
    ]
    if layer == SUPERVISOR_LAYER:
        from ai_agent.transcript import TranscriptMiddleware

        items.append(TranscriptMiddleware())
    items.extend(build_limit_middleware(agent_settings))
    if agent_settings.compaction_enabled:
        from ai_agent.compaction import build_compaction_middleware

        items.append(build_compaction_middleware(agent_settings))
    if layer and agent_settings.memory_enabled:
        from ai_agent.memory.middleware import (
            MemoryCuratorMiddleware,
            MemoryRecallMiddleware,
            MemoryWriteMiddleware,
        )

        items.append(
            MemoryRecallMiddleware(layer, settings=agent_settings)
        )
        if agent_settings.is_memory_background():
            items.append(
                MemoryWriteMiddleware(layer, agent_runtime=agent_runtime)
            )
        elif agent_settings.is_memory_hot():
            items.append(
                MemoryCuratorMiddleware(layer, agent_runtime=agent_runtime)
            )
    items.extend(resolve_agent_middleware(settings=agent_settings))
    if extra:
        items.extend(extra)
    return items


def _memory_tools(layer: str) -> list:
    from ai_agent.memory.tools import build_memory_tools

    return build_memory_tools(layer)


def _bound_store(store, agent_settings: AgentSettings):
    if store is None or not agent_settings.memory_enabled:
        return None
    return store


def _on_tool_error(exc, request) -> str:
    name = (request.tool_call or {}).get("name") or "tool"
    logger.exception("AI agent tool %s failed", name, exc_info=exc)
    return TOOL_FAILURE_MESSAGE


class AgentUserMiddleware(AgentMiddleware):
    """Bind in-process API tools to the JWT user from LangGraph auth.

    Prefers ``configurable.langgraph_auth_user`` / ``langgraph_auth_user_id``.
    Tests still bind via ``configurable.user_id`` / ``phone_number``.
    """

    def wrap_tool_call(self, request, handler):
        with agent_user_from_config(_runtime_config(request), required=True):
            return handler(request)

    async def awrap_tool_call(self, request, handler):
        bound = await agent_user_from_config_async(
            _runtime_config(request), required=True
        )
        with bound:
            return await handler(request)


def _runtime_config(request):
    """Collect user-binding keys from LangGraph Runtime and RunnableConfig.

    ``langgraph.runtime.Runtime`` no longer carries ``config``. Studio and tests
    still pass ``configurable.user_id`` on the invoke config, so also read
    ``get_config()`` and the child runnable config.
    """
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) if runtime is not None else None
    context = getattr(runtime, "context", None) if runtime is not None else None
    runnable = var_child_runnable_config.get()

    if not isinstance(config, dict) or not (
        config.get("configurable") or config.get("context")
    ):
        try:
            from langgraph.config import get_config

            live = get_config()
        except RuntimeError:
            live = None
        if isinstance(live, dict):
            config = {**(config if isinstance(config, dict) else {}), **live}

    merged: dict = {}
    for source in (runnable, config):
        if not isinstance(source, dict):
            continue
        configurable = {
            **(merged.get("configurable") or {}),
            **(source.get("configurable") or {}),
        }
        merged.update(source)
        if configurable:
            merged["configurable"] = configurable

    if isinstance(context, dict):
        existing = dict(merged.get("context") or {})
        merged["context"] = {**existing, **context}
        configurable = dict(merged.get("configurable") or {})
        for key in (
            "user_id",
            "phone_number",
            "phone",
            "langgraph_auth_user_id",
            "langgraph_auth_user",
        ):
            if key in context and key not in configurable:
                configurable[key] = context[key]
        if configurable:
            merged["configurable"] = configurable

    return merged or None


def build_checkpointer():
    if _use_memory_checkpointer():
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    conninfo = _postgres_conninfo()
    pool = ConnectionPool(
        conninfo=conninfo,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        min_size=1,
        max_size=4,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def _use_memory_checkpointer() -> bool:
    if getattr(settings, "TESTING", False):
        return True
    engine = settings.DATABASES["default"]["ENGINE"]
    return engine.endswith("sqlite3")


def _postgres_conninfo() -> str:
    from urllib.parse import quote_plus

    db = settings.DATABASES["default"]
    user = quote_plus(str(db.get("USER") or ""))
    password = quote_plus(str(db.get("PASSWORD") or ""))
    host = db.get("HOST") or "localhost"
    port = db.get("PORT") or "5432"
    name = db.get("NAME")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def _app_description(app_label: str) -> str:
    try:
        config = apps.get_app_config(app_label)
    except LookupError:
        logger.warning(
            "App %s is not installed; router will get a generic blurb. "
            "Set AppConfig.agent_description.",
            app_label,
        )
        return f"Handle {app_label} operations."
    custom = getattr(config, "agent_description", None)
    if custom:
        return custom
    logger.warning(
        "App %s has no agent_description; router will get a generic blurb. "
        "Set AppConfig.agent_description.",
        app_label,
    )
    verbose = getattr(config, "verbose_name", None) or app_label
    return f"Handle {verbose} operations for the authenticated user."


def _memory_prompt_section(agent_settings: AgentSettings) -> str:
    if not agent_settings.memory_enabled:
        return ""
    parts = [
        "Long-term memory is in place. Recalled profile, facts, episodes, "
        "and optional prompt overlays may appear in this prompt; use them "
        "when they are relevant."
    ]
    if agent_settings.is_memory_background():
        parts.append(
            "A background memory saver records durable information after the turn; "
            "you do not need to save memories yourself."
        )
    return " " + " ".join(parts)


def _subagent_prompt(
    app_label: str,
    blurb: str,
    tools: list[StructuredTool],
    agent_settings: Optional[AgentSettings] = None,
) -> str:
    agent_settings = agent_settings or get_agent_settings()
    names = ", ".join(tool.name for tool in tools)
    entity_terms = agent_settings.platform.entity_terms
    prompt = (
        f"You are the {app_label} specialist for an authenticated customer. {blurb} "
        f"Use these tools to fulfill the request: {names}. "
        f"Call tools instead of guessing {entity_terms}. "
        "Return a concise result the supervisor can relay to the user. "
        "Do not mention internal HTTP status codes unless the action failed. "
        f"Do not invent {entity_terms}. "
        f"{DOMAIN_SAFETY_POLICY}"
    )
    prompt += _memory_prompt_section(agent_settings)
    if agent_settings.is_memory_hot():
        prompt += (
            " Use the memory tools to patch this domain's standing profile "
            "(never a second document) and to create, update, or delete facts and "
            "episodes when the user shares durable information or an approach, "
            "success or failure, is worth reusing. Do not store the current request."
        )
    return prompt


def _supervisor_prompt(
    domain_lines: list[str],
    agent_settings: Optional[AgentSettings] = None,
) -> str:
    agent_settings = agent_settings or get_agent_settings()
    catalog = "\n".join(domain_lines)
    platform = agent_settings.platform
    prompt = (
        f"You are a helpful assistant for {platform.platform_name}. "
        "You operate only as the logged-in customer; never impersonate another user. "
        "Route each request to the matching domain subagent. "
        "You may call multiple tools when the user asks about more than one domain.\n\n"
        f"Available capabilities:\n{catalog}\n\n"
        "If a tool result is an error, tell the user you could not complete "
        "the request and to try again. Do not mention tool names, stack traces, "
        "or internal limits. "
        f"Do not invent {platform.entity_terms}. "
        f"{supervisor_safety_policy(platform)}"
    )
    prompt += _memory_prompt_section(agent_settings)
    if agent_settings.is_memory_hot():
        prompt += (
            " Use the memory tools for the overall profile (name, language, "
            "reply style, standing constraints; never a second document) plus "
            "cross-domain facts and episodes. Do not store the current task on "
            "the overall profile. Domain specialists keep their own memories."
        )
    return prompt


def _wrap_subagent(app_label: str, blurb: str, subagent) -> StructuredTool:
    tool_name = subagent_tool_name(app_label)
    description = (
        f"{blurb} Pass a natural-language instruction for this domain. "
        "Use this instead of calling the domain APIs yourself."
    )

    def _call(query: str) -> str:
        parent_config = var_child_runnable_config.get()
        config = _child_config(parent_config, app_label)
        try:
            result = subagent.invoke(_subagent_payload(query, parent_config), config)
        except GraphBubbleUp:
            raise
        except Exception:
            logger.exception("Domain agent %s failed", app_label)
            _submit_failed_domain_memory(app_label, query, config, subagent)
            return TOOL_FAILURE_MESSAGE
        reraise_nested_interrupt(result)
        return _last_message_text(result)

    async def _acall(query: str) -> str:
        parent_config = var_child_runnable_config.get()
        config = _child_config(parent_config, app_label)
        try:
            result = await subagent.ainvoke(
                _subagent_payload(query, parent_config), config
            )
        except GraphBubbleUp:
            raise
        except Exception:
            logger.exception("Domain agent %s failed", app_label)
            await _asubmit_failed_domain_memory(app_label, query, config, subagent)
            return TOOL_FAILURE_MESSAGE
        reraise_nested_interrupt(result)
        return _last_message_text(result)

    _call.__name__ = tool_name
    _call.__doc__ = description
    _acall.__name__ = tool_name
    _acall.__doc__ = description
    return StructuredTool.from_function(
        func=_call,
        coroutine=_acall,
        name=tool_name,
        description=description,
    )


def _stub_subagent(app_label: str, blurb: str) -> StructuredTool:
    """Record routing without invoking a nested model."""
    tool_name = subagent_tool_name(app_label)
    description = (
        f"{blurb} Pass a natural-language instruction for this domain. "
        "Use this instead of calling the domain APIs yourself."
    )

    def _call(query: str) -> str:
        return f"[stub] {app_label}: {query}"

    async def _acall(query: str) -> str:
        return f"[stub] {app_label}: {query}"

    _call.__name__ = tool_name
    _call.__doc__ = description
    _acall.__name__ = tool_name
    _acall.__doc__ = description
    return StructuredTool.from_function(
        func=_call,
        coroutine=_acall,
        name=tool_name,
        description=description,
    )


def reraise_nested_interrupt(result) -> None:
    """Bubble a child-graph interrupt so parent HITL can see it."""
    interrupts = interrupt_tuple(result)
    if interrupts:
        raise GraphInterrupt(interrupts)


def interrupt_tuple(result) -> tuple | None:
    if not isinstance(result, dict):
        return None
    payload = result.get("__interrupt__")
    if not payload:
        return None
    from langgraph.types import Interrupt

    items = payload if isinstance(payload, (list, tuple)) else (payload,)
    normalized = []
    for item in items:
        if isinstance(item, Interrupt) or hasattr(item, "value"):
            normalized.append(item)
        else:
            normalized.append(Interrupt(value=item))
    return tuple(normalized)


def _child_thread_id(thread_id: str, app_label: str) -> str:
    """Isolate nested checkpoints without breaking Agent Server UUID thread ids.

    LangGraph Agent Server stores ``thread_id`` as UUID. The previous
    ``{thread}::{app}`` suffix is kept only for non-UUID ids used in tests.
    """
    try:
        parent = uuid.UUID(str(thread_id))
    except ValueError:
        return f"{thread_id}::{app_label}"
    return str(uuid.uuid5(parent, app_label))


def _child_config(config, app_label: str):
    if not isinstance(config, dict):
        return config
    configurable = dict(config.get("configurable") or {})
    thread_id = configurable.get("thread_id")
    if thread_id:
        configurable["thread_id"] = _child_thread_id(thread_id, app_label)
        # Parent checkpoint coordinates are invalid on the child thread.
        configurable["checkpoint_ns"] = ""
        configurable.pop("checkpoint_id", None)
        configurable.pop("checkpoint_map", None)
    from ai_agent.memory.namespaces import normalize_memory_config

    return bind_recursion_limit(
        normalize_memory_config(
            {**config, "configurable": configurable}, app_label
        )
    )


def _subagent_payload(query: str, config) -> dict:
    """Forward host frontend tools/context into nested domain agents."""
    payload: dict = {"messages": [{"role": "user", "content": query}]}
    copilotkit = _copilotkit_from_config(config)
    if copilotkit is not None:
        payload["copilotkit"] = copilotkit
    return payload


def _copilotkit_from_config(config):
    if not isinstance(config, dict):
        return None
    for source in (config.get("configurable"), config.get("context"), config):
        if isinstance(source, dict) and "copilotkit" in source:
            return source["copilotkit"]
    return None


def _last_message_text(result) -> str:
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            return message_text(messages[-1])
        interrupt_payload = result.get("__interrupt__")
        if interrupt_payload:
            return str(interrupt_payload)
    return str(result)


def _submit_failed_domain_memory(app_label: str, query: str, config, subagent) -> None:
    """Persist the domain transcript when the nested graph dies before after_agent."""
    if not get_agent_settings().is_memory_background():
        return
    messages = _failed_domain_messages(query, subagent, config)
    try:
        from ai_agent.memory.managers import submit_memory

        submit_memory(app_label, messages, config if isinstance(config, dict) else {})
    except Exception:
        logger.exception("Domain memory submit after %s failure failed", app_label)


async def _asubmit_failed_domain_memory(
    app_label: str, query: str, config, subagent
) -> None:
    if not get_agent_settings().is_memory_background():
        return
    messages = await _afailed_domain_messages(query, subagent, config)
    try:
        from ai_agent.memory.managers import submit_memory

        submit_memory(app_label, messages, config if isinstance(config, dict) else {})
    except Exception:
        logger.exception("Domain memory submit after %s failure failed", app_label)


def _failed_domain_messages(query: str, subagent, config) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    messages = _checkpoint_messages(subagent, config)
    if messages:
        return [*messages, AIMessage(content=TOOL_FAILURE_MESSAGE)]
    return [
        HumanMessage(content=query),
        AIMessage(content=TOOL_FAILURE_MESSAGE),
    ]


async def _afailed_domain_messages(query: str, subagent, config) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    messages = await _acheckpoint_messages(subagent, config)
    if messages:
        return [*messages, AIMessage(content=TOOL_FAILURE_MESSAGE)]
    return [
        HumanMessage(content=query),
        AIMessage(content=TOOL_FAILURE_MESSAGE),
    ]


def _checkpoint_messages(subagent, config) -> list:
    get_state = getattr(subagent, "get_state", None)
    if not callable(get_state) or not isinstance(config, dict):
        return []
    try:
        snapshot = get_state(config)
    except Exception:
        logger.debug("Could not read child checkpoint after domain failure", exc_info=True)
        return []
    return _messages_from_snapshot(snapshot)


async def _acheckpoint_messages(subagent, config) -> list:
    aget_state = getattr(subagent, "aget_state", None)
    if callable(aget_state) and isinstance(config, dict):
        try:
            snapshot = await aget_state(config)
            messages = _messages_from_snapshot(snapshot)
            if messages:
                return messages
        except Exception:
            logger.debug(
                "Could not read child checkpoint after domain failure",
                exc_info=True,
            )
    return _checkpoint_messages(subagent, config)


def _messages_from_snapshot(snapshot) -> list:
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return []
    messages = values.get("messages") or []
    return list(messages) if isinstance(messages, list) else []
