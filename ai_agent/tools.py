"""Build LangChain StructuredTools from discovered DRF endpoints."""

from __future__ import annotations

from collections import defaultdict

from asgiref.sync import sync_to_async
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt

from ai_agent.context import get_current_user, peek_current_user, resolve_agent_user
from ai_agent.discovery import DiscoveredEndpoint, discover_endpoints
from ai_agent.executor import invoke_endpoint
from ai_agent.schema import build_args_model

_invoke_endpoint_async = sync_to_async(invoke_endpoint, thread_sensitive=True)

_AGUI_CANCELLED = "__agui_cancelled__"
_APPROVED_VALUES = (True, "true", "True", 1, "1")


def build_tools(endpoints: list[DiscoveredEndpoint] | None = None) -> list[StructuredTool]:
    if endpoints is None:
        endpoints = discover_endpoints()
    return [build_tool(endpoint) for endpoint in endpoints]


def tools_by_app(endpoints: list[DiscoveredEndpoint] | None = None) -> dict[str, list[StructuredTool]]:
    if endpoints is None:
        endpoints = discover_endpoints()
    grouped: dict[str, list[StructuredTool]] = defaultdict(list)
    for endpoint in endpoints:
        grouped[endpoint.app_label].append(build_tool(endpoint))
    return dict(grouped)


def build_tool(endpoint: DiscoveredEndpoint) -> StructuredTool:
    args_model = build_args_model(endpoint)
    description = _tool_description(endpoint)

    def _run(**kwargs):
        return _execute_endpoint_tool(endpoint, kwargs)

    async def _arun(**kwargs):
        return await _aexecute_endpoint_tool(endpoint, kwargs)

    _run.__name__ = endpoint.operation_id
    _run.__doc__ = description
    _arun.__name__ = endpoint.operation_id
    _arun.__doc__ = description
    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=endpoint.operation_id,
        description=description,
        args_schema=args_model,
    )


def _resume_is_approved(value) -> bool:
    """True only for an explicit HITL approval.

    CopilotKit ``useInterrupt`` resumes with ``{approved: bool}`` (or a
    cancelled sentinel), not a bare boolean. Unknown shapes fail closed so a
    mutation cannot proceed from a reject payload.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        value = value[0]
    if isinstance(value, dict):
        if value.get(_AGUI_CANCELLED) or value.get("status") == "cancelled":
            return False
        if "approved" in value:
            value = value["approved"]
        elif "payload" in value:
            value = value["payload"]
        else:
            return False
        return _resume_is_approved(value)
    return value in _APPROVED_VALUES


def _confirm_or_none(endpoint: DiscoveredEndpoint, arguments: dict):
    if not endpoint.confirm:
        return None
    approved = interrupt(
        {
            "action": endpoint.operation_id,
            "method": endpoint.method,
            "path": endpoint.path_template,
            "args": arguments,
            "summary": endpoint.summary or endpoint.operation_id,
        }
    )
    if not _resume_is_approved(approved):
        return "The user declined this action."
    return None


def _execute_endpoint_tool(endpoint: DiscoveredEndpoint, arguments: dict) -> str:
    declined = _confirm_or_none(endpoint, arguments)
    if declined is not None:
        return declined
    return invoke_endpoint(endpoint, get_current_user(), arguments)


async def _aexecute_endpoint_tool(endpoint: DiscoveredEndpoint, arguments: dict) -> str:
    declined = _confirm_or_none(endpoint, arguments)
    if declined is not None:
        return declined
    user = peek_current_user()
    if user is None:
        user = await sync_to_async(resolve_agent_user, thread_sensitive=True)(
            var_child_runnable_config.get()
        )
    if user is None:
        raise RuntimeError("No authenticated user in AI agent context")
    return await _invoke_endpoint_async(endpoint, user, arguments)


def _tool_description(endpoint: DiscoveredEndpoint) -> str:
    parts = [
        endpoint.summary or endpoint.operation_id,
        f"{endpoint.method} {endpoint.path_template}",
    ]
    if endpoint.description and endpoint.description != endpoint.summary:
        parts.append(endpoint.description)
    return " — ".join(part for part in parts if part)
