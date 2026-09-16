"""Shared helpers for per-endpoint AI-agent graph tests.

Not a test module. Import from ``ai_agent.tests.graph_test_utils``.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

from django.contrib.auth import get_user_model
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from ai_agent.context import AgentUserContext
from ai_agent.discovery import DiscoveredEndpoint, discover_endpoints
from ai_agent.evals.runtime import _immediate_executor, make_test_runtime
from ai_agent.graph import message_text
from ai_agent.schema import build_args_model
from ai_agent.tools import build_tool, build_tools

User = get_user_model()

__all__ = (
    "AgentEndpointAssertions",
    "ToolCallingFakeModel",
    "_immediate_executor",
    "all_graph_text",
    "find_endpoint",
    "http_status",
    "invoke_graph_forced_tool",
    "invoke_tool",
    "last_graph_text",
    "make_agent_user",
    "make_test_runtime",
    "offline_agent_settings",
    "parse_tool_args",
    "registered_tool_names",
)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel that accepts ``bind_tools`` from ``create_agent``."""

    def bind_tools(self, tools, **kwargs):
        return self


def offline_agent_settings(**overrides) -> dict:
    """AI_AGENT dict that will not call live models while compiling graphs."""
    from django.conf import settings as django_settings

    payload = {
        **django_settings.AI_AGENT,
        "COMPACTION_ENABLED": False,
        "MEMORY_ENABLED": False,
    }
    payload.update(overrides)
    return payload


def find_endpoint(*, operation_id=None, url_name=None, method=None) -> DiscoveredEndpoint:
    method = method.upper() if method else None
    for item in discover_endpoints():
        if operation_id and item.operation_id != operation_id:
            continue
        if url_name and item.url_name != url_name:
            continue
        if method and item.method != method:
            continue
        return item
    raise AssertionError(
        f"Endpoint not discovered: operation_id={operation_id!r} "
        f"url_name={url_name!r} method={method!r}"
    )


def make_agent_user(username: str, password: str = "pass12345"):
    return User.objects.create_user(username=username, password=password)


def invoke_graph_forced_tool(
    user,
    tool_name: str,
    tool_args: dict | None = None,
    *,
    thread_id: str = "graph-test",
    final_text: str = "done",
    approve_mutation: bool = True,
    prompt: str = "please run the tool",
):
    """Invoke the domain agent with a fake model that calls ``tool_name``."""
    from django.test.utils import override_settings

    tool_args = tool_args or {}
    endpoint = None
    for item in discover_endpoints():
        if item.operation_id == tool_name:
            endpoint = item
            break
    if endpoint is None:
        raise AssertionError(f"Unknown agent tool {tool_name!r}")
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content=final_text),
        ]
    )
    interrupt_ctx = patch(
        "ai_agent.tools.interrupt", return_value=approve_mutation
    )
    with override_settings(AI_AGENT=offline_agent_settings()):
        agent_runtime = make_test_runtime()
        graph = agent_runtime.domain_agent(
            endpoint.app_label,
            model=model,
        )
        try:
            with (
                interrupt_ctx,
                AgentUserContext(user),
                patch(
                    "langgraph.prebuilt.tool_node.get_executor_for_config",
                    _immediate_executor,
                ),
            ):
                return graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    {"configurable": {"thread_id": thread_id, "user_id": user.pk}},
                )
        finally:
            agent_runtime.close()


def registered_tool_names() -> set[str]:
    return {tool.name for tool in build_tools()}


def parse_tool_args(endpoint: DiscoveredEndpoint, arguments: dict):
    return build_args_model(endpoint)(**arguments)


def invoke_tool(
    endpoint: DiscoveredEndpoint,
    user,
    arguments: dict | None = None,
    *,
    approve_mutation: bool = True,
):
    """Run the same StructuredTool the graph binds, as ``user``."""
    tool = build_tool(endpoint)
    arguments = arguments or {}
    interrupt_ctx = (
        patch("ai_agent.tools.interrupt", return_value=approve_mutation)
        if endpoint.confirm
        else nullcontext()
    )
    with AgentUserContext(user), interrupt_ctx:
        return tool.invoke(arguments)


def http_status(payload: str) -> int | None:
    if not isinstance(payload, str) or not payload.startswith("HTTP "):
        return None
    token = payload.split(":", 1)[0].removeprefix("HTTP ").strip()
    try:
        return int(token)
    except ValueError:
        return None


def last_graph_text(result) -> str:
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            return message_text(messages[-1])
        if result.get("__interrupt__"):
            return str(result["__interrupt__"])
    return str(result)


def all_graph_text(result) -> str:
    """Concatenate every graph message so tool output is visible, not just the final AI text."""
    if isinstance(result, dict):
        messages = result.get("messages") or []
        parts = [message_text(item) for item in messages]
        if result.get("__interrupt__"):
            parts.append(str(result["__interrupt__"]))
        return "\n".join(part for part in parts if part)
    return str(result)


class AgentEndpointAssertions:
    """Mixin of per-endpoint checks. Pair with django.test.TestCase."""

    def assert_tool_registered(self, endpoint: DiscoveredEndpoint):
        names = registered_tool_names()
        self.assertIn(
            endpoint.operation_id,
            names,
            f"{endpoint.operation_id} missing from graph tools: {sorted(names)}",
        )

    def assert_args_model_builds(self, endpoint: DiscoveredEndpoint):
        model = build_args_model(endpoint)
        self.assertTrue(hasattr(model, "model_fields"))

    def assert_missing_required_rejected(self, endpoint: DiscoveredEndpoint, user):
        required = list(endpoint.args_required or [])
        if not required:
            return
        with self.assertRaises(ValidationError):
            invoke_tool(endpoint, user, {})

    def assert_invalid_values_rejected_or_http_error(
        self, endpoint: DiscoveredEndpoint, user, bad_arguments: dict
    ):
        try:
            result = invoke_tool(endpoint, user, bad_arguments)
        except (ValidationError, ValueError, TypeError):
            return
        status = http_status(result)
        self.assertIsNotNone(status, result)
        self.assertGreaterEqual(status, 400, result)

    def assert_happy_tool(
        self,
        endpoint: DiscoveredEndpoint,
        user,
        arguments: dict | None = None,
        *,
        allowed_status: tuple[int, ...] = (200, 201, 202, 204),
        contains: str | None = None,
    ) -> str:
        result = invoke_tool(endpoint, user, arguments or {})
        status = http_status(result)
        self.assertIsNotNone(status, result)
        self.assertIn(status, allowed_status, result)
        if contains:
            self.assertIn(contains, result)
        return result

    def assert_graph_forced_tool(
        self,
        endpoint: DiscoveredEndpoint,
        user,
        arguments: dict | None = None,
        *,
        contains: str | None = None,
    ) -> str:
        result = invoke_graph_forced_tool(
            user,
            endpoint.operation_id,
            arguments or {},
            thread_id=f"{endpoint.operation_id}-{user.pk}",
        )
        text = all_graph_text(result)
        self.assertTrue(text, result)
        if contains:
            self.assertIn(contains, text)
        return text

    def assert_mutation_requires_approval(
        self, endpoint: DiscoveredEndpoint, user, arguments: dict
    ):
        if not endpoint.confirm:
            return
        declined = invoke_tool(
            endpoint, user, arguments, approve_mutation=False
        )
        self.assertIn("declined", str(declined).lower())
