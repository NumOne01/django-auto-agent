"""Max-turn and timeout loop guards for supervisor and domain agents."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent.conf import get_agent_settings
from ai_agent.context import AgentUserContext
from ai_agent.graph import _agent_middleware, build_domain_agent, build_supervisor
from ai_agent.tests.graph_test_utils import (
    ToolCallingFakeModel,
    _immediate_executor,
    last_graph_text,
)
from ai_agent.limits import (
    TIMEOUT_MESSAGE,
    RunTimeoutMiddleware,
    apply_run_limits,
    bind_recursion_limit,
    build_limit_middleware,
    recursion_limit_for_max_turn,
    recursion_limit_for_settings,
)

User = get_user_model()


def _limit_settings(**overrides):
    payload = {
        **settings.AI_AGENT,
        "MEMORY_ENABLED": False,
        "COMPACTION_ENABLED": False,
        "MAX_TURN": 25,
        "TIMEOUT_SECONDS": 120,
    }
    payload.update(overrides)
    return payload


class LoopingToolModel(ToolCallingFakeModel):
    """Always requests the same tool so the agent loop would never stop on its own."""

    tool_name: str = "dummy_item_list"
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": {},
                    "id": f"call_{self.calls}",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class LimitsDisabledTests(TestCase):
    def test_no_limit_middleware_when_both_zero(self):
        agent_settings = _limit_settings(MAX_TURN=0, TIMEOUT_SECONDS=0)
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("dummy")
            kinds = [
                type(item).__name__
                for item in mock_create.call_args.kwargs["middleware"]
            ]
            self.assertNotIn("ModelCallLimitMiddleware", kinds)
            self.assertNotIn("RunTimeoutMiddleware", kinds)
            self.assertEqual(build_limit_middleware(), [])


class LimitsEnabledTests(TestCase):
    def test_supervisor_and_domain_include_limit_middleware(self):
        agent_settings = _limit_settings()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor(stub_subagents=True)
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertIn("ModelCallLimitMiddleware", kinds)
        self.assertIn("RunTimeoutMiddleware", kinds)

        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("dummy")
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertIn("ModelCallLimitMiddleware", kinds)
        self.assertIn("RunTimeoutMiddleware", kinds)

    def test_limits_run_after_transcript_and_before_compaction(self):
        fake = object()
        agent_settings = _limit_settings(
            COMPACTION_ENABLED=True,
            MEMORY_ENABLED=True,
            MEMORY_MODE="background",
            MEMORY_STORE="memory",
        )
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.compaction.build_compaction_middleware", return_value=fake
            ),
        ):
            items = _agent_middleware(layer="supervisor")
        kinds = [type(item).__name__ for item in items]
        self.assertEqual(kinds[2], "SafetyMiddleware")
        self.assertEqual(kinds[3], "CopilotKitMiddleware")
        self.assertEqual(kinds[4], "TranscriptMiddleware")
        self.assertEqual(kinds[5], "ModelCallLimitMiddleware")
        self.assertEqual(kinds[6], "RunTimeoutMiddleware")
        self.assertIs(items[7], fake)
        self.assertEqual(kinds[8], "MemoryRecallMiddleware")
        self.assertEqual(kinds[9], "MemoryWriteMiddleware")


class LimitsConfigTests(TestCase):
    def test_defaults_are_on(self):
        with override_settings(AI_AGENT=_limit_settings()):
            parsed = get_agent_settings()
        self.assertEqual(parsed.max_turn, 25)
        self.assertEqual(parsed.timeout_seconds, 120.0)

    def test_zero_disables(self):
        with override_settings(
            AI_AGENT=_limit_settings(MAX_TURN=0, TIMEOUT_SECONDS=0)
        ):
            parsed = get_agent_settings()
        self.assertIsNone(parsed.max_turn)
        self.assertIsNone(parsed.timeout_seconds)

    def test_invalid_values_fall_back_to_defaults(self):
        with override_settings(
            AI_AGENT=_limit_settings(MAX_TURN="bogus", TIMEOUT_SECONDS="nope")
        ):
            parsed = get_agent_settings()
        self.assertEqual(parsed.max_turn, 25)
        self.assertEqual(parsed.timeout_seconds, 120.0)

    def test_negative_values_fall_back_to_defaults(self):
        with override_settings(
            AI_AGENT=_limit_settings(MAX_TURN=-3, TIMEOUT_SECONDS=-1)
        ):
            parsed = get_agent_settings()
        self.assertEqual(parsed.max_turn, 25)
        self.assertEqual(parsed.timeout_seconds, 120.0)

    def test_custom_positive_values(self):
        with override_settings(
            AI_AGENT=_limit_settings(MAX_TURN=8, TIMEOUT_SECONDS=45.5)
        ):
            parsed = get_agent_settings()
        self.assertEqual(parsed.max_turn, 8)
        self.assertEqual(parsed.timeout_seconds, 45.5)

    def test_build_limit_middleware_uses_run_limit(self):
        with override_settings(
            AI_AGENT=_limit_settings(MAX_TURN=4, TIMEOUT_SECONDS=0)
        ):
            items = build_limit_middleware()
        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], ModelCallLimitMiddleware)
        self.assertEqual(items[0].run_limit, 4)
        self.assertEqual(items[0].exit_behavior, "end")


class ApplyRunLimitsTests(TestCase):
    def test_sets_recursion_limit_from_max_turn(self):
        graph = MagicMock()
        graph.with_config.return_value = "limited"
        with override_settings(AI_AGENT=_limit_settings(MAX_TURN=10)):
            result = apply_run_limits(graph)
        graph.with_config.assert_called_once_with(
            {"recursion_limit": recursion_limit_for_max_turn(10)}
        )
        self.assertEqual(result, "limited")
        self.assertEqual(recursion_limit_for_max_turn(10), 96)

    def test_bind_recursion_limit_overrides_parent_default(self):
        with override_settings(AI_AGENT=_limit_settings(MAX_TURN=10)):
            bound = bind_recursion_limit({"recursion_limit": 25, "tags": ["t"]})
        self.assertEqual(bound["recursion_limit"], recursion_limit_for_max_turn(10))
        self.assertEqual(bound["tags"], ["t"])
        with override_settings(AI_AGENT=_limit_settings(MAX_TURN=0)):
            self.assertEqual(recursion_limit_for_settings(), 9_999)

    def test_noop_when_max_turn_disabled(self):
        graph = object()
        with override_settings(AI_AGENT=_limit_settings(MAX_TURN=0)):
            self.assertIs(apply_run_limits(graph), graph)

    def test_skips_graphs_without_with_config(self):
        graph = SimpleNamespace()
        with override_settings(AI_AGENT=_limit_settings(MAX_TURN=5)):
            self.assertIs(apply_run_limits(graph), graph)


class RunTimeoutMiddlewareTests(TestCase):
    def test_records_start_on_before_agent(self):
        middleware = RunTimeoutMiddleware(30, clock=lambda: 100.0)
        result = middleware.before_agent({}, None)
        self.assertEqual(result, {"run_started_at": 100.0})

    def test_jumps_to_end_when_elapsed(self):
        middleware = RunTimeoutMiddleware(1.0, clock=lambda: 100.0)
        result = middleware.before_model({"run_started_at": 0.0}, None)
        self.assertEqual(result["jump_to"], "end")
        self.assertEqual(result["messages"][0].content, TIMEOUT_MESSAGE)

    def test_continues_when_within_budget(self):
        middleware = RunTimeoutMiddleware(10.0, clock=lambda: 5.0)
        result = middleware.before_model({"run_started_at": 0.0}, None)
        self.assertIsNone(result)

    def test_initializes_start_on_first_before_model(self):
        middleware = RunTimeoutMiddleware(10.0, clock=lambda: 42.0)
        result = middleware.before_model({}, None)
        self.assertEqual(result, {"run_started_at": 42.0})


class MaxTurnLoopTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="09001115555", password="pass12345"
        )

    def test_looping_model_stops_at_max_turn(self):
        model = LoopingToolModel(
            responses=[AIMessage(content="unused")],
            tool_name="dummy_item_list",
        )
        agent_settings = _limit_settings(MAX_TURN=2, TIMEOUT_SECONDS=0)
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.tools.invoke_endpoint", return_value="HTTP 200: {}"
            ),
            AgentUserContext(self.user),
            patch(
                "langgraph.prebuilt.tool_node.get_executor_for_config",
                _immediate_executor,
            ),
        ):
            graph = build_domain_agent(
                "dummy",
                model=model,
                checkpointer=InMemorySaver(),
            )
            result = graph.invoke(
                {"messages": [HumanMessage(content="keep going")]},
                {
                    "configurable": {
                        "thread_id": "limit-loop",
                        "user_id": self.user.pk,
                    }
                },
            )
        self.assertEqual(model.calls, 2)
        self.assertIn("limit", last_graph_text(result).lower())
