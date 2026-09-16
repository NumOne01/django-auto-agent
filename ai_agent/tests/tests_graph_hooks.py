"""Unit tests for eval/graph hooks: domain agents, stubs, nested interrupts."""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from asgiref.sync import async_to_sync
from django.conf import settings
from django.test import TestCase, override_settings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.types import Interrupt

from ai_agent.conf import get_agent_settings
from ai_agent.graph import (
    TOOL_FAILURE_MESSAGE,
    _agent_middleware,
    _child_config,
    _child_thread_id,
    _on_tool_error,
    _wrap_subagent,
    build_domain_agent,
    build_supervisor,
    reraise_nested_interrupt,
    subagent_tool_name,
)
from ai_agent.limits import recursion_limit_for_max_turn


class GraphHookTests(TestCase):
    def setUp(self):
        patcher = override_settings(
            AI_AGENT={
                **settings.AI_AGENT,
                "COMPACTION_ENABLED": False,
                "MEMORY_ENABLED": False,
            }
        )
        patcher.enable()
        self.addCleanup(patcher.disable)

    def test_subagent_tool_name(self):
        self.assertEqual(subagent_tool_name("wallet"), "call_wallet_agent")

    def test_build_domain_agent_uses_subagent_model(self):
        agent_settings = {
            **settings.AI_AGENT,
            "SUBAGENT_MODEL": "openai:subagent-test",
        }
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("dummy")

        mock_create.assert_called_once()
        self.assertEqual(mock_create.call_args.kwargs["model"], "openai:subagent-test")
        self.assertTrue(mock_create.call_args.kwargs["tools"])

    def test_build_domain_agent_unknown_app_raises(self):
        with self.assertRaises(LookupError):
            build_domain_agent("not_an_exposed_app")

    def test_stub_subagents_skips_nested_create_agent(self):
        agent_settings = {
            **settings.AI_AGENT,
            "SUPERVISOR_MODEL": "openai:supervisor-test",
            "SUBAGENT_MODEL": "openai:subagent-test",
        }
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor(stub_subagents=True)

        self.assertEqual(mock_create.call_count, 1)
        self.assertEqual(
            mock_create.call_args.kwargs["model"], "openai:supervisor-test"
        )
        tool_names = [tool.name for tool in mock_create.call_args.kwargs["tools"]]
        self.assertIn("call_dummy_agent", tool_names)

    def test_extra_middleware_is_passed_to_create_agent(self):
        extra = object()
        with patch(
            "ai_agent.graph.create_agent", return_value=MagicMock()
        ) as mock_create:
            build_supervisor(
                stub_subagents=True,
                extra_middleware=[extra],
            )
        middleware = mock_create.call_args.kwargs["middleware"]
        self.assertIs(middleware[-1], extra)

    def test_extra_middleware_is_appended(self):
        extra = MagicMock()
        extra.__class__ = type("EvalRecorder", (), {})
        kinds = [type(item) for item in _agent_middleware([extra])]
        self.assertIs(kinds[-1], type(extra))

    def test_reraise_nested_interrupt_from_result_dict(self):
        payload = Interrupt(value={"action": "withdraw", "args": {"amount": "1"}})
        with self.assertRaises(GraphInterrupt) as ctx:
            reraise_nested_interrupt({"__interrupt__": (payload,)})
        self.assertEqual(ctx.exception.args[0][0].value["action"], "withdraw")

    def test_reraise_nested_interrupt_noop_without_payload(self):
        reraise_nested_interrupt({"messages": []})
        reraise_nested_interrupt("ok")

    def test_wrap_subagent_reraises_interrupt_result(self):
        child = MagicMock()
        child.invoke.return_value = {
            "__interrupt__": (
                Interrupt(value={"action": "place_order", "args": {"side": "BUY"}}),
            )
        }
        tool = _wrap_subagent("trading", "trading blurb", child)
        token = var_child_runnable_config.set({"configurable": {"thread_id": "t1"}})
        try:
            with self.assertRaises(GraphInterrupt) as ctx:
                tool.invoke({"query": "buy silver"})
        finally:
            var_child_runnable_config.reset(token)
        self.assertEqual(ctx.exception.args[0][0].value["action"], "place_order")
        child.invoke.assert_called_once()
        child_config = child.invoke.call_args.args[1]
        self.assertEqual(child_config["configurable"]["thread_id"], "t1::trading")

    def test_child_thread_id_keeps_suffix_for_non_uuid(self):
        self.assertEqual(_child_thread_id("t1", "trading"), "t1::trading")

    def test_child_thread_id_uuid_stays_uuid(self):
        parent = "5943ed40-e036-4035-83dd-5048e58c4295"
        child = _child_thread_id(parent, "wallet")
        self.assertEqual(str(uuid.UUID(child)), child)
        self.assertNotIn("::", child)
        self.assertEqual(child, _child_thread_id(parent, "wallet"))
        self.assertNotEqual(child, _child_thread_id(parent, "trading"))

    def test_child_config_drops_parent_checkpoint_coordinates(self):
        parent = "5943ed40-e036-4035-83dd-5048e58c4295"
        config = _child_config(
            {
                "configurable": {
                    "thread_id": parent,
                    "checkpoint_ns": "tools:abc",
                    "checkpoint_id": "11111111-1111-1111-1111-111111111111",
                    "checkpoint_map": {"": "ignored"},
                    "user_id": 7,
                }
            },
            "wallet",
        )
        configurable = config["configurable"]
        self.assertEqual(configurable["thread_id"], _child_thread_id(parent, "wallet"))
        self.assertEqual(configurable["checkpoint_ns"], "")
        self.assertNotIn("checkpoint_id", configurable)
        self.assertNotIn("checkpoint_map", configurable)
        self.assertEqual(configurable["user_id"], "7")
        self.assertEqual(configurable["memory_layer"], "wallet")
        self.assertEqual(
            config["recursion_limit"],
            recursion_limit_for_max_turn(get_agent_settings().max_turn),
        )

    def test_child_config_overrides_parent_recursion_limit(self):
        config = _child_config(
            {
                "recursion_limit": 25,
                "configurable": {"thread_id": "t-delivery"},
            },
            "delivery",
        )
        self.assertGreater(config["recursion_limit"], 25)
        self.assertEqual(
            config["recursion_limit"],
            recursion_limit_for_max_turn(get_agent_settings().max_turn),
        )

    def test_on_tool_error_hides_internal_details(self):
        request = SimpleNamespace(tool_call={"name": "call_delivery_agent"})
        exc = GraphRecursionError(
            "Recursion limit of 25 reached without hitting a stop condition. "
            "You can increase the limit by setting the `recursion_limit` config key.\n"
            "For troubleshooting, visit: https://docs.langchain.com/oss/python/"
            "langgraph/errors/GRAPH_RECURSION_LIMIT"
        )
        with self.assertLogs("ai_agent.graph", level="ERROR"):
            text = _on_tool_error(exc, request)
        self.assertEqual(text, TOOL_FAILURE_MESSAGE)
        self.assertNotIn("call_delivery_agent", text)
        self.assertNotIn("recursion", text.lower())
        self.assertNotIn("langchain", text.lower())

    def test_wrap_subagent_hides_recursion_error(self):
        child = MagicMock()
        child.invoke.side_effect = GraphRecursionError(
            "Recursion limit of 25 reached without hitting a stop condition."
        )
        tool = _wrap_subagent("delivery", "delivery blurb", child)
        token = var_child_runnable_config.set({"configurable": {"thread_id": "t4"}})
        try:
            with self.assertLogs("ai_agent.graph", level="ERROR"):
                result = tool.invoke({"query": "where is my shipment"})
        finally:
            var_child_runnable_config.reset(token)
        self.assertEqual(result, TOOL_FAILURE_MESSAGE)
        self.assertNotIn("recursion", result.lower())
        self.assertNotIn("call_delivery_agent", result)

    def test_wrap_subagent_submits_domain_memory_after_recursion_error(self):
        child = MagicMock()
        child.invoke.side_effect = GraphRecursionError(
            "Recursion limit of 25 reached without hitting a stop condition."
        )
        child.get_state.side_effect = RuntimeError("no checkpoint")
        tool = _wrap_subagent("delivery", "delivery blurb", child)
        token = var_child_runnable_config.set(
            {"configurable": {"thread_id": "t-mem", "user_id": "22"}}
        )
        agent_settings = {
            **settings.AI_AGENT,
            "MEMORY_ENABLED": True,
            "MEMORY_MODE": "background",
            "MEMORY_STORE": "memory",
            "COMPACTION_ENABLED": False,
        }
        try:
            with (
                override_settings(AI_AGENT=agent_settings),
                patch("ai_agent.memory.managers.submit_memory") as submit,
                self.assertLogs("ai_agent.graph", level="ERROR"),
            ):
                result = tool.invoke({"query": "register silver delivery in Tehran"})
        finally:
            var_child_runnable_config.reset(token)
        self.assertEqual(result, TOOL_FAILURE_MESSAGE)
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[0], "delivery")
        messages = submit.call_args.args[1]
        self.assertEqual(messages[0].content, "register silver delivery in Tehran")
        self.assertEqual(messages[-1].content, TOOL_FAILURE_MESSAGE)

    def test_wrap_subagent_async_hides_recursion_error(self):
        async def _ainvoke(_payload, _config=None):
            raise GraphRecursionError("Recursion limit of 25 reached.")

        child = MagicMock()
        child.ainvoke = _ainvoke
        tool = _wrap_subagent("delivery", "delivery blurb", child)
        token = var_child_runnable_config.set({"configurable": {"thread_id": "t5"}})
        try:
            with self.assertLogs("ai_agent.graph", level="ERROR"):
                result = async_to_sync(tool.ainvoke)({"query": "track package"})
        finally:
            var_child_runnable_config.reset(token)
        self.assertEqual(result, TOOL_FAILURE_MESSAGE)

    def test_wrap_subagent_reraises_graph_bubble_up(self):
        child = MagicMock()
        child.invoke.side_effect = GraphInterrupt(
            (Interrupt(value={"action": "withdraw"}),)
        )
        tool = _wrap_subagent("wallet", "wallet blurb", child)
        token = var_child_runnable_config.set({"configurable": {"thread_id": "t2"}})
        try:
            with self.assertRaises(GraphInterrupt):
                tool.invoke({"query": "withdraw"})
        finally:
            var_child_runnable_config.reset(token)

    def test_wrap_subagent_async_reraises_interrupt_result(self):
        async def _ainvoke(_payload, _config=None):
            return {
                "__interrupt__": (Interrupt(value={"action": "create_ticket"}),)
            }

        child = MagicMock()
        child.ainvoke = _ainvoke
        tool = _wrap_subagent("support", "support blurb", child)
        token = var_child_runnable_config.set({"configurable": {"thread_id": "t3"}})
        try:
            with self.assertRaises(GraphInterrupt):
                async_to_sync(tool.ainvoke)({"query": "open a ticket"})
        finally:
            var_child_runnable_config.reset(token)

    def test_domain_agent_builds_with_fake_model(self):
        graph = build_domain_agent(
            "dummy",
            model=FakeListChatModel(responses=["ok"]),
            checkpointer=InMemorySaver(),
        )
        self.assertTrue(hasattr(graph, "invoke"))
