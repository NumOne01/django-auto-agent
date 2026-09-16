"""Discovery, executor, and mutation-interrupt tests for the AI agent bridge."""

import json
from dataclasses import replace
from decimal import Decimal
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.test import TestCase, override_settings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from rest_framework import serializers

from dummy.models import Item

from ai_agent.context import AgentUserContext
from ai_agent.discovery import discover_endpoints
from ai_agent.executor import _serialize_response, invoke_endpoint
from ai_agent.expose import agent_expose, get_view_attr
from ai_agent.schema import build_args_model, _resolve_ref
from ai_agent.tools import _execute_endpoint_tool, _resume_is_approved

User = get_user_model()


def _endpoint(url_name, method="GET"):
    method = method.upper()
    for item in discover_endpoints():
        if item.url_name == url_name and item.method == method:
            return item
    raise AssertionError(f"Endpoint {method} {url_name} was not discovered")


class DiscoveryTests(TestCase):
    def test_dummy_app_exposed_except_excludes_and_hard_skips(self):
        endpoints = discover_endpoints()
        names = {(item.url_name, item.method) for item in endpoints}

        self.assertIn(("dummy_item_list", "GET"), names)
        self.assertIn(("dummy_item_create", "POST"), names)
        self.assertNotIn(("dummy_hidden", "GET"), names)
        self.assertNotIn(("dummy_webhook", "POST"), names)
        self.assertNotIn(("dummy_internal", "GET"), names)
        self.assertNotIn(("dummy_upload", "POST"), names)
        self.assertNotIn(("dummy_receipt", "GET"), names)

    def test_internal_and_webhook_paths_are_skipped(self):
        names = {item.url_name for item in discover_endpoints()}
        self.assertNotIn("dummy_internal", names)
        self.assertNotIn("dummy_webhook", names)

    def test_agent_memory_routes_are_excluded(self):
        names = {item.url_name for item in discover_endpoints()}
        self.assertNotIn("agent_memories", names)
        self.assertNotIn("agent_memory_detail", names)

    def test_function_level_expose_while_app_is_off(self):
        names = {item.url_name for item in discover_endpoints()}
        self.assertIn("public_note", names)
        self.assertNotIn("private_note", names)

    def test_mutating_endpoints_require_confirmation(self):
        create = _endpoint("dummy_item_create", "POST")
        listing = _endpoint("dummy_item_list", "GET")
        self.assertTrue(create.confirm)
        self.assertFalse(listing.confirm)

    def test_exposed_views_have_no_response_serializer_by_default(self):
        self.assertIsNone(_endpoint("dummy_item_list").response_serializer)

    def test_expose_stores_response_serializer(self):
        class TinySerializer(serializers.Serializer):
            id = serializers.CharField()

        @agent_expose(response_serializer=TinySerializer)
        def decorated(request):
            pass

        @agent_expose
        def bare(request):
            pass

        self.assertIs(
            get_view_attr(decorated, "_agent_response_serializer"), TinySerializer
        )
        self.assertIsNone(get_view_attr(bare, "_agent_response_serializer"))


class SchemaMappingTests(TestCase):
    def test_allof_enum_ref_stays_a_string(self):
        schema = {
            "components": {
                "schemas": {
                    "SideEnum": {"type": "string", "enum": ["BUY", "SELL"]},
                }
            }
        }
        spec = {
            "allOf": [{"$ref": "#/components/schemas/SideEnum"}],
            "description": "Order side: BUY or SELL",
        }
        resolved = _resolve_ref(schema, spec)
        self.assertEqual(resolved.get("type"), "string")
        self.assertEqual(resolved.get("enum"), ["BUY", "SELL"])
        self.assertNotEqual(resolved.get("type"), "object")

    def test_create_item_choice_fields_accept_enum_strings(self):
        endpoint = _endpoint("dummy_item_create", "POST")
        args_model = build_args_model(endpoint)
        parsed = args_model(name="Widget", status="OPEN", quantity=1)
        self.assertEqual(parsed.name, "Widget")
        self.assertEqual(parsed.status, "OPEN")
        self.assertEqual(parsed.quantity, "1")
        parsed_float = args_model(name="Widget", status="CLOSED", quantity=1.5)
        self.assertEqual(parsed_float.quantity, "1.5")

    def test_multipart_json_body_fields_are_mapped(self):
        endpoint = _endpoint("dummy_create_ticket", "POST")
        self.assertIn("category", endpoint.args_properties)
        self.assertIn("message", endpoint.args_properties)
        self.assertNotIn("attachments", endpoint.args_properties)
        args_model = build_args_model(endpoint)
        parsed = args_model(category=1, message="help")
        self.assertEqual(parsed.category, 1)
        self.assertEqual(parsed.message, "help")

    def test_item_detail_maps_pk_and_body_fields(self):
        endpoint = _endpoint("dummy_item_detail", "PATCH")
        self.assertIn("pk", endpoint.args_properties)
        self.assertNotIn("id", endpoint.args_properties)
        self.assertIn("allocations", endpoint.args_properties)
        self.assertIn("name", endpoint.args_properties)


class ExecutorTests(TestCase):
    def setUp(self):
        self.user_a = User.objects.create_user(
            username="user_a", password="pass12345"
        )
        self.user_b = User.objects.create_user(
            username="user_b", password="pass12345"
        )
        Item.objects.create(
            owner=self.user_a, name="Alpha", secret="12.5", quantity=Decimal("12.5")
        )
        Item.objects.create(
            owner=self.user_b, name="Beta", secret="99", quantity=Decimal("99")
        )
        self.endpoint = _endpoint("dummy_item_list", "GET")

    def test_unauthenticated_invoke_is_rejected(self):
        result = invoke_endpoint(self.endpoint, AnonymousUser(), {})
        self.assertTrue(result.startswith("HTTP 401") or result.startswith("HTTP 403"), result)

    def test_authenticated_user_only_sees_own_items(self):
        result_a = invoke_endpoint(self.endpoint, self.user_a, {})
        result_b = invoke_endpoint(self.endpoint, self.user_b, {})
        self.assertIn("12.5", result_a)
        self.assertNotIn("99", result_a)
        self.assertIn("99", result_b)
        self.assertNotIn("12.5", result_b)

    def test_response_serializer_subsets_item_fields(self):
        class NameOnlySerializer(serializers.Serializer):
            name = serializers.CharField()

        full = invoke_endpoint(self.endpoint, self.user_a, {})
        self.assertIn("12.5", full)
        self.assertIn("secret", full)

        shaped = invoke_endpoint(
            replace(self.endpoint, response_serializer=NameOnlySerializer),
            self.user_a,
            {},
        )
        self.assertTrue(shaped.startswith("HTTP 200: "), shaped)
        payload = json.loads(shaped[len("HTTP 200: ") :])
        self.assertEqual(payload, [{"name": "Alpha"}])
        self.assertNotIn("secret", shaped)

    def test_response_serializer_is_skipped_on_error_status(self):
        class NameOnlySerializer(serializers.Serializer):
            name = serializers.CharField()

        result = invoke_endpoint(
            replace(self.endpoint, response_serializer=NameOnlySerializer),
            AnonymousUser(),
            {},
        )
        self.assertTrue(
            result.startswith("HTTP 401") or result.startswith("HTTP 403"), result
        )
        self.assertIn("detail", result)

    def test_truncates_tool_response_with_explicit_marker(self):
        agent_settings = {**settings.AI_AGENT, "MAX_TOOL_RESPONSE_CHARS": 40}
        with override_settings(AI_AGENT=agent_settings):
            result = invoke_endpoint(self.endpoint, self.user_a, {})
        self.assertTrue(result.startswith("HTTP 200: "), result)
        self.assertIn("truncated, showing 40 of", result)
        self.assertIn("do not invent omitted fields", result)
        shown = result[len("HTTP 200: ") :]
        marker = shown.index("…[truncated")
        self.assertEqual(marker, 40)

    def test_callback_exception_returns_http_500_and_logs(self):
        with patch.object(
            self.endpoint, "callback", side_effect=RuntimeError("boom")
        ):
            with self.assertLogs("ai_agent.executor", level="ERROR") as captured:
                result = invoke_endpoint(self.endpoint, self.user_a, {})
        self.assertTrue(result.startswith("HTTP 500: RuntimeError: boom"), result)
        self.assertTrue(
            any("raised during invoke" in line for line in captured.output)
        )


class ResponseSerializerDumpTests(TestCase):
    def test_list_payload_uses_many(self):
        class ItemOutSerializer(serializers.Serializer):
            id = serializers.CharField()

        response = MagicMock(status_code=200)
        response.data = [
            {"id": "1", "secret": "x"},
            {"id": "2", "secret": "y"},
        ]
        endpoint = MagicMock(response_serializer=ItemOutSerializer)
        payload = json.loads(_serialize_response(response, endpoint))
        self.assertEqual(payload, [{"id": "1"}, {"id": "2"}])

    def test_serializer_exception_falls_back_to_full_dump(self):
        class BrokenSerializer(serializers.Serializer):
            id = serializers.CharField()

            def to_representation(self, instance):
                raise RuntimeError("boom")

        response = MagicMock(status_code=200)
        response.data = {"id": "1", "secret": "x"}
        endpoint = MagicMock(response_serializer=BrokenSerializer)
        with self.assertLogs("ai_agent.executor", level="WARNING") as captured:
            payload = json.loads(_serialize_response(response, endpoint))
        self.assertEqual(payload, {"id": "1", "secret": "x"})
        self.assertTrue(
            any("response_serializer" in line for line in captured.output)
        )


class ResumeApprovalTests(TestCase):
    def test_explicit_approvals(self):
        for value in (
            True,
            "true",
            "True",
            1,
            "1",
            {"approved": True},
            {"approved": "true"},
            {"interruptId": "abc", "status": "resolved", "payload": {"approved": True}},
            [{"interruptId": "abc", "status": "resolved", "payload": {"approved": True}}],
        ):
            with self.subTest(value=value):
                self.assertTrue(_resume_is_approved(value))

    def test_rejects_and_unknown_shapes_fail_closed(self):
        for value in (
            False,
            "false",
            0,
            "0",
            None,
            "",
            {},
            {"approved": False},
            {"approved": "false"},
            {"__agui_cancelled__": True, "interrupt_id": "abc"},
            {"interruptId": "abc", "status": "cancelled"},
            {"interruptId": "abc", "status": "resolved", "payload": {"approved": False}},
            [{"interruptId": "abc", "status": "resolved", "payload": {"approved": False}}],
            {"action": "dummy_item_create", "args": {"name": "Widget"}},
        ):
            with self.subTest(value=value):
                self.assertFalse(_resume_is_approved(value))


class MutationInterruptTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="mutator", password="pass12345"
        )
        self.endpoint = _endpoint("dummy_item_create", "POST")

    def _run_tool(self, resume):
        with (
            patch("ai_agent.tools.interrupt", return_value=resume) as mock_interrupt,
            patch("ai_agent.tools.invoke_endpoint", return_value="HTTP 201: ok") as mock_invoke,
        ):
            with AgentUserContext(self.user):
                result = _execute_endpoint_tool(self.endpoint, {"name": "Widget"})
        return result, mock_interrupt, mock_invoke

    def test_create_does_not_hit_executor_until_resume(self):
        result, mock_interrupt, mock_invoke = self._run_tool(False)
        mock_interrupt.assert_called_once()
        mock_invoke.assert_not_called()
        self.assertIn("declined", result.lower())

        result, _, mock_invoke = self._run_tool(True)
        mock_invoke.assert_called_once()
        self.assertEqual(result, "HTTP 201: ok")

    def test_agui_reject_payload_does_not_mutate(self):
        result, mock_interrupt, mock_invoke = self._run_tool({"approved": False})
        mock_interrupt.assert_called_once()
        mock_invoke.assert_not_called()
        self.assertIn("declined", result.lower())

    def test_agui_approve_payload_mutates(self):
        result, _, mock_invoke = self._run_tool({"approved": True})
        mock_invoke.assert_called_once()
        self.assertEqual(result, "HTTP 201: ok")

    def test_agui_cancel_sentinel_does_not_mutate(self):
        result, _, mock_invoke = self._run_tool(
            {"__agui_cancelled__": True, "interrupt_id": "abc"}
        )
        mock_invoke.assert_not_called()
        self.assertIn("declined", result.lower())


class SupervisorGraphTests(TestCase):
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

    def test_supervisor_builds_with_fake_model(self):
        from ai_agent.graph import build_supervisor

        graph = build_supervisor(
            model=FakeListChatModel(responses=["ok"]),
            checkpointer=InMemorySaver(),
        )
        self.assertTrue(hasattr(graph, "invoke"))

    def test_supervisor_builds_without_checkpointer_for_studio(self):
        from langgraph.pregel import Pregel

        from ai_agent.graph import build_studio_graph

        graph = build_studio_graph(model=FakeListChatModel(responses=["ok"]))
        self.assertIsInstance(graph, Pregel)

    def test_uses_distinct_supervisor_and_subagent_models(self):
        from ai_agent.graph import build_supervisor

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
            build_supervisor()

        self.assertGreater(mock_create.call_count, 1)
        for call in mock_create.call_args_list[:-1]:
            self.assertEqual(call.kwargs["model"], "openai:subagent-test")
        self.assertEqual(
            mock_create.call_args_list[-1].kwargs["model"],
            "openai:supervisor-test",
        )
