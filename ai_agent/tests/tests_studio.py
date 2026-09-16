"""LangGraph Studio user-binding tests."""

from types import SimpleNamespace

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.conf import settings
from django.test import TestCase, override_settings

from ai_agent.tests.graph_test_utils import offline_agent_settings

from ai_agent.context import (
    AgentUserContext,
    agent_user_from_config,
    get_current_user,
    peek_current_user,
    resolve_agent_user,
)
from ai_agent.graph import AgentUserMiddleware

User = get_user_model()


class ResolveAgentUserTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="studio_user", password="pass12345"
        )

    def test_resolves_langgraph_auth_user_id(self):
        found = resolve_agent_user(
            {"configurable": {"langgraph_auth_user_id": self.user.pk}}
        )
        self.assertEqual(found.pk, self.user.pk)

    def test_resolves_langgraph_auth_user_dict(self):
        found = resolve_agent_user(
            {
                "configurable": {
                    "langgraph_auth_user": {"identity": str(self.user.pk)}
                }
            }
        )
        self.assertEqual(found.pk, self.user.pk)

    def test_auth_user_wins_over_forged_user_id(self):
        other = User.objects.create_user(
            username="studio_other", password="pass12345"
        )
        found = resolve_agent_user(
            {
                "configurable": {
                    "langgraph_auth_user_id": self.user.pk,
                    "user_id": other.pk,
                    "phone_number": other.username,
                }
            }
        )
        self.assertEqual(found.pk, self.user.pk)

    def test_resolves_user_id_from_configurable(self):
        found = resolve_agent_user({"configurable": {"user_id": self.user.pk}})
        self.assertEqual(found.pk, self.user.pk)

    @override_settings(TESTING=False)
    def test_non_testing_ignores_client_supplied_user_id(self):
        self.assertIsNone(
            resolve_agent_user({"configurable": {"user_id": self.user.pk}})
        )

    def test_resolves_username_from_configurable(self):
        found = resolve_agent_user(
            {"configurable": {"username": self.user.username}}
        )
        self.assertEqual(found.pk, self.user.pk)

    def test_resolves_login_from_settings_fallback(self):
        from django.conf import settings

        with override_settings(
            AI_AGENT={**settings.AI_AGENT, "STUDIO_USER_PHONE": self.user.username}
        ):
            found = resolve_agent_user({})
        self.assertEqual(found.pk, self.user.pk)

    def test_missing_user_returns_none(self):
        from django.conf import settings

        with override_settings(
            AI_AGENT={**settings.AI_AGENT, "STUDIO_USER_PHONE": ""}
        ):
            self.assertIsNone(resolve_agent_user({}))

    @override_settings(TESTING=False)
    def test_missing_auth_user_lookup_is_generic_outside_tests(self):
        with self.assertRaises(RuntimeError) as caught:
            resolve_agent_user(
                {"configurable": {"langgraph_auth_user_id": 999999}}
            )
        self.assertEqual(str(caught.exception), "AI agent user was not found.")
        self.assertNotIn("999999", str(caught.exception))

    def test_db_unreachable_message_includes_detail_in_tests(self):
        from ai_agent.context import _db_unreachable_message

        message = _db_unreachable_message(Exception("boom"))
        self.assertIn("boom", message)

    @override_settings(TESTING=False)
    def test_db_unreachable_message_is_generic_outside_tests(self):
        from ai_agent.context import _db_unreachable_message

        message = _db_unreachable_message(
            Exception("could not connect to DB_HOST=db")
        )
        self.assertEqual(message, "AI agent cannot reach the database.")
        self.assertNotIn("DB_HOST", message)

    def test_existing_chat_user_is_not_replaced(self):
        other = User.objects.create_user(
            username="studio_other2", password="pass12345"
        )
        with AgentUserContext(self.user):
            with agent_user_from_config(
                {"configurable": {"user_id": other.pk}}, required=True
            ):
                self.assertEqual(get_current_user().pk, self.user.pk)


class AgentUserMiddlewareTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="studio_mw", password="pass12345"
        )
        self.middleware = AgentUserMiddleware()

    def _request(self, config):
        return SimpleNamespace(runtime=SimpleNamespace(config=config))

    def test_wrap_tool_call_binds_user_from_config(self):
        def handler(_request):
            return get_current_user().username

        result = self.middleware.wrap_tool_call(
            self._request({"configurable": {"user_id": self.user.pk}}),
            handler,
        )
        self.assertEqual(result, self.user.username)
        self.assertIsNone(peek_current_user())

    def test_wrap_tool_call_requires_a_user(self):
        def handler(_request):
            return "should not run"

        with self.assertRaises(RuntimeError):
            self.middleware.wrap_tool_call(self._request({}), handler)

    def test_awrap_tool_call_binds_user_from_config(self):
        async def handler(_request):
            return get_current_user().username

        result = async_to_sync(self.middleware.awrap_tool_call)(
            self._request({"configurable": {"user_id": self.user.pk}}),
            handler,
        )
        self.assertEqual(result, self.user.username)
        self.assertIsNone(peek_current_user())


class CopilotKitWiringTests(TestCase):
    def setUp(self):
        patcher = override_settings(AI_AGENT=offline_agent_settings())
        patcher.enable()
        self.addCleanup(patcher.disable)

    def test_agent_middleware_includes_copilotkit(self):
        from copilotkit import CopilotKitMiddleware

        from ai_agent.graph import _agent_middleware

        kinds = [type(item) for item in _agent_middleware()]
        self.assertIn(CopilotKitMiddleware, kinds)

    def test_studio_graph_builds_with_copilotkit_state(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langgraph.pregel import Pregel

        from ai_agent.graph import build_studio_graph

        graph = build_studio_graph(model=FakeListChatModel(responses=["ok"]))
        self.assertIsInstance(graph, Pregel)
        channels = getattr(graph, "channels", None) or {}
        self.assertIn("copilotkit", channels)

    def test_copilotkit_http_app_exposes_agui_route(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from ai_agent.graph import build_copilotkit_http_app, build_studio_graph

        graph = build_studio_graph(model=FakeListChatModel(responses=["ok"]))
        app = build_copilotkit_http_app(graph)
        paths = [getattr(route, "path", "") for route in app.routes]
        self.assertTrue(
            any("/copilotkit" in path for path in paths),
            paths,
        )

    def test_subagent_payload_forwards_copilotkit_state(self):
        from ai_agent.graph import _subagent_payload

        payload = _subagent_payload(
            "check wallet",
            {"configurable": {"copilotkit": {"actions": [{"name": "ui_tool"}]}}},
        )
        self.assertEqual(payload["messages"][0]["content"], "check wallet")
        self.assertEqual(payload["copilotkit"]["actions"][0]["name"], "ui_tool")


class ProductionStudioGuardTests(TestCase):
    def test_desktop_mode_rejected_when_debug_is_false(self):
        from unittest.mock import patch

        from ai_agent.conf import assert_production_studio_safe

        with (
            override_settings(DEBUG=False, TESTING=True),
            patch.dict("os.environ", {"LANGSMITH_LANGGRAPH_DESKTOP": "true"}),
        ):
            with self.assertRaises(RuntimeError) as caught:
                assert_production_studio_safe()
        self.assertIn("desktop mode", str(caught.exception))

    def test_studio_phone_rejected_outside_debug(self):
        from django.conf import settings

        from ai_agent.conf import assert_production_studio_safe

        with override_settings(
            DEBUG=False,
            TESTING=False,
            AI_AGENT={**settings.AI_AGENT, "STUDIO_USER_PHONE": "09001110000"},
        ):
            with self.assertRaises(RuntimeError) as caught:
                assert_production_studio_safe()
        self.assertIn("STUDIO_USER_PHONE", str(caught.exception))

    def test_allows_production_without_studio_fallbacks(self):
        from unittest.mock import patch

        from django.conf import settings

        from ai_agent.conf import assert_production_studio_safe

        with (
            override_settings(
                DEBUG=False,
                TESTING=False,
                AI_AGENT={**settings.AI_AGENT, "STUDIO_USER_PHONE": ""},
            ),
            patch.dict("os.environ", {"LANGSMITH_LANGGRAPH_DESKTOP": ""}, clear=False),
        ):
            assert_production_studio_safe()
