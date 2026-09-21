"""LangGraph JWT auth handlers and AG-UI Bearer gate."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from fastapi import HTTPException

from ai_agent.auth import (
    HEALTHCHECK_IDENTITY,
    _FALLBACK_DISPLAY_NAME,
    _display_name,
    authenticate,
    on_assistants,
    on_threads,
)
from ai_agent.graph import build_agui_http_app, build_studio_graph
from ai_agent.http_auth import _client_ip, _enforce_http_throttle
from langgraph_sdk import Auth

User = get_user_model()


def _offline_agent_settings(**overrides):
    from django.conf import settings

    payload = {
        **settings.AI_AGENT,
        "COMPACTION_ENABLED": False,
        "MEMORY_ENABLED": False,
    }
    payload.update(overrides)
    return payload


class _Ctx:
    def __init__(self, identity, action="search"):
        self.user = SimpleNamespace(identity=identity)
        self.action = action
        self.resource = "threads"


class LangGraphAuthenticateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="ada",
            password="pass12345",
            first_name="Ada",
            last_name="Agent",
        )

    def test_healthcheck_allowed_without_token(self):
        result = async_to_sync(authenticate)(
            authorization=None, path="/ok", method="GET"
        )
        self.assertEqual(result["identity"], HEALTHCHECK_IDENTITY)

    def test_healthcheck_trailing_slash(self):
        result = async_to_sync(authenticate)(
            authorization=None, path="/ok/", method="GET"
        )
        self.assertEqual(result["identity"], HEALTHCHECK_IDENTITY)

    def test_missing_token_is_unauthorized(self):
        with self.assertRaises(Auth.exceptions.HTTPException) as caught:
            async_to_sync(authenticate)(
                authorization=None, path="/threads", method="POST"
            )
        self.assertEqual(caught.exception.status_code, 401)

    def test_invalid_token_is_unauthorized(self):
        with self.assertRaises(Auth.exceptions.HTTPException) as caught:
            async_to_sync(authenticate)(
                authorization="Bearer not-a-jwt",
                path="/threads",
                method="GET",
            )
        self.assertEqual(caught.exception.status_code, 401)

    def test_valid_token_returns_user_identity(self):
        result = async_to_sync(authenticate)(
            authorization=f"Bearer {self.user.username}",
            path="/threads",
            method="POST",
        )
        self.assertEqual(result["identity"], str(self.user.pk))
        self.assertEqual(result["display_name"], "Ada Agent")
        self.assertTrue(result["is_authenticated"])

    def test_healthcheck_skips_token_and_db_refresh(self):
        with patch("django.db.close_old_connections") as close, patch(
            "ai_agent.conf.resolve_authenticate_token"
        ) as resolve, patch("ai_agent.db.refresh_db_connections") as refresh:
            result = async_to_sync(authenticate)(
                authorization=None, path="/ok", method="GET"
            )
        self.assertEqual(result["identity"], HEALTHCHECK_IDENTITY)
        close.assert_not_called()
        refresh.assert_not_called()
        resolve.assert_not_called()

    def test_valid_token_refreshes_db_before_authenticate_token(self):
        call_order = []

        def close_old_connections():
            call_order.append("refresh")

        def authenticate_token(token):
            call_order.append("authenticate")
            user = User.objects.filter(username=token).first()
            if user is None:
                raise ValueError("Invalid token")
            return user

        with override_settings(
            TESTING=False,
            AI_AGENT=_offline_agent_settings(AUTHENTICATE_TOKEN=authenticate_token),
        ), patch("django.db.close_old_connections", side_effect=close_old_connections):
            result = async_to_sync(authenticate)(
                authorization=f"Bearer {self.user.username}",
                path="/threads",
                method="POST",
            )
        self.assertEqual(call_order, ["refresh", "authenticate"])
        self.assertEqual(result["identity"], str(self.user.pk))
        self.assertTrue(result["is_authenticated"])

    def test_invalid_token_is_unauthorized_after_db_refresh(self):
        with override_settings(TESTING=False), patch("django.db.close_old_connections"):
            with self.assertRaises(Auth.exceptions.HTTPException) as caught:
                async_to_sync(authenticate)(
                    authorization="Bearer not-a-jwt",
                    path="/threads",
                    method="GET",
                )
        self.assertEqual(caught.exception.status_code, 401)

    def test_display_name_uses_username_then_generic_fallback(self):
        nameless = User.objects.create_user(
            username="nameless",
            password="pass12345",
        )
        self.assertEqual(_display_name(nameless), nameless.get_username())
        nameless.get_username = MagicMock(return_value="")
        self.assertEqual(_display_name(nameless), _FALLBACK_DISPLAY_NAME)
        self.assertNotEqual(_display_name(nameless), str(nameless.pk))


class LangGraphOwnerFilterTests(TestCase):
    def test_create_stamps_owner_metadata(self):
        value = {"metadata": {}}
        filters = async_to_sync(on_threads)(_Ctx("42", action="create"), value)
        self.assertEqual(filters, {"owner": "42"})
        self.assertEqual(value["metadata"]["owner"], "42")

    def test_search_returns_owner_filter(self):
        value = {"limit": 50}
        filters = async_to_sync(on_threads)(_Ctx("7", action="search"), value)
        self.assertEqual(filters, {"owner": "7"})
        self.assertEqual(value["metadata"]["owner"], "7")

    def test_healthcheck_identity_is_forbidden(self):
        with self.assertRaises(Auth.exceptions.HTTPException) as caught:
            async_to_sync(on_threads)(
                _Ctx(HEALTHCHECK_IDENTITY, action="search"), {"metadata": {}}
            )
        self.assertEqual(caught.exception.status_code, 403)


class LangGraphAssistantAuthTests(TestCase):
    def test_search_is_unfiltered_so_builtin_graph_is_visible(self):
        value = {"graph_id": "assistant", "limit": 1}
        filters = async_to_sync(on_assistants)(
            _Ctx("42", action="search"), value
        )
        self.assertEqual(filters, {})
        self.assertNotIn("owner", value.get("metadata") or {})

    def test_read_is_allowed(self):
        filters = async_to_sync(on_assistants)(
            _Ctx("42", action="read"), {"assistant_id": "assistant"}
        )
        self.assertEqual(filters, {})

    def test_create_is_forbidden(self):
        with self.assertRaises(Auth.exceptions.HTTPException) as caught:
            async_to_sync(on_assistants)(
                _Ctx("42", action="create"), {"graph_id": "assistant"}
            )
        self.assertEqual(caught.exception.status_code, 403)

    def test_healthcheck_identity_is_forbidden(self):
        with self.assertRaises(Auth.exceptions.HTTPException) as caught:
            async_to_sync(on_assistants)(
                _Ctx(HEALTHCHECK_IDENTITY, action="search"), {}
            )
        self.assertEqual(caught.exception.status_code, 403)


class AguiAuthGateTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            username="agui",
            password="pass12345",
        )

        def authenticate_token(token: str):
            if token == self.user.username:
                return self.user
            raise ValueError("Invalid token")

        self._agent_override = override_settings(
            AI_AGENT=_offline_agent_settings(AUTHENTICATE_TOKEN=authenticate_token)
        )
        self._agent_override.enable()
        self.addCleanup(self._agent_override.disable)
        graph = build_studio_graph(model=FakeListChatModel(responses=["ok"]))
        self.app = build_agui_http_app(graph)

    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(self.app, raise_server_exceptions=False)

    def test_missing_authorization_is_401(self):
        response = self._client().post("/agui", json={})
        self.assertEqual(response.status_code, 401)

    def test_invalid_authorization_is_401(self):
        response = self._client().post(
            "/agui",
            json={},
            headers={"Authorization": "Bearer not-a-jwt"},
        )
        self.assertEqual(response.status_code, 401)

    def test_valid_token_passes_the_gate(self):
        response = self._client().post(
            "/agui",
            json={},
            headers={"Authorization": f"Bearer {self.user.username}"},
        )
        self.assertNotEqual(response.status_code, 401)

    def test_options_preflight_skips_auth(self):
        response = self._client().options("/agui")
        self.assertNotEqual(response.status_code, 401)

    def test_rate_limit_returns_429(self):
        cache.clear()
        with override_settings(
            AI_AGENT=_offline_agent_settings(HTTP_THROTTLE="1/minute")
        ):
            client = self._client()
            first = client.post(
                "/agui",
                json={},
                headers={"Authorization": "Bearer not-a-jwt"},
            )
            second = client.post(
                "/agui",
                json={},
                headers={"Authorization": "Bearer not-a-jwt"},
            )
        self.assertEqual(first.status_code, 401)
        self.assertEqual(second.status_code, 429)

    def test_options_preflight_does_not_consume_rate_limit(self):
        cache.clear()
        with override_settings(
            AI_AGENT=_offline_agent_settings(HTTP_THROTTLE="1/minute")
        ):
            client = self._client()
            self.assertNotEqual(client.options("/agui").status_code, 401)
            response = client.post(
                "/agui",
                json={},
                headers={"Authorization": "Bearer not-a-jwt"},
            )
        self.assertEqual(response.status_code, 401)


class AguiBearerDbRefreshTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="agui_db",
            password="pass12345",
        )

    def test_require_bearer_user_refreshes_db_before_authenticate_token(self):
        from ai_agent.http_auth import require_bearer_user

        call_order = []

        def close_old_connections():
            call_order.append("refresh")

        def authenticate_token(token):
            call_order.append("authenticate")
            if token == self.user.username:
                return self.user
            raise ValueError("Invalid token")

        request = SimpleNamespace(
            method="POST",
            headers={},
            client=SimpleNamespace(host="203.0.113.9"),
        )
        with override_settings(
            TESTING=False,
            AI_AGENT=_offline_agent_settings(AUTHENTICATE_TOKEN=authenticate_token),
        ), patch("django.db.close_old_connections", side_effect=close_old_connections):
            result = async_to_sync(require_bearer_user)(
                request, authorization=f"Bearer {self.user.username}"
            )
        self.assertEqual(result.pk, self.user.pk)
        self.assertEqual(call_order, ["refresh", "authenticate"])


class HttpThrottleTests(TestCase):
    def setUp(self):
        cache.clear()

    def _request(self, ip="203.0.113.9"):
        return SimpleNamespace(
            headers={"x-forwarded-for": ip},
            client=SimpleNamespace(host=ip),
        )

    def test_second_request_from_same_ip_is_429(self):
        with override_settings(
            AI_AGENT=_offline_agent_settings(HTTP_THROTTLE="1/minute")
        ):
            _enforce_http_throttle(self._request())
            with self.assertRaises(HTTPException) as caught:
                _enforce_http_throttle(self._request())
        self.assertEqual(caught.exception.status_code, 429)

    def test_zero_rate_disables_throttle(self):
        with override_settings(
            AI_AGENT=_offline_agent_settings(HTTP_THROTTLE="0")
        ):
            _enforce_http_throttle(self._request())
            _enforce_http_throttle(self._request())

    def test_spoofed_forwarded_for_is_ignored_by_default(self):
        with override_settings(
            AI_AGENT=_offline_agent_settings(HTTP_THROTTLE="1/minute")
        ):
            first = SimpleNamespace(
                headers={"x-forwarded-for": "198.51.100.1"},
                client=SimpleNamespace(host="203.0.113.9"),
            )
            second = SimpleNamespace(
                headers={"x-forwarded-for": "198.51.100.2"},
                client=SimpleNamespace(host="203.0.113.9"),
            )
            self.assertEqual(_client_ip(first), "203.0.113.9")
            _enforce_http_throttle(first)
            with self.assertRaises(HTTPException) as caught:
                _enforce_http_throttle(second)
        self.assertEqual(caught.exception.status_code, 429)

    def test_trusted_proxy_uses_forwarded_hop(self):
        with override_settings(
            AI_AGENT=_offline_agent_settings(
                HTTP_THROTTLE="1/minute",
                HTTP_TRUSTED_PROXY_COUNT=1,
            )
        ):
            first = SimpleNamespace(
                headers={"x-forwarded-for": "198.51.100.1, 203.0.113.9"},
                client=SimpleNamespace(host="10.0.0.1"),
            )
            second = SimpleNamespace(
                headers={"x-forwarded-for": "198.51.100.99, 203.0.113.9"},
                client=SimpleNamespace(host="10.0.0.1"),
            )
            self.assertEqual(_client_ip(first), "203.0.113.9")
            _enforce_http_throttle(first)
            with self.assertRaises(HTTPException) as caught:
                _enforce_http_throttle(second)
        self.assertEqual(caught.exception.status_code, 429)
