"""User-visible transcript persistence, independent of checkpoint compaction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from django.conf import settings
from django.test import TestCase, override_settings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rest_framework.test import APIClient

from ai_agent.graph import _agent_middleware, build_domain_agent, build_supervisor
from ai_agent.models import AgentMessage
from ai_agent.transcript import persist_transcript, visible_transcript_rows
from django.contrib.auth import get_user_model

User = get_user_model()


class _Runtime:
    def __init__(self, config):
        self.config = config


def _config(user_id, thread_id="thread-1"):
    return {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
        }
    }


class VisibleTranscriptTests(TestCase):
    def test_keeps_user_and_assistant_text(self):
        rows = visible_transcript_rows(
            [
                HumanMessage(content="سلام", id="h1"),
                AIMessage(content="چطور کمک کنم؟", id="a1"),
            ]
        )
        self.assertEqual(
            [(row["role"], row["content"], row["external_id"]) for row in rows],
            [
                ("user", "سلام", "h1"),
                ("assistant", "چطور کمک کنم؟", "a1"),
            ],
        )

    def test_skips_summaries_and_tools(self):
        rows = visible_transcript_rows(
            [
                HumanMessage(
                    content="Here is a summary of the conversation to date:\n\nسلام",
                    id="sum-1",
                    additional_kwargs={"lc_source": "summarization"},
                ),
                AIMessage(
                    content="",
                    id="tool-ai",
                    tool_calls=[
                        {
                            "name": "call_wallet_agent",
                            "args": {"query": "balance"},
                            "id": "c1",
                        }
                    ],
                ),
                ToolMessage(content="ok", tool_call_id="c1", id="t1"),
                HumanMessage(content="فامیل من فتحلی زاده ست", id="h2"),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "فامیل من فتحلی زاده ست")


class PersistTranscriptTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="09001112222", password="pass12345"
        )

    def test_saves_visible_messages_once(self):
        state = {
            "messages": [
                HumanMessage(content="سلام", id="h1"),
                AIMessage(content="hello", id="a1"),
            ]
        }
        runtime = _Runtime(_config(self.user.pk))
        self.assertEqual(persist_transcript(state, runtime), 2)
        self.assertEqual(persist_transcript(state, runtime), 0)
        rows = list(AgentMessage.objects.filter(user=self.user).order_by("id"))
        self.assertEqual([item.role for item in rows], ["user", "assistant"])
        self.assertEqual(rows[0].thread_id, "thread-1")
        self.assertEqual(rows[0].content, "سلام")

    def test_skips_compacted_summary_on_later_turn(self):
        persist_transcript(
            {
                "messages": [
                    HumanMessage(content="سلام", id="h1"),
                    AIMessage(content="hello", id="a1"),
                ]
            },
            _Runtime(_config(self.user.pk)),
        )
        persist_transcript(
            {
                "messages": [
                    HumanMessage(
                        content="Here is a summary of the conversation to date:\n\nسلام",
                        id="sum-1",
                        additional_kwargs={"lc_source": "summarization"},
                    ),
                    HumanMessage(content="من در مونترال زندگی میکنم", id="h2"),
                    AIMessage(content="متوجه شدم", id="a2"),
                ]
            },
            _Runtime(_config(self.user.pk)),
        )
        contents = list(
            AgentMessage.objects.filter(user=self.user)
            .order_by("id")
            .values_list("content", flat=True)
        )
        self.assertEqual(
            contents,
            ["سلام", "hello", "من در مونترال زندگی میکنم", "متوجه شدم"],
        )

    def test_skips_child_threads_and_interrupts(self):
        runtime = _Runtime(_config(self.user.pk, thread_id="parent::wallet"))
        created = persist_transcript(
            {"messages": [HumanMessage(content="x", id="h1")]},
            runtime,
        )
        self.assertEqual(created, 0)
        created = persist_transcript(
            {
                "__interrupt__": True,
                "messages": [HumanMessage(content="x", id="h2")],
            },
            _Runtime(_config(self.user.pk)),
        )
        self.assertEqual(created, 0)
        self.assertEqual(AgentMessage.objects.count(), 0)


class TranscriptMiddlewareWiringTests(TestCase):
    def test_supervisor_gets_transcript_middleware_domain_does_not(self):
        agent_settings = {**settings.AI_AGENT, "COMPACTION_ENABLED": False}
        with override_settings(AI_AGENT=agent_settings):
            kinds = [
                type(item).__name__ for item in _agent_middleware(layer="supervisor")
            ]
            self.assertIn("TranscriptMiddleware", kinds)
            kinds = [type(item).__name__ for item in _agent_middleware(layer="wallet")]
            self.assertNotIn("TranscriptMiddleware", kinds)

    def test_create_agent_includes_transcript_on_supervisor_only(self):
        agent_settings = {**settings.AI_AGENT, "COMPACTION_ENABLED": False}
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=SimpleNamespace()
            ) as mock_create,
        ):
            build_supervisor(stub_subagents=True)
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertIn("TranscriptMiddleware", kinds)

        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=SimpleNamespace()
            ) as mock_create,
        ):
            build_domain_agent("dummy")
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertNotIn("TranscriptMiddleware", kinds)


class AgentTranscriptAPITests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="09001113333", password="pass12345"
        )
        self.other = User.objects.create_user(
            username="09001114444", password="pass12345"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        AgentMessage.objects.create(
            user=self.user,
            thread_id="thread-a",
            role=AgentMessage.ROLE_USER,
            content="سلام",
            external_id="h1",
        )
        AgentMessage.objects.create(
            user=self.user,
            thread_id="thread-a",
            role=AgentMessage.ROLE_ASSISTANT,
            content="hello",
            external_id="a1",
        )
        AgentMessage.objects.create(
            user=self.other,
            thread_id="thread-b",
            role=AgentMessage.ROLE_USER,
            content="secret",
            external_id="h-other",
        )

    def test_lists_own_threads_and_messages(self):
        response = self.client.get("/api/agent/threads/")
        self.assertEqual(response.status_code, 200)
        threads = response.json()["threads"]
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]["thread_id"], "thread-a")
        self.assertEqual(threads[0]["preview"], "hello")

        response = self.client.get("/api/agent/threads/thread-a/messages/")
        self.assertEqual(response.status_code, 200)
        messages = response.json()["messages"]
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "سلام")

    def test_thread_preview_is_latest_message(self):
        AgentMessage.objects.create(
            user=self.user,
            thread_id="thread-a",
            role=AgentMessage.ROLE_USER,
            content="follow-up",
            external_id="h2",
        )
        AgentMessage.objects.create(
            user=self.user,
            thread_id="thread-c",
            role=AgentMessage.ROLE_ASSISTANT,
            content="older-other-thread",
            external_id="c1",
        )
        response = self.client.get("/api/agent/threads/")
        self.assertEqual(response.status_code, 200)
        threads = {item["thread_id"]: item for item in response.json()["threads"]}
        self.assertEqual(set(threads), {"thread-a", "thread-c"})
        self.assertEqual(threads["thread-a"]["preview"], "follow-up")
        self.assertEqual(threads["thread-c"]["preview"], "older-other-thread")

    def test_hides_other_users_threads(self):
        response = self.client.get("/api/agent/threads/thread-b/messages/")
        self.assertEqual(response.status_code, 404)

    def test_requires_auth(self):
        guest = APIClient()
        response = guest.get("/api/agent/threads/")
        self.assertEqual(response.status_code, 401)


class TranscriptAsyncMiddlewareTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="09001115555", password="pass12345"
        )
        self.runtime = _Runtime(_config(self.user.pk))

    def test_async_hooks_persist_without_sync_orm_error(self):
        from asgiref.sync import async_to_sync

        from ai_agent.transcript import TranscriptMiddleware

        middleware = TranscriptMiddleware()

        async def _run():
            await middleware.abefore_agent(
                {"messages": [HumanMessage(content="سلام", id="h1")]},
                self.runtime,
            )
            await middleware.aafter_agent(
                {
                    "messages": [
                        HumanMessage(content="سلام", id="h1"),
                        AIMessage(content="hello", id="a1"),
                    ]
                },
                self.runtime,
            )

        async_to_sync(_run)()
        rows = list(AgentMessage.objects.filter(user=self.user).order_by("id"))
        self.assertEqual([item.role for item in rows], ["user", "assistant"])
        self.assertEqual([item.content for item in rows], ["سلام", "hello"])
