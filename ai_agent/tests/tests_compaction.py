"""Conversation compaction wiring: enable/disable, model, trigger, and keep."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase, override_settings

from ai_agent.compaction import (
    _quiet_summary_config,
    _silence_summarizer_stream,
    build_compaction_middleware,
)
from ai_agent.conf import compaction_enabled, get_agent_settings
from ai_agent.graph import _agent_middleware, build_domain_agent, build_supervisor


def _compaction_settings(**overrides):
    payload = {
        **settings.AI_AGENT,
        "COMPACTION_ENABLED": True,
        "MEMORY_ENABLED": False,
        "COMPACTION_TRIGGER_TYPE": "tokens",
        "COMPACTION_TRIGGER": 8000,
        "COMPACTION_KEEP_TYPE": "messages",
        "COMPACTION_KEEP": 20,
    }
    payload.update(overrides)
    return payload


class CompactionDisabledTests(TestCase):
    def test_no_summarization_middleware_when_disabled(self):
        agent_settings = _compaction_settings(COMPACTION_ENABLED=False)
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("wallet")
            kinds = [
                type(item).__name__
                for item in mock_create.call_args.kwargs["middleware"]
            ]
            self.assertNotIn("SummarizationMiddleware", kinds)
            self.assertNotIn("_QuietSummarizationMiddleware", kinds)
            self.assertFalse(compaction_enabled())


class CompactionEnabledTests(TestCase):
    def test_supervisor_and_domain_include_compaction_middleware(self):
        fake = object()
        agent_settings = _compaction_settings()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.compaction.build_compaction_middleware", return_value=fake
            ),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor(stub_subagents=True)
        self.assertIn(fake, mock_create.call_args.kwargs["middleware"])

        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.compaction.build_compaction_middleware", return_value=fake
            ),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("wallet")
        self.assertIn(fake, mock_create.call_args.kwargs["middleware"])

    def test_compaction_runs_after_copilotkit_and_before_memory(self):
        fake = object()
        agent_settings = _compaction_settings(
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


class CompactionConfigTests(TestCase):
    def test_defaults_to_token_trigger_and_message_keep(self):
        agent_settings = _compaction_settings(COMPACTION_MODEL="openai:summarizer")
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.compaction._QuietSummarizationMiddleware") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            build_compaction_middleware()
        kwargs = mock_cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai:summarizer")
        self.assertEqual(kwargs["trigger"], ("tokens", 8000))
        self.assertEqual(kwargs["keep"], ("messages", 20))

    def test_model_defaults_to_subagent_when_empty(self):
        agent_settings = _compaction_settings(
            COMPACTION_MODEL="",
            SUBAGENT_MODEL="openai:subagent-test",
        )
        with override_settings(AI_AGENT=agent_settings):
            self.assertEqual(
                get_agent_settings().compaction_model, "openai:subagent-test"
            )

    def test_trigger_and_keep_tokens(self):
        agent_settings = _compaction_settings(
            COMPACTION_TRIGGER_TYPE="tokens",
            COMPACTION_TRIGGER=4000,
            COMPACTION_KEEP_TYPE="tokens",
            COMPACTION_KEEP=1500,
        )
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.compaction._QuietSummarizationMiddleware") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            build_compaction_middleware()
        kwargs = mock_cls.call_args.kwargs
        self.assertEqual(kwargs["trigger"], ("tokens", 4000))
        self.assertEqual(kwargs["keep"], ("tokens", 1500))

    def test_trigger_and_keep_messages(self):
        agent_settings = _compaction_settings(
            COMPACTION_TRIGGER_TYPE="messages",
            COMPACTION_TRIGGER=40,
            COMPACTION_KEEP_TYPE="messages",
            COMPACTION_KEEP=8,
        )
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.compaction._QuietSummarizationMiddleware") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            build_compaction_middleware()
        kwargs = mock_cls.call_args.kwargs
        self.assertEqual(kwargs["trigger"], ("messages", 40))
        self.assertEqual(kwargs["keep"], ("messages", 8))

    def test_trigger_and_keep_fraction(self):
        agent_settings = _compaction_settings(
            COMPACTION_TRIGGER_TYPE="fraction",
            COMPACTION_TRIGGER=0.8,
            COMPACTION_KEEP_TYPE="fraction",
            COMPACTION_KEEP=0.3,
        )
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.compaction._QuietSummarizationMiddleware") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            build_compaction_middleware()
        kwargs = mock_cls.call_args.kwargs
        self.assertEqual(kwargs["trigger"], ("fraction", 0.8))
        self.assertEqual(kwargs["keep"], ("fraction", 0.3))

    def test_invalid_types_fall_back_to_defaults(self):
        agent_settings = _compaction_settings(
            COMPACTION_TRIGGER_TYPE="bogus",
            COMPACTION_TRIGGER=4000,
            COMPACTION_KEEP_TYPE="nope",
            COMPACTION_KEEP=8,
        )
        with override_settings(AI_AGENT=agent_settings):
            parsed = get_agent_settings()
        self.assertEqual(parsed.compaction_trigger_type, "tokens")
        self.assertEqual(parsed.compaction_trigger, 4000)
        self.assertEqual(parsed.compaction_keep_type, "messages")
        self.assertEqual(parsed.compaction_keep, 8)


class CompactionStreamSilenceTests(TestCase):
    def test_quiet_config_sets_copilotkit_and_agui_flags(self):
        config = _quiet_summary_config(
            {"metadata": {"lc_source": "summarization", "keep": "me"}}
        )
        metadata = config["metadata"]
        self.assertEqual(metadata["lc_source"], "summarization")
        self.assertEqual(metadata["keep"], "me")
        self.assertIs(metadata["copilotkit:emit-messages"], False)
        self.assertIs(metadata["copilotkit:emit-tool-calls"], False)
        self.assertIs(metadata["emit-messages"], False)
        self.assertIs(metadata["emit-tool-calls"], False)

    def test_quiet_config_does_not_mutate_parent_metadata(self):
        parent_meta = {"lc_source": "agent"}
        parent = {"metadata": parent_meta, "tags": ["keep"]}
        quiet = _quiet_summary_config(parent)
        self.assertEqual(parent_meta, {"lc_source": "agent"})
        self.assertNotIn("copilotkit:emit-messages", parent_meta)
        self.assertNotIn("emit-messages", parent_meta)
        self.assertIs(quiet["metadata"]["copilotkit:emit-messages"], False)
        self.assertEqual(parent["tags"], ["keep"])

    def test_build_wraps_summary_model_invoke(self):
        inner = MagicMock()
        inner.invoke.return_value = "ok"
        middleware = _silence_summarizer_stream(
            SimpleNamespace(_summary_model=inner)
        )
        middleware._summary_model.invoke(
            "prompt",
            config={"metadata": {"lc_source": "summarization"}},
        )
        passed = inner.invoke.call_args.kwargs["config"]["metadata"]
        self.assertEqual(passed["lc_source"], "summarization")
        self.assertIs(passed["copilotkit:emit-messages"], False)
        self.assertIs(passed["emit-messages"], False)

    def test_build_wraps_summary_model_ainvoke(self):
        inner = MagicMock()

        async def _ainvoke(input, config=None, **kwargs):
            inner.last_config = config
            return "ok"

        inner.ainvoke.side_effect = _ainvoke
        middleware = _silence_summarizer_stream(
            SimpleNamespace(_summary_model=inner)
        )
        asyncio.run(middleware._summary_model.ainvoke("prompt"))
        passed = inner.last_config["metadata"]
        self.assertIs(passed["copilotkit:emit-messages"], False)
        self.assertIs(passed["emit-messages"], False)

    def test_warns_when_summary_model_is_missing(self):
        middleware = SimpleNamespace()
        with self.assertLogs("ai_agent.compaction", level="WARNING") as captured:
            result = _silence_summarizer_stream(middleware)
        self.assertIs(result, middleware)
        self.assertTrue(any("_summary_model" in line for line in captured.output))
