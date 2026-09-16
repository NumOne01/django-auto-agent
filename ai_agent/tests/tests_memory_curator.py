"""Deterministic reconcile, curator submit/recall, and extract-tool isolation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase, override_settings
from langchain_core.messages import HumanMessage

from ai_agent.memory.curator import (
    CURATOR_ASSISTANT_ID,
    curator_after_seconds,
    curator_thread_id,
    run_curator,
    submit_curator,
)
from ai_agent.memory.middleware import MemoryCuratorMiddleware, recall_memory_block
from ai_agent.memory.namespaces import (
    PLAYBOOK_KEY,
    PROFILE_KEY,
    SUPERVISOR_LAYER,
    bind_namespace,
    memory_namespaces,
)
from ai_agent.memory.reconcile import reconcile_layer
from ai_agent.tests.graph_test_utils import make_test_runtime
from ai_agent.memory.tools import build_curator_tools, build_memory_tools, memory_tool_names


def _settings(**overrides):
    payload = {
        **settings.AI_AGENT,
        "MEMORY_ENABLED": True,
        "MEMORY_MODE": "background",
        "MEMORY_STORE": "memory",
        "MEMORY_DEBOUNCE_SECONDS": 0,
        "MEMORY_CURATOR_ENABLED": True,
        "MEMORY_CURATOR_DELAY_SECONDS": 0,
        "MEMORY_FACT_CAP": 30,
        "MEMORY_EPISODE_CAP": 15,
        "MEMORY_COLLECTION_CAP": 20,
        "COMPACTION_ENABLED": False,
    }
    payload.update(overrides)
    return payload


def _put(store, namespace, key, content):
    store.put(namespace, key, {"content": content})


def _contents(store, namespace):
    items = list(store.search(namespace, limit=80) or [])
    return {item.key: item.value.get("content") if isinstance(item.value, dict) else item.value for item in items}


class DeterministicReconcileTests(TestCase):
    def test_semantic_facts_merge_on_subject_predicate(self):
        agent_settings = _settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(memory_namespaces("wallet")["semantic"], user_id="15")
            _put(
                store,
                ns,
                "a",
                {
                    "subject": "User",
                    "predicate": "uses",
                    "object": "gold",
                    "context": "",
                },
            )
            _put(
                store,
                ns,
                "b",
                {
                    "subject": "user",
                    "predicate": "USES",
                    "object": "gold wallet",
                    "context": "bullion",
                },
            )
            result = reconcile_layer(store, "15", "wallet")
        remaining = _contents(store, ns)
        self.assertEqual(len(remaining), 1)
        kept = next(iter(remaining.values()))
        self.assertEqual(kept["object"], "gold wallet")
        self.assertEqual(kept["context"], "bullion")
        self.assertTrue(result.dirty)
        self.assertFalse(result.needs_llm)

    def test_dummy_account_natural_keys(self):
        agent_settings = _settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            dummy_ns = bind_namespace(
                memory_namespaces("dummy")["semantic"], user_id="15"
            )
            _put(
                store,
                dummy_ns,
                "c1",
                {"bank_name": "Acme", "label": "primary", "use_for": None},
            )
            _put(
                store,
                dummy_ns,
                "c2",
                {
                    "bank_name": "Acme",
                    "label": "primary",
                    "use_for": "withdraw",
                },
            )
            reconcile_layer(store, "15", "dummy")
            accounts = _contents(store, dummy_ns)
            self.assertEqual(len(accounts), 1)
            self.assertEqual(next(iter(accounts.values()))["use_for"], "withdraw")

    def test_schema_without_natural_key_is_not_merge_deduped_by_class_name(self):
        from typing import ClassVar

        from pydantic import BaseModel

        from ai_agent.memory.reconcile import _natural_key, _schema_natural_key

        class UntaggedNote(BaseModel):
            title: str
            body: str

        class TaggedNote(BaseModel):
            natural_key: ClassVar[tuple[str, ...]] = ("title",)
            title: str
            body: str

        self.assertEqual(_schema_natural_key(UntaggedNote), ())
        self.assertEqual(_schema_natural_key(TaggedNote), ("title",))
        content = {"title": "same", "body": "one"}
        self.assertIsNone(_natural_key("UntaggedNote", content, {}))
        self.assertEqual(
            _natural_key("TaggedNote", content, {"TaggedNote": ("title",)}),
            ("same",),
        )

    def test_profile_uuid_keys_collapse_to_default(self):
        agent_settings = _settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(
                memory_namespaces(SUPERVISOR_LAYER)["profile"], user_id="15"
            )
            _put(store, ns, "uuid-1", {"name": "محمد", "language": "fa"})
            _put(store, ns, "uuid-2", {"name": "رضا", "language": "fa"})
            reconcile_layer(store, "15", SUPERVISOR_LAYER)
            items = _contents(store, ns)
            self.assertEqual(list(items), [PROFILE_KEY])
            self.assertEqual(items[PROFILE_KEY]["language"], "fa")

    def test_fact_matching_profile_field_is_deleted(self):
        agent_settings = _settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            profile_ns = bind_namespace(
                memory_namespaces("trading")["profile"], user_id="15"
            )
            fact_ns = bind_namespace(
                memory_namespaces("trading")["semantic"], user_id="15"
            )
            _put(store, profile_ns, PROFILE_KEY, {"default_quote": "IRT"})
            _put(
                store,
                fact_ns,
                "f1",
                {
                    "subject": "user",
                    "predicate": "quote",
                    "object": "IRT",
                    "context": None,
                },
            )
            reconcile_layer(store, "15", "trading")
            self.assertEqual(_contents(store, fact_ns), {})

    def test_episode_cap_keeps_failure_over_success(self):
        agent_settings = _settings(MEMORY_EPISODE_CAP=2)
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(memory_namespaces("wallet")["episodes"], user_id="15")
            _put(
                store,
                ns,
                "old-ok",
                {
                    "observation": "check balance",
                    "thoughts": "I looked",
                    "action": "get_user_wallet",
                    "result": "ok",
                },
            )
            _put(
                store,
                ns,
                "fail",
                {
                    "observation": "withdraw",
                    "thoughts": "I tried",
                    "action": "withdraw",
                    "result": "failed: missing amount. Prevent by asking first.",
                },
            )
            _put(
                store,
                ns,
                "newer-ok",
                {
                    "observation": "list cards",
                    "thoughts": "I listed",
                    "action": "list_cards",
                    "result": "ok",
                },
            )
            result = reconcile_layer(store, "15", "wallet")
            remaining = _contents(store, ns)
            self.assertEqual(len(remaining), 2)
            self.assertIn("fail", remaining)
            self.assertTrue(result.needs_optimize)
            self.assertFalse(result.needs_llm)


class CuratorGraphSkipTests(TestCase):
    def test_run_curator_skips_llm_when_under_cap(self):
        agent_settings = _settings()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.memory.curator._run_llm_curate") as curate,
            patch("ai_agent.memory.curator._run_local_optimize") as optimize,
        ):
            store = make_test_runtime().store
            ns = bind_namespace(memory_namespaces("wallet")["semantic"], user_id="15")
            _put(
                store,
                ns,
                "one",
                {
                    "subject": "user",
                    "predicate": "likes",
                    "object": "silver",
                    "context": None,
                },
            )
            run_curator("15", "wallet", store=store)
        curate.assert_not_called()
        optimize.assert_not_called()


class CuratorSubmitTests(TestCase):
    def test_submit_curator_skipped_when_memory_disabled(self):
        agent_settings = _settings(MEMORY_ENABLED=False)
        fake = MagicMock()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.memory.curator._curator_executor", return_value=fake),
        ):
            submit_curator(
                "wallet", {"configurable": {"user_id": "9", "thread_id": "t-9"}}
            )
        fake.submit.assert_not_called()

    def test_submit_curator_delay_and_thread_id(self):
        agent_settings = _settings(
            MEMORY_DEBOUNCE_SECONDS=12, MEMORY_CURATOR_DELAY_SECONDS=5
        )
        fake = MagicMock()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.memory.curator._curator_executor", return_value=fake),
        ):
            submit_curator(
                "wallet", {"configurable": {"user_id": "9", "thread_id": "chat-9"}}
            )
            self.assertEqual(curator_after_seconds(), 17)
        fake.submit.assert_called_once()
        self.assertEqual(fake.submit.call_args.kwargs["after_seconds"], 17)
        self.assertEqual(fake.submit.call_args.kwargs["thread_id"], "curator:9:wallet")
        self.assertEqual(curator_thread_id("9", "wallet"), "curator:9:wallet")

    def test_remote_executor_uses_memory_curator_assistant(self):
        from ai_agent.memory.managers import _RemoteMemoryExecutor

        agent_settings = _settings(MEMORY_STORE="platform")
        client = MagicMock()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.memory.store._force_in_memory_store", return_value=False
            ),
            patch("ai_agent.memory.managers._use_remote_executor", return_value=True),
            patch(
                "ai_agent.memory.managers._remote_executor_url",
                return_value="http://langgraph",
            ),
            patch("langgraph_sdk.get_sync_client", return_value=client),
        ):
            from ai_agent.memory import curator as curator_mod

            executor = curator_mod._curator_executor("wallet")
        self.assertIsInstance(executor, _RemoteMemoryExecutor)
        self.assertEqual(executor._assistant_id, CURATOR_ASSISTANT_ID)


class CuratorMiddlewareTests(TestCase):
    def test_hot_middleware_submits_and_skips_interrupt(self):
        from ai_agent.graph import _agent_middleware
        from ai_agent.memory.middleware import MemoryWriteMiddleware

        hot = _settings(MEMORY_MODE="hot")
        with override_settings(AI_AGENT=hot):
            store = make_test_runtime().store
            kinds = [type(item).__name__ for item in _agent_middleware(layer="wallet")]
        self.assertIn("MemoryCuratorMiddleware", kinds)
        self.assertNotIn("MemoryWriteMiddleware", kinds)

        background = _settings(MEMORY_MODE="background")
        with override_settings(AI_AGENT=background):
            store = make_test_runtime().store
            kinds = [type(item).__name__ for item in _agent_middleware(layer="wallet")]
        self.assertIn("MemoryWriteMiddleware", kinds)
        self.assertNotIn("MemoryCuratorMiddleware", kinds)

        agent_settings = _settings()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.memory.curator.submit_curator") as submit,
        ):
            middleware = MemoryCuratorMiddleware("wallet")
            middleware.after_agent(
                {"messages": [HumanMessage(content="hi")], "__interrupt__": True},
                runtime=None,
            )
            submit.assert_not_called()
            middleware.after_agent(
                {"messages": [HumanMessage(content="hi")]},
                runtime=None,
            )
            submit.assert_called_once()

        with (
            override_settings(AI_AGENT=agent_settings),
            patch("ai_agent.memory.managers.submit_memory") as mem,
            patch("ai_agent.memory.curator.submit_curator") as cur,
        ):
            writer = MemoryWriteMiddleware("wallet")
            writer.after_agent(
                {"messages": [HumanMessage(content="hi")]},
                runtime=None,
            )
        mem.assert_called_once()
        cur.assert_called_once()


class PlaybookRecallAndToolsTests(TestCase):
    def test_playbook_recall_and_extract_has_no_playbook_tool(self):
        agent_settings = _settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(memory_namespaces("wallet")["playbook"], user_id="15")
            _put(
                store,
                ns,
                PLAYBOOK_KEY,
                {
                    "rules": [
                        {
                            "trigger": "withdraw",
                            "do": "confirm amount first",
                            "dont": None,
                            "why": "user declined once",
                        }
                    ]
                },
            )
            runtime = MagicMock()
            runtime.store = store
            runtime.config = {"configurable": {"user_id": "15"}}
            block = recall_memory_block(
                "wallet", [HumanMessage(content="withdraw")], runtime=runtime
            )
            self.assertIn("<playbook>", block)
            self.assertIn("confirm amount first", block)

            extract_names = {tool.name for tool in build_memory_tools("wallet", store=store)}
            self.assertNotIn("wallet_manage_playbook", extract_names)
            self.assertTrue(memory_tool_names("wallet") <= extract_names)
            curator_names = {
                tool.name for tool in build_curator_tools("wallet", store=store)
            }
            self.assertIn("wallet_manage_playbook", curator_names)
            self.assertIn("wallet_list_layer_memories", curator_names)


class ReconcileCommandTests(TestCase):
    def test_command_requires_layer(self):
        from django.core.management import call_command
        from django.core.management.base import CommandError

        with override_settings(AI_AGENT=_settings()):
            with self.assertRaises(CommandError):
                call_command("reconcile_agent_memory", "--user-id", "15")

    def test_command_invokes_run_curator(self):
        from django.core.management import call_command

        with (
            override_settings(AI_AGENT=_settings()),
            patch("ai_agent.memory.curator.run_curator") as run,
        ):
            call_command(
                "reconcile_agent_memory",
                "--user-id",
                "15",
                "--layer",
                "wallet",
            )
        run.assert_called_once()
        self.assertEqual(run.call_args.args[:2], ("15", "wallet"))
