"""Long-term memory wiring: enable/disable, hot vs background, namespaces."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.conf import settings
from django.test import TestCase, override_settings
from langchain_core.messages import HumanMessage

from ai_agent.graph import build_domain_agent, build_supervisor, subagent_tool_name
from ai_agent.memory.middleware import arecall_memory_block, format_memory_block, recall_memory_block
from ai_agent.memory.namespaces import (
    PROFILE_KEY,
    SUPERVISOR_LAYER,
    bind_namespace,
    memory_namespaces,
    normalize_memory_config,
    user_id_from_config,
)
from ai_agent.memory.spec import resolve_memory_spec
from ai_agent.tests.graph_test_utils import make_test_runtime
from ai_agent.memory.store import build_memory_store
from ai_agent.memory.supervisor_schemas import (
    DomainHabit,
    SupervisorFact,
    SupervisorProfile,
)
from ai_agent.memory.tools import memory_tool_names


def _memory_settings(**overrides):
    payload = {
        **settings.AI_AGENT,
        "MEMORY_ENABLED": True,
        "MEMORY_MODE": "background",
        "MEMORY_STORE": "memory",
        "MEMORY_DEBOUNCE_SECONDS": 0,
        "MEMORY_CURATOR_ENABLED": False,
        "COMPACTION_ENABLED": False,
    }
    payload.update(overrides)
    return payload


class MemoryConfigTests(TestCase):
    def test_supervisor_spec_uses_app_config_schemas(self):
        spec = resolve_memory_spec(SUPERVISOR_LAYER)
        self.assertIs(spec.profile, SupervisorProfile)
        self.assertEqual(spec.collections, (SupervisorFact, DomainHabit))
        self.assertTrue(
            {"name", "language", "preferred_domains"} <= set(spec.profile.model_fields)
        )
        self.assertNotIn("current_state", spec.profile.model_fields)
        self.assertNotIn("goals", spec.profile.model_fields)
        self.assertNotIn("natural_key", spec.collections[0].model_fields)
        self.assertEqual(spec.collections[0].natural_key, ("subject", "predicate"))
        self.assertEqual(spec.collections[1].natural_key, ("domain",))

    def test_domain_apps_use_app_config_schemas(self):
        from ai_agent.memory.schemas import Episode, SemanticFact
        from dummy.memory_schemas import DummyAccount, DummyProfile

        spec = resolve_memory_spec("dummy")
        self.assertIs(spec.profile, DummyProfile)
        self.assertEqual(spec.collections, (SemanticFact, DummyAccount))
        self.assertIs(spec.episode, Episode)
        self.assertEqual(spec.collections[1].natural_key, ("bank_name", "label"))


class NestedSupervisorCatalogTests(TestCase):
    def test_supervisor_tools_are_only_domain_subagents_when_memory_off(self):
        agent_settings = _memory_settings(MEMORY_ENABLED=False)
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor()
        names = [tool.name for tool in mock_create.call_args.kwargs["tools"]]
        self.assertTrue(names)
        self.assertTrue(all(item.startswith("call_") for item in names))
        self.assertIn(subagent_tool_name("dummy"), names)
        self.assertNotIn("dummy_item_list", names)
        self.assertIsNone(mock_create.call_args.kwargs.get("store"))


class MemoryDisabledTests(TestCase):
    def test_no_memory_tools_or_store_when_disabled(self):
        agent_settings = _memory_settings(MEMORY_ENABLED=False)
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("dummy")
        names = [tool.name for tool in mock_create.call_args.kwargs["tools"]]
        self.assertFalse(memory_tool_names("dummy") & set(names))
        self.assertIsNone(mock_create.call_args.kwargs.get("store"))
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertNotIn("MemoryRecallMiddleware", kinds)
        self.assertNotIn("MemoryWriteMiddleware", kinds)


_MEMORY_IN_PLACE = "Long-term memory is in place"
_BACKGROUND_SAVER = "background memory saver"
_USE_MEMORY_TOOLS = "Use the memory tools"


def _supervisor_and_domain_prompts(**overrides):
    agent_settings = _memory_settings(**overrides)
    with (
        override_settings(AI_AGENT=agent_settings),
        patch("ai_agent.graph.create_agent", return_value=MagicMock()) as mock_create,
    ):
        build_supervisor(stub_subagents=True)
        supervisor = mock_create.call_args.kwargs["system_prompt"]
    with (
        override_settings(AI_AGENT=agent_settings),
        patch("ai_agent.graph.create_agent", return_value=MagicMock()) as mock_create,
    ):
        build_domain_agent("dummy")
        domain = mock_create.call_args.kwargs["system_prompt"]
    return supervisor, domain


class MemoryPromptTests(TestCase):
    def test_disabled_omits_memory_section(self):
        supervisor, domain = _supervisor_and_domain_prompts(MEMORY_ENABLED=False)
        for prompt in (supervisor, domain):
            self.assertNotIn(_MEMORY_IN_PLACE, prompt)
            self.assertNotIn(_BACKGROUND_SAVER, prompt)
            self.assertNotIn(_USE_MEMORY_TOOLS, prompt)

    def test_background_mentions_memory_and_saver(self):
        supervisor, domain = _supervisor_and_domain_prompts(MEMORY_MODE="background")
        for prompt in (supervisor, domain):
            self.assertIn(_MEMORY_IN_PLACE, prompt)
            self.assertIn(_BACKGROUND_SAVER, prompt)
            self.assertNotIn(_USE_MEMORY_TOOLS, prompt)

    def test_hot_mentions_memory_and_tools_not_saver(self):
        supervisor, domain = _supervisor_and_domain_prompts(MEMORY_MODE="hot")
        for prompt in (supervisor, domain):
            self.assertIn(_MEMORY_IN_PLACE, prompt)
            self.assertIn(_USE_MEMORY_TOOLS, prompt)
            self.assertNotIn(_BACKGROUND_SAVER, prompt)


class MemoryHotPathTests(TestCase):
    def test_hot_path_attaches_memory_tools_to_supervisor_and_domain(self):
        agent_settings = _memory_settings(MEMORY_MODE="hot")
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor(stub_subagents=True)
        supervisor_names = {
            tool.name for tool in mock_create.call_args.kwargs["tools"]
        }
        self.assertTrue(memory_tool_names("supervisor") <= supervisor_names)
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertIn("MemoryRecallMiddleware", kinds)
        self.assertNotIn("MemoryWriteMiddleware", kinds)
        self.assertIn("MemoryCuratorMiddleware", kinds)

        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("dummy")
        domain_names = {tool.name for tool in mock_create.call_args.kwargs["tools"]}
        self.assertTrue(memory_tool_names("dummy") <= domain_names)


class MemoryBackgroundTests(TestCase):
    def test_background_has_write_middleware_not_memory_tools(self):
        agent_settings = _memory_settings(MEMORY_MODE="background")
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("dummy")
        names = {tool.name for tool in mock_create.call_args.kwargs["tools"]}
        self.assertFalse(memory_tool_names("dummy") & names)
        kinds = [
            type(item).__name__
            for item in mock_create.call_args.kwargs["middleware"]
        ]
        self.assertIn("MemoryRecallMiddleware", kinds)
        self.assertIn("MemoryWriteMiddleware", kinds)
        self.assertNotIn("MemoryCuratorMiddleware", kinds)

    def test_write_middleware_submits_after_agent(self):
        from ai_agent.memory.middleware import MemoryWriteMiddleware

        agent_settings = _memory_settings(MEMORY_MODE="background")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            middleware = MemoryWriteMiddleware("wallet")
            with patch(
                "ai_agent.memory.managers.submit_memory"
            ) as submit:
                middleware.after_agent(
                    {
                        "messages": [
                            HumanMessage(content="hello"),
                        ]
                    },
                    runtime=None,
                )
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[0], "wallet")

    def test_write_middleware_skips_interrupts(self):
        from ai_agent.memory.middleware import MemoryWriteMiddleware

        agent_settings = _memory_settings(MEMORY_MODE="background")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            middleware = MemoryWriteMiddleware("supervisor")
            with patch(
                "ai_agent.memory.managers.submit_memory"
            ) as submit:
                middleware.after_agent(
                    {
                        "messages": [HumanMessage(content="x")],
                        "__interrupt__": True,
                    },
                    runtime=None,
                )
        submit.assert_not_called()

    def test_submit_memory_uses_debounce(self):
        from ai_agent.memory import managers

        agent_settings = _memory_settings(
            MEMORY_MODE="background", MEMORY_DEBOUNCE_SECONDS=12
        )
        fake_executor = MagicMock()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch.object(managers, "_reflection_executor", return_value=fake_executor),
        ):
            managers.submit_memory(
                "supervisor",
                [HumanMessage(content="hi")],
                {"configurable": {"user_id": "9", "thread_id": "t-9"}},
            )
        fake_executor.submit.assert_called_once()
        self.assertEqual(fake_executor.submit.call_args.kwargs["after_seconds"], 12)
        self.assertEqual(fake_executor.submit.call_args.kwargs["thread_id"], "t-9")

    def test_submit_memory_skips_when_hot(self):
        from ai_agent.memory import managers

        agent_settings = _memory_settings(MEMORY_MODE="hot")
        fake_executor = MagicMock()
        with (
            override_settings(AI_AGENT=agent_settings),
            patch.object(managers, "_reflection_executor", return_value=fake_executor),
        ):
            managers.submit_memory(
                "supervisor",
                [HumanMessage(content="hi")],
                {"configurable": {"user_id": "9", "thread_id": "t-9"}},
            )
        fake_executor.submit.assert_not_called()

    def test_write_middleware_passes_runtime_store(self):
        from ai_agent.memory.middleware import MemoryWriteMiddleware

        runtime = MagicMock()
        runtime.store = object()
        runtime.config = {"configurable": {"user_id": "9"}}
        agent_settings = _memory_settings(MEMORY_MODE="background")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            middleware = MemoryWriteMiddleware("supervisor")
            with patch(
                "ai_agent.memory.managers.submit_memory"
            ) as submit:
                middleware.after_agent(
                    {"messages": [HumanMessage(content="hi")]},
                    runtime=runtime,
                )
        self.assertIs(submit.call_args.kwargs["store"], runtime.store)

    def test_local_reflection_executor_accepts_create_agent_graph(self):
        from ai_agent.memory.managers import _NamespacedReflector, _reflection_executor

        fake_agent = MagicMock()
        fake_store = MagicMock()
        agent_settings = _memory_settings(MEMORY_MODE="background")
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.memory.managers._local_memory_agent",
                return_value=fake_agent,
            ),
        ):
            executor = _reflection_executor("wallet", store=fake_store)
        try:
            self.assertTrue(hasattr(executor, "submit"))
            self.assertEqual(
                _NamespacedReflector(fake_agent).namespace.template,
                ("memories", "{user_id}"),
            )
        finally:
            shutdown = getattr(executor, "shutdown", None)
            if shutdown:
                shutdown(wait=False, cancel_futures=True)


class MemoryStoreIsolationTests(TestCase):
    def test_profile_semantic_and_episode_namespaces_are_isolated(self):
        agent_settings = _memory_settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            user_a = "1"
            user_b = "2"
            wallet_ns = bind_namespace(
                memory_namespaces("wallet")["semantic"], user_id=user_a
            )
            trading_ns = bind_namespace(
                memory_namespaces("trading")["semantic"], user_id=user_a
            )
            supervisor_ns = bind_namespace(
                memory_namespaces(SUPERVISOR_LAYER)["semantic"], user_id=user_a
            )
            other_user = bind_namespace(
                memory_namespaces("wallet")["semantic"], user_id=user_b
            )
            store.put(wallet_ns, "w1", {"kind": "fact", "content": {"n": "wallet"}})
            store.put(trading_ns, "t1", {"kind": "fact", "content": {"n": "trading"}})
            store.put(
                supervisor_ns, "s1", {"kind": "fact", "content": {"n": "supervisor"}}
            )
            store.put(other_user, "o1", {"kind": "fact", "content": {"n": "other"}})

            wallet_hit = [item.value["content"]["n"] for item in store.search(wallet_ns)]
            self.assertEqual(wallet_hit, ["wallet"])
            trading_hit = [
                item.value["content"]["n"] for item in store.search(trading_ns)
            ]
            self.assertEqual(trading_hit, ["trading"])
            supervisor_hit = [
                item.value["content"]["n"] for item in store.search(supervisor_ns)
            ]
            self.assertEqual(supervisor_hit, ["supervisor"])
            other_hit = [item.value["content"]["n"] for item in store.search(other_user)]
            self.assertEqual(other_hit, ["other"])

    def test_profile_update_replaces_same_key(self):
        agent_settings = _memory_settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(
                memory_namespaces(SUPERVISOR_LAYER)["profile"], user_id="7"
            )
            store.put(
                ns, PROFILE_KEY, {"kind": "profile", "content": {"goals": "A"}}
            )
            store.put(
                ns, PROFILE_KEY, {"kind": "profile", "content": {"goals": "B"}}
            )
            items = list(store.search(ns))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].value["content"]["goals"], "B")

    def test_semantic_create_update_delete(self):
        agent_settings = _memory_settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(
                memory_namespaces("wallet")["semantic"], user_id="7"
            )
            store.put(ns, "f1", {"kind": "fact", "content": {"n": "one"}})
            store.put(ns, "f1", {"kind": "fact", "content": {"n": "two"}})
            self.assertEqual(
                store.get(ns, "f1").value["content"]["n"],
                "two",
            )
            store.delete(ns, "f1")
            self.assertIsNone(store.get(ns, "f1"))

    def test_recall_formats_profile_facts_and_episodes(self):
        agent_settings = _memory_settings()
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            user_id = "42"
            store.put(
                bind_namespace(
                    memory_namespaces("wallet")["profile"], user_id=user_id
                ),
                PROFILE_KEY,
                {
                    "kind": "profile",
                    "content": {"preferences": "reply in fa"},
                },
            )
            store.put(
                bind_namespace(
                    memory_namespaces("wallet")["semantic"], user_id=user_id
                ),
                "f1",
                {
                    "kind": "fact",
                    "content": {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": "IRT",
                    },
                },
            )
            store.put(
                bind_namespace(
                    memory_namespaces("wallet")["episodes"], user_id=user_id
                ),
                "e1",
                {
                    "kind": "episode",
                    "content": {
                        "observation": "asked for balance",
                        "thoughts": "I listed wallets",
                        "action": "get_user_wallet",
                        "result": "ok",
                    },
                },
            )
            runtime = MagicMock()
            runtime.store = store
            runtime.config = {"configurable": {"user_id": user_id}}
            block = recall_memory_block(
                "wallet",
                [HumanMessage(content="موجودی")],
                runtime=runtime,
            )
        self.assertIn("<profile>", block)
        self.assertIn("<facts>", block)
        self.assertIn("<episodes>", block)
        self.assertIn("IRT", block)

    def test_async_recall_uses_asearch_on_event_loop_store(self):
        import asyncio
        from types import SimpleNamespace

        from asgiref.sync import async_to_sync

        class LoopStore:
            def __init__(self):
                self.searched_sync = False
                self.searched_async = False

            def search(self, namespace, **kwargs):
                self.searched_sync = True
                raise asyncio.InvalidStateError("use asearch")

            async def asearch(self, namespace, **kwargs):
                self.searched_async = True
                return [
                    SimpleNamespace(
                        value={"kind": "fact", "content": {"object": "async-hit"}}
                    )
                ]

        store = LoopStore()
        runtime = MagicMock()
        runtime.store = store
        runtime.config = {"configurable": {"user_id": "22"}}
        block = async_to_sync(arecall_memory_block)(
            "supervisor",
            [HumanMessage(content="hello")],
            runtime=runtime,
        )
        self.assertTrue(store.searched_async)
        self.assertFalse(store.searched_sync)
        self.assertIn("async-hit", block)

    def test_format_memory_block_empty(self):
        self.assertEqual(format_memory_block([], [], []), "")

    def test_user_id_from_auth_identity(self):
        self.assertEqual(
            user_id_from_config(
                {"configurable": {"langgraph_auth_user_id": "99"}}
            ),
            "99",
        )

    def test_user_id_from_config_prefers_auth_over_forged_user_id(self):
        self.assertEqual(
            user_id_from_config(
                {
                    "configurable": {
                        "user_id": "forged",
                        "langgraph_auth_user_id": "99",
                    }
                }
            ),
            "99",
        )

    @override_settings(TESTING=False)
    def test_user_id_from_config_ignores_client_user_id_outside_tests(self):
        self.assertIsNone(
            user_id_from_config({"configurable": {"user_id": "forged"}})
        )

    def test_user_id_from_config_allows_user_id_in_tests(self):
        self.assertEqual(
            user_id_from_config({"configurable": {"user_id": "22"}}),
            "22",
        )

    @override_settings(TESTING=False)
    def test_normalize_memory_config_does_not_promote_forged_user_id(self):
        bound = normalize_memory_config(
            {"configurable": {"user_id": "forged"}}, "wallet"
        )
        configurable = bound["configurable"]
        self.assertEqual(configurable["memory_layer"], "wallet")
        self.assertNotIn("user_id", configurable)
        self.assertIsNone(user_id_from_config(bound))
        self.assertNotEqual(
            configurable.get("langgraph_auth_user_id"), "forged"
        )

    def test_normalize_memory_config_overwrites_forged_user_id_with_auth(self):
        bound = normalize_memory_config(
            {
                "configurable": {
                    "user_id": "forged",
                    "langgraph_auth_user_id": "99",
                }
            },
            "wallet",
        )
        self.assertEqual(bound["configurable"]["user_id"], "99")
        self.assertEqual(bound["configurable"]["langgraph_auth_user_id"], "99")

    def test_platform_store_is_none_outside_test_force(self):
        agent_settings = _memory_settings(MEMORY_STORE="platform")
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.memory.store._force_in_memory_store", return_value=False
            ),
        ):
            self.assertIsNone(build_memory_store())


class MemoryLangmemToolTests(TestCase):
    def test_manage_semantic_tool_create_and_search(self):
        from ai_agent.memory.tools import build_memory_tools

        agent_settings = _memory_settings(MEMORY_MODE="hot")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            tools = {
                tool.name: tool
                for tool in build_memory_tools("dummy", store=store)
            }
            manage = tools["dummy_manage_semantic_memory"]
            search = tools["dummy_search_semantic_memory"]
            config = {"configurable": {"user_id": "15"}}
            created = manage.invoke(
                {
                    "action": "create",
                    "content": {
                        "subject": "user",
                        "predicate": "uses",
                        "object": "gold wallet",
                        "context": "when talking about bullion",
                    },
                },
                config=config,
            )
            self.assertTrue(created)
            found = search.invoke({"query": "gold wallet"}, config=config)
            self.assertTrue(str(found))
            profile = tools["dummy_manage_profile"]
            schema = profile.args_schema.model_json_schema()
            action = schema.get("properties", {}).get("action", {})
            enum_values = action.get("enum") or []
            if not enum_values:
                for option in action.get("anyOf", []):
                    enum_values.extend(option.get("enum") or [])
            self.assertIn("create", enum_values)
            self.assertIn("update", enum_values)
            self.assertNotIn("delete", enum_values)
            self.assertIn("dummy_manage_dummyaccount", tools)
            created_account = tools["dummy_manage_dummyaccount"].invoke(
                {
                    "action": "create",
                    "content": {
                        "bank_name": "Mellat",
                        "label": "primary card",
                        "use_for": "withdraw",
                    },
                },
                config=config,
            )
            self.assertTrue(created_account)

    def test_refuses_same_turn_episode_delete_but_allows_older_delete(self):
        from ai_agent.memory.tools import (
            SAME_TURN_DELETE_REFUSAL,
            build_memory_tools,
            reset_created_memory_ids,
        )

        agent_settings = _memory_settings(MEMORY_MODE="hot")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            tools = {
                tool.name: tool
                for tool in build_memory_tools("supervisor", store=store)
            }
            manage = tools["supervisor_manage_episode"]
            config = {"configurable": {"user_id": "22"}}
            reset_created_memory_ids()
            created = manage.invoke(
                {
                    "action": "create",
                    "content": {
                        "observation": "delivery specialist failed",
                        "thoughts": "I should record the failure.",
                        "action": "Tell the user to retry.",
                        "result": "failed: specialist stopped early; retry with less scope.",
                    },
                },
                config=config,
            )
            created_id = str(created).rsplit(" ", 1)[-1]
            refused = manage.invoke(
                {"action": "delete", "id": created_id},
                config=config,
            )
            self.assertEqual(
                refused, SAME_TURN_DELETE_REFUSAL.format(id=created_id)
            )
            ns = bind_namespace(
                memory_namespaces("supervisor")["episodes"], user_id="22"
            )
            self.assertTrue(any(item.key == created_id for item in store.search(ns)))

            reset_created_memory_ids()
            deleted = manage.invoke(
                {"action": "delete", "id": created_id},
                config=config,
            )
            self.assertIn("Deleted memory", str(deleted))
            self.assertFalse(any(item.key == created_id for item in store.search(ns)))

    def test_recall_middleware_resets_same_turn_ids(self):
        from ai_agent.memory.middleware import MemoryRecallMiddleware
        from ai_agent.memory.tools import (
            SAME_TURN_DELETE_REFUSAL,
            build_memory_tools,
        )

        agent_settings = _memory_settings(MEMORY_MODE="background")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            manage = {
                tool.name: tool
                for tool in build_memory_tools("supervisor", store=store)
            }["supervisor_manage_episode"]
            config = {"configurable": {"user_id": "22"}}
            created = manage.invoke(
                {
                    "action": "create",
                    "content": {
                        "observation": "first turn",
                        "thoughts": "I noted it.",
                        "action": "Saved an episode.",
                        "result": "ok",
                    },
                },
                config=config,
            )
            created_id = str(created).rsplit(" ", 1)[-1]
            MemoryRecallMiddleware("supervisor").before_agent({}, None)
            deleted = manage.invoke(
                {"action": "delete", "id": created_id},
                config=config,
            )
            self.assertIn("Deleted memory", str(deleted))
            self.assertNotIn("Refused", str(deleted))
            self.assertNotEqual(
                deleted, SAME_TURN_DELETE_REFUSAL.format(id=created_id)
            )

    def test_profile_create_twice_updates_the_same_document(self):
        from ai_agent.memory.tools import build_memory_tools

        agent_settings = _memory_settings(MEMORY_MODE="hot")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            tools = {
                tool.name: tool
                for tool in build_memory_tools("supervisor", store=store)
            }
            manage = tools["supervisor_manage_profile"]
            config = {"configurable": {"user_id": "15"}}
            manage.invoke(
                {
                    "action": "create",
                    "content": {"name": "محمد", "language": "fa"},
                },
                config=config,
            )
            manage.invoke(
                {
                    "action": "create",
                    "content": {
                        "name": "رضا",
                        "language": None,
                        "goals": None,
                    },
                },
                config=config,
            )
            ns = bind_namespace(
                memory_namespaces(SUPERVISOR_LAYER)["profile"], user_id="15"
            )
            items = list(store.search(ns))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].key, PROFILE_KEY)
            content = items[0].value["content"]
            self.assertEqual(content["name"], "رضا")
            self.assertEqual(content["language"], "fa")

    def test_profile_patch_drops_fields_not_on_schema(self):
        from ai_agent.memory.tools import build_memory_tools

        agent_settings = _memory_settings(MEMORY_MODE="hot")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(
                memory_namespaces(SUPERVISOR_LAYER)["profile"], user_id="15"
            )
            store.put(
                ns,
                PROFILE_KEY,
                {
                    "content": {
                        "name": "محمد",
                        "language": "fa",
                        "current_state": "portfolio chart dump",
                    }
                },
            )
            manage = {
                tool.name: tool
                for tool in build_memory_tools("supervisor", store=store)
            }["supervisor_manage_profile"]
            manage.invoke(
                {"action": "update", "content": {"name": "محمد"}},
                config={"configurable": {"user_id": "15"}},
            )
            content = store.get(ns, PROFILE_KEY).value["content"]
            self.assertEqual(content["name"], "محمد")
            self.assertEqual(content["language"], "fa")
            self.assertNotIn("current_state", content)

    def test_profile_create_collapses_duplicate_uuid_documents(self):
        from ai_agent.memory.tools import build_memory_tools

        agent_settings = _memory_settings(MEMORY_MODE="hot")
        with override_settings(AI_AGENT=agent_settings):
            store = make_test_runtime().store
            ns = bind_namespace(
                memory_namespaces(SUPERVISOR_LAYER)["profile"], user_id="15"
            )
            store.put(
                ns,
                "356bcba8-57e2-485c-81ad-bf688f375a5d",
                {"content": {"name": "محمد", "language": "fa"}},
            )
            store.put(
                ns,
                "d20ad741-5d34-490e-afcb-0d7ca2517b90",
                {"content": {"name": "رضا", "language": "fa"}},
            )
            manage = {
                tool.name: tool
                for tool in build_memory_tools("supervisor", store=store)
            }["supervisor_manage_profile"]
            manage.invoke(
                {"action": "create", "content": {"name": "رضا"}},
                config={"configurable": {"user_id": "15"}},
            )
            items = list(store.search(ns))
            self.assertEqual([item.key for item in items], [PROFILE_KEY])
            self.assertEqual(items[0].value["content"]["name"], "رضا")
            self.assertEqual(items[0].value["content"]["language"], "fa")


class MemoryAgentTests(TestCase):
    def test_memory_agent_uses_tools_and_recall(self):
        from ai_agent.memory import managers

        agent_settings = _memory_settings(MEMORY_MODE="background")
        with (
            override_settings(AI_AGENT=agent_settings),
            patch(
                "ai_agent.memory.managers.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            managers.build_layer_memory_agent("wallet")
        kwargs = mock_create.call_args.kwargs
        names = {tool.name for tool in kwargs["tools"]}
        self.assertTrue(memory_tool_names("wallet") <= names)
        self.assertIn("never a second document", kwargs["system_prompt"])
        self.assertIn("Playbooks are curator-owned", kwargs["system_prompt"])
        self.assertIn(
            "Never delete a document you created in this same turn",
            kwargs["system_prompt"],
        )
        self.assertIn("unique failure episode", kwargs["system_prompt"])
        kinds = [type(item).__name__ for item in kwargs["middleware"]]
        self.assertEqual(kinds, ["MemoryRecallMiddleware"])
