"""Host ModelAgent specialists registered via AI_AGENT.EXTRA_AGENTS."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.core.exceptions import ImproperlyConfigured
from django.test import TestCase, override_settings

from ai_agent.agents import ModelAgent, agent_tool, resolve_model_agents
from ai_agent.context import AgentUserContext, get_current_user
from ai_agent.evals.scoring import routing_targets, tool_to_domain
from ai_agent.graph import build_domain_agent, build_supervisor, subagent_tool_name
from ai_agent.memory.namespaces import agent_memory_layers
from ai_agent.memory.spec import resolve_memory_spec
from ai_agent.tests.graph_test_utils import make_agent_user, offline_agent_settings


class _ResearchAgent(ModelAgent):
    name = "research"
    description = "Look up notes for the authenticated user."

    @agent_tool
    def search_notes(self, query: str) -> str:
        """Return a canned search result that includes the user id."""
        user = get_current_user()
        return f"{user.username}:{query}"


class _ConfirmAgent(ModelAgent):
    name = "confirm_demo"
    description = "Mutating helper that requires confirmation."

    @agent_tool(confirm=True)
    def delete_note(self, title: str) -> str:
        """Delete a note by title after confirmation."""
        return f"deleted:{title}"


class _EmptyNameAgent(ModelAgent):
    description = "Missing name."


class _DuplicateNameAgent(ModelAgent):
    name = "research"
    description = "Collides with research."


class _DummyNamedAgent(ModelAgent):
    name = "dummy"
    description = "Collides with the dummy app."

    @agent_tool
    def ping(self) -> str:
        """No-op."""
        return "pong"


class _MemoryAgent(ModelAgent):
    name = "research_memory"
    description = "Specialist with custom memory schemas."
    memory_profile = "dummy.memory_schemas.DummyProfile"
    memory_collections = (
        "ai_agent.memory.schemas.SemanticFact",
        "dummy.memory_schemas.DummyAccount",
    )


class ModelAgentResolutionTests(TestCase):
    def test_settings_dotted_path(self):
        payload = offline_agent_settings(
            EXTRA_AGENTS=["dummy.agents.CatalogSearchAgent"]
        )
        with override_settings(AI_AGENT=payload):
            agents = resolve_model_agents()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0].validated_name(), "catalog_search")
        names = {spec.name for spec in agents[0].tool_specs()}
        self.assertEqual(names, {"find_items"})

    def test_duplicate_names_raise(self):
        with self.assertRaises(ImproperlyConfigured) as caught:
            resolve_model_agents(extra_agents=[_ResearchAgent(), _DuplicateNameAgent()])
        self.assertIn("Duplicate ModelAgent name", str(caught.exception))

    def test_reserved_supervisor_name_raises(self):
        class SupervisorNamed(ModelAgent):
            name = "supervisor"
            description = "Reserved."

        with self.assertRaises(ImproperlyConfigured) as caught:
            resolve_model_agents(extra_agents=[SupervisorNamed()])
        self.assertIn("reserved", str(caught.exception))

    def test_missing_name_raises(self):
        with self.assertRaises(ImproperlyConfigured):
            resolve_model_agents(extra_agents=[_EmptyNameAgent()])

    def test_name_collides_with_exposed_app(self):
        with self.assertRaises(ImproperlyConfigured) as caught:
            resolve_model_agents(extra_agents=[_DummyNamedAgent()])
        self.assertIn("collides with an exposed Django app", str(caught.exception))

    def test_tool_name_collides_with_operation_id(self):
        from ai_agent.discovery import discover_endpoints

        op_id = next(
            item.operation_id
            for item in discover_endpoints()
            if item.app_label == "dummy"
        )

        class Colliding(ModelAgent):
            name = "other"
            description = "Tool name matches an API operationId."

            @agent_tool(name=op_id)
            def list_items(self) -> str:
                return "nope"

        with self.assertRaises(ImproperlyConfigured) as caught:
            resolve_model_agents(extra_agents=[Colliding()])
        self.assertIn("collides with API operationId", str(caught.exception))

    def test_bad_path_raises(self):
        with self.assertRaises(ImproperlyConfigured) as caught:
            resolve_model_agents(extra_agents=["ai_agent.tests.missing.Nope"])
        self.assertIn("EXTRA_AGENTS", str(caught.exception))


class ModelAgentGraphTests(TestCase):
    def test_supervisor_includes_model_agent(self):
        payload = offline_agent_settings()
        with (
            override_settings(AI_AGENT=payload),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor(stub_subagents=True, extra_agents=[_ResearchAgent()])
        names = [tool.name for tool in mock_create.call_args.kwargs["tools"]]
        self.assertIn(subagent_tool_name("dummy"), names)
        self.assertIn(subagent_tool_name("research"), names)
        prompt = mock_create.call_args.kwargs["system_prompt"]
        self.assertIn("call_research_agent", prompt)

    def test_build_domain_agent_for_model_agent(self):
        payload = offline_agent_settings()
        with (
            override_settings(AI_AGENT=payload),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_domain_agent("research", extra_agents=[_ResearchAgent()])
        mock_create.assert_called_once()
        names = [tool.name for tool in mock_create.call_args.kwargs["tools"]]
        self.assertEqual(names, ["search_notes"])

    def test_model_agent_only_graph(self):
        payload = offline_agent_settings()
        with (
            override_settings(AI_AGENT=payload),
            patch(
                "ai_agent.graph.create_agent", return_value=MagicMock()
            ) as mock_create,
        ):
            build_supervisor(
                stub_subagents=True,
                endpoints=[],
                extra_agents=[_ResearchAgent()],
            )
        names = [tool.name for tool in mock_create.call_args.kwargs["tools"]]
        self.assertEqual(names, [subagent_tool_name("research")])

    def test_empty_specialists_raise(self):
        with self.assertRaises(RuntimeError) as caught:
            build_supervisor(endpoints=[], extra_agents=[])
        self.assertIn("No specialists", str(caught.exception))


class ModelAgentToolTests(TestCase):
    def test_tool_uses_current_user(self):
        user = make_agent_user("researcher")
        agent = _ResearchAgent()
        tool = agent.build_tools()[0]
        with AgentUserContext(user):
            result = tool.invoke({"query": "hello"})
        self.assertEqual(result, "researcher:hello")

    def test_confirm_true_interrupts_before_body(self):
        user = make_agent_user("confirmer")
        agent = _ConfirmAgent()
        tool = agent.build_tools()[0]
        with (
            patch("ai_agent.agents.interrupt", return_value=False) as mock_interrupt,
            AgentUserContext(user),
        ):
            result = tool.invoke({"title": "secret"})
        mock_interrupt.assert_called_once()
        self.assertIn("declined", result.lower())

    def test_confirm_true_runs_after_approval(self):
        user = make_agent_user("confirmer")
        agent = _ConfirmAgent()
        tool = agent.build_tools()[0]
        with (
            patch("ai_agent.agents.interrupt", return_value={"approved": True}),
            AgentUserContext(user),
        ):
            result = tool.invoke({"title": "secret"})
        self.assertEqual(result, "deleted:secret")

    def test_catalog_search_uses_orm(self):
        from dummy.agents import CatalogSearchAgent
        from dummy.models import Item

        user = make_agent_user("shopper")
        Item.objects.create(owner=user, name="Alpha Widget")
        Item.objects.create(owner=user, name="Beta Gadget")
        agent = CatalogSearchAgent()
        tool = agent.build_tools()[0]
        with AgentUserContext(user):
            result = tool.invoke({"query": "widget"})
        self.assertEqual(result, "Alpha Widget")


class ModelAgentMemoryAndEvalTests(TestCase):
    def test_memory_layers_include_model_agent(self):
        payload = offline_agent_settings(EXTRA_AGENTS=[_ResearchAgent])
        with override_settings(AI_AGENT=payload):
            layers = agent_memory_layers()
        self.assertIn("research", layers)
        self.assertIn("dummy", layers)
        self.assertIn("supervisor", layers)

    def test_memory_spec_uses_model_agent_schemas(self):
        from dummy.memory_schemas import DummyAccount, DummyProfile
        from ai_agent.memory.schemas import SemanticFact

        payload = offline_agent_settings(EXTRA_AGENTS=[_MemoryAgent])
        with override_settings(AI_AGENT=payload):
            spec = resolve_memory_spec("research_memory")
        self.assertIs(spec.profile, DummyProfile)
        self.assertEqual(spec.collections, (SemanticFact, DummyAccount))

    def test_routing_targets_and_tool_to_domain(self):
        payload = offline_agent_settings(EXTRA_AGENTS=[_ResearchAgent])
        with override_settings(AI_AGENT=payload):
            self.assertEqual(
                routing_targets("research"), {subagent_tool_name("research")}
            )
            mapping = tool_to_domain()
        self.assertEqual(mapping[subagent_tool_name("research")], "research")
        self.assertEqual(mapping["search_notes"], "research")
