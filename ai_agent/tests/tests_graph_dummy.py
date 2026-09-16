"""Graph tests against the library dummy host app."""

from django.test import TestCase

from dummy.models import Item

from ai_agent.tests.graph_test_utils import (
    AgentEndpointAssertions,
    find_endpoint,
    make_agent_user,
)


class DummyGraphTests(AgentEndpointAssertions, TestCase):
    def setUp(self):
        self.user = make_agent_user("graph_user")
        self.other = make_agent_user("graph_other")
        Item.objects.create(owner=self.user, name="Mine", secret="alpha")
        Item.objects.create(owner=self.other, name="Theirs", secret="beta")
        self.list_endpoint = find_endpoint(url_name="dummy_item_list", method="GET")
        self.create_endpoint = find_endpoint(
            url_name="dummy_item_create", method="POST"
        )

    def test_list_tool_is_registered_and_scoped(self):
        self.assert_tool_registered(self.list_endpoint)
        self.assert_args_model_builds(self.list_endpoint)
        result = self.assert_happy_tool(self.list_endpoint, self.user, contains="Mine")
        self.assertIn("alpha", result)
        self.assertNotIn("Theirs", result)

    def test_graph_forced_list_tool(self):
        self.assert_graph_forced_tool(
            self.list_endpoint, self.user, contains="Mine"
        )

    def test_create_requires_approval_and_creates_after_confirm(self):
        self.assertTrue(self.create_endpoint.confirm)
        self.assert_missing_required_rejected(self.create_endpoint, self.user)
        self.assert_mutation_requires_approval(
            self.create_endpoint, self.user, {"name": "New"}
        )
        self.assertEqual(Item.objects.filter(owner=self.user).count(), 1)
        self.assert_happy_tool(
            self.create_endpoint,
            self.user,
            {"name": "New", "status": "OPEN"},
            allowed_status=(201,),
            contains="New",
        )
        self.assertTrue(Item.objects.filter(owner=self.user, name="New").exists())
