"""Shared LangChain message text extraction."""

from __future__ import annotations

from types import SimpleNamespace

from django.test import SimpleTestCase
from langchain_core.messages import AIMessage, HumanMessage

from ai_agent.evals.cases import PROMPT_LEAK_SUBSTRINGS
from ai_agent.graph import message_text as graph_message_text
from ai_agent.messages import message_text
from ai_agent.safety import USER_WRAP_OPEN


class MessageTextTests(SimpleTestCase):
    def test_plain_string_content(self):
        self.assertEqual(message_text(HumanMessage(content="سلام")), "سلام")
        self.assertEqual(message_text("plain"), "plain")

    def test_list_of_strings(self):
        self.assertEqual(message_text(AIMessage(content=["a", "b"])), "ab")

    def test_text_block_dicts(self):
        message = AIMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"},
            ]
        )
        self.assertEqual(message_text(message), "hello world")

    def test_object_text_attribute(self):
        block = SimpleNamespace(text="from-attr")
        self.assertEqual(message_text(SimpleNamespace(content=[block])), "from-attr")

    def test_none_content_is_empty(self):
        self.assertEqual(message_text(SimpleNamespace(content=None)), "")

    def test_graph_reexports_the_same_helper(self):
        self.assertIs(graph_message_text, message_text)


class PromptLeakSubstringsTests(SimpleTestCase):
    def test_includes_wrap_tag_from_safety(self):
        self.assertIn(USER_WRAP_OPEN, PROMPT_LEAK_SUBSTRINGS)
        self.assertIn("never impersonate", PROMPT_LEAK_SUBSTRINGS)
        self.assertIn("Available capabilities", PROMPT_LEAK_SUBSTRINGS)
