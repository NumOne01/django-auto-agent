"""Admin views for AI agent threads and memories."""

from datetime import datetime, timezone
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone as django_timezone

from ai_agent.admin._helpers import safe_next
from ai_agent.admin._query import (
    MemoryRecord,
    PromptOverlayRecord,
    QueryPage,
    ThreadRecord,
    TranscriptMessage,
    _memory_filters,
    _prompt_filters,
    child_thread_id,
    decode_memory_id,
    decode_messages,
    encode_memory_id,
    layer_label,
    owner_from_payload,
    parse_memory_prefix,
    preview_text,
    prompt_overlay_from_memory,
)
from ai_agent.memory.namespaces import SUPERVISOR_LAYER
from ai_agent.memory.prompt_optimize import OverlayVersion

User = get_user_model()


class QueryHelperTests(TestCase):
    def test_parse_memory_prefix(self):
        user_id, layer, kind = parse_memory_prefix("memories.42.supervisor.semantic")
        self.assertEqual(user_id, "42")
        self.assertEqual(layer, "supervisor")
        self.assertEqual(kind, "semantic")

    def test_parse_global_prompt_prefix(self):
        user_id, layer, kind = parse_memory_prefix("prompts.global.wallet")
        self.assertEqual(user_id, "")
        self.assertEqual(layer, "wallet")
        self.assertEqual(kind, "prompt")

    def test_layer_label_supervisor_is_overall(self):
        self.assertEqual(layer_label(SUPERVISOR_LAYER), "Overall")
        self.assertEqual(layer_label("wallet"), "wallet")

    def test_owner_from_metadata(self):
        self.assertEqual(owner_from_payload({"owner": "9"}), "9")
        self.assertEqual(
            owner_from_payload({}, {"configurable": {"user_id": "4"}}),
            "4",
        )

    def test_child_thread_id_keeps_suffix_for_non_uuid(self):
        self.assertEqual(child_thread_id("t1", "wallet"), "t1::wallet")

    def test_child_thread_id_uuid_stays_uuid(self):
        parent = "11111111-1111-1111-1111-111111111111"
        child = child_thread_id(parent, "wallet")
        self.assertNotEqual(child, parent)
        self.assertEqual(child, child_thread_id(parent, "wallet"))
        self.assertNotEqual(child, child_thread_id(parent, "trading"))

    def test_decode_langchain_and_plain_messages(self):
        messages = decode_messages(
            [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": "hello"},
                },
                {"type": "ai", "content": "hi there"},
                {
                    "type": "tool",
                    "name": "call_wallet_agent",
                    "content": "balance 1",
                },
            ]
        )
        self.assertEqual([item.role for item in messages], ["human", "ai", "tool"])
        self.assertEqual(messages[0].content, "hello")
        self.assertEqual(messages[2].name, "call_wallet_agent")

    def test_preview_text_truncates(self):
        text = preview_text("x" * 500, limit=20)
        self.assertTrue(text.endswith("…"))
        self.assertEqual(len(text), 21)

    def test_memory_filters_bind_like_wildcards_for_psycopg3(self):
        where, params = _memory_filters(
            search="gold", user_id="7", layer="supervisor", kind="semantic"
        )
        self.assertEqual(where.count("%"), where.count("%s"))
        self.assertIn("ESCAPE", where)
        self.assertEqual(params[0], "memories.%")
        self.assertEqual(
            params[1:],
            ["7", "supervisor", "semantic", "%gold%", "%gold%", "%gold%"],
        )
        self.assertIn("value::text", where)

    def test_memory_filters_kinds_uses_any(self):
        where, params = _memory_filters(
            search="gold",
            user_id="7",
            layer="",
            kind="semantic",
            kinds=("profile", "semantic"),
        )
        self.assertIn("ANY(%s)", where)
        self.assertNotIn("split_part(prefix, '.', 4) = %s", where)
        self.assertEqual(params[0], "memories.%")
        self.assertEqual(params[1], "7")
        self.assertEqual(params[2], ["profile", "semantic"])
        self.assertEqual(params[3:], ["%gold%", "%gold%", "%gold%"])
        self.assertEqual(where.count("%"), where.count("%s"))

    def test_memory_filters_short_search_skips_value_text(self):
        where, params = _memory_filters(
            search="ab", user_id="", layer="", kind=""
        )
        self.assertNotIn("value::text", where)
        self.assertEqual(params[-2:], ["%ab%", "%ab%"])
        self.assertEqual(where.count("%"), where.count("%s"))

    def test_memory_filters_escape_user_wildcards(self):
        where, params = _memory_filters(
            search="a_b%", user_id="", layer="", kind=""
        )
        self.assertIn("ESCAPE", where)
        self.assertEqual(params[-3:], [r"%a\_b\%%", r"%a\_b\%%", r"%a\_b\%%"])

    def test_prompt_filters_escape_user_wildcards(self):
        where, params = _prompt_filters(
            search="100%", user_id="", layer="", scope=""
        )
        self.assertIn("ESCAPE", where)
        self.assertEqual(params[-3:], [r"%100\%%", r"%100\%%", r"%100\%%"])

    def test_prompt_filters_short_search_skips_value_text(self):
        where, params = _prompt_filters(
            search="x", user_id="", layer="", scope=""
        )
        self.assertNotIn("value::text", where)
        self.assertEqual(params[-2:], ["%x%", "%x%"])

    def test_memory_filters_prompt_kind_includes_global_prefix(self):
        where, params = _memory_filters(
            search="", user_id="", layer="", kind="prompt"
        )
        self.assertIn("prompts.global.%", params)
        self.assertEqual(where.count("%"), where.count("%s"))

    def test_prompt_filters_scope_global(self):
        where, params = _prompt_filters(
            search="", user_id="", layer="wallet", scope="global"
        )
        self.assertEqual(params[0], "prompts.global.%")
        self.assertEqual(params[1], "wallet")
        self.assertEqual(where.count("%"), where.count("%s"))

    def test_prompt_overlay_from_store_value_reads_history(self):
        item = MemoryRecord(
            prefix="prompts.global.wallet",
            key="default",
            layer="wallet",
            kind="prompt",
            value={
                "content": {
                    "text": "current overlay",
                    "previous": "old overlay",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                    "item_count": 4,
                    "history": [
                        {
                            "text": "old overlay",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                            "item_count": 2,
                        }
                    ],
                }
            },
        )
        overlay = prompt_overlay_from_memory(item)
        self.assertEqual(overlay.scope, "global")
        self.assertEqual(overlay.current, "current overlay")
        self.assertEqual(overlay.history[0].text, "old overlay")
        self.assertEqual(overlay.version_count, 2)

    def test_memory_layer_display(self):
        item = MemoryRecord(
            prefix="memories.1.supervisor.profile",
            key="default",
            layer=SUPERVISOR_LAYER,
            kind="profile",
        )
        self.assertEqual(item.layer_display, "Overall")

    def test_resolve_user_query_prefers_phone_over_numeric_string(self):
        from ai_agent.admin._query import resolve_user_query

        user = User.objects.create_user(
            username="lookup_user",
            password="pass",
        )
        self.assertEqual(resolve_user_query(user.username), str(user.pk))
        self.assertEqual(resolve_user_query(str(user.pk)), str(user.pk))

    def test_encode_decode_memory_id_roundtrip(self):
        prefix = "memories.1.supervisor.semantic"
        key = "fact::with spaces/and?chars"
        encoded = encode_memory_id(prefix, key)
        self.assertEqual(decode_memory_id(encoded), (prefix, key))
        self.assertNotIn("::", encoded.split("::", 1)[0])

    def test_decode_memory_id_rejects_missing_separator(self):
        self.assertEqual(decode_memory_id("no-separator"), ("", ""))

    def test_safe_next_keeps_query_on_same_path(self):
        fallback = "/admin/ai_agent/agentthread/"
        self.assertEqual(
            safe_next("?user=4&q=gold", fallback),
            "/admin/ai_agent/agentthread/?user=4&q=gold",
        )
        self.assertEqual(
            safe_next("https://evil.example/?x=1", fallback),
            fallback,
        )


class AdminSessionMixin:
    def _login_with_2fa(self, user):
        self.client.force_login(user)
        session = self.client.session
        session["_admin_2fa_verified"] = True
        session["_admin_2fa_verified_at"] = django_timezone.now().timestamp()
        session.save()


class AgentAdminViewTests(AdminSessionMixin, TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(
            "admin",
            "admin@example.com",
            "adminpass",
        )
        self.customer = User.objects.create_user(
            username="customer02",
            password="userpass",
            first_name="Ada",
            last_name="User",
        )
        self._login_with_2fa(self.admin)

    def test_thread_changelist_shows_not_configured_banner(self):
        url = reverse("admin:ai_agent_agentthread_changelist")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "LangGraph database is not configured")

    def test_memory_changelist_shows_not_configured_banner(self):
        url = reverse("admin:ai_agent_agentmemory_changelist")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "LangGraph database is not configured")

    def test_thread_changelist_renders_rows_and_user_filter(self):
        thread = ThreadRecord(
            thread_id="thread-1",
            owner_id=str(self.customer.pk),
            status="idle",
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            message_count=3,
            last_message_preview="Need gold price",
            user=self.customer,
        )
        with patch(
            "ai_agent.admin.threads.list_threads",
            return_value=QueryPage(items=[thread], total=1),
        ) as mocked:
            url = reverse("admin:ai_agent_agentthread_changelist")
            response = self.client.get(url, {"user": self.customer.username})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "thread-1")
        self.assertContains(response, "Need gold price")
        self.assertContains(response, "Ada User")
        mocked.assert_called_once()
        self.assertEqual(
            mocked.call_args.kwargs["owner_id"], str(self.customer.pk)
        )

    def test_thread_detail_renders_transcript(self):
        thread = ThreadRecord(
            thread_id="thread-9",
            owner_id=str(self.customer.pk),
            status="idle",
            user=self.customer,
            messages=[
                TranscriptMessage(role="human", content="hello"),
                TranscriptMessage(role="ai", content="welcome"),
            ],
        )
        with patch(
            "ai_agent.admin.threads.get_thread",
            return_value=(thread, None),
        ):
            url = reverse(
                "admin:ai_agent_agentthread_change", args=["thread-9"]
            )
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "hello")
        self.assertContains(response, "welcome")
        self.assertContains(response, "Transcript")

    def test_thread_detail_escapes_script_tags_in_transcript(self):
        payload = "<script>alert(1)</script>"
        thread = ThreadRecord(
            thread_id="thread-xss",
            owner_id=str(self.customer.pk),
            status="idle",
            user=self.customer,
            messages=[
                TranscriptMessage(role="human", content=payload),
                TranscriptMessage(role="tool", content=payload, name="wallet"),
            ],
        )
        with patch(
            "ai_agent.admin.threads.get_thread",
            return_value=(thread, None),
        ):
            url = reverse(
                "admin:ai_agent_agentthread_change", args=["thread-xss"]
            )
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "&lt;script&gt;alert(1)&lt;/script&gt;")
        self.assertNotContains(response, payload)

    def test_memory_changelist_escapes_script_tags_in_json(self):
        payload = "<script>alert(1)</script>"
        item = MemoryRecord(
            prefix="memories.1.supervisor.semantic",
            key="f1",
            user_id=str(self.customer.pk),
            layer=SUPERVISOR_LAYER,
            kind="semantic",
            preview=payload,
            value_json=payload,
            user=self.customer,
        )
        with patch(
            "ai_agent.admin.memories.list_memories",
            return_value=QueryPage(items=[item], total=1),
        ):
            url = reverse("admin:ai_agent_agentmemory_changelist")
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "&lt;script&gt;alert(1)&lt;/script&gt;")
        self.assertNotContains(response, payload)

    def test_memory_changelist_labels_supervisor_overall(self):
        item = MemoryRecord(
            prefix="memories.1.supervisor.semantic",
            key="f1",
            user_id=str(self.customer.pk),
            layer=SUPERVISOR_LAYER,
            kind="semantic",
            preview='{"n": "overall-fact"}',
            value_json='{\n  "n": "overall-fact"\n}',
            user=self.customer,
        )
        with patch(
            "ai_agent.admin.memories.list_memories",
            return_value=QueryPage(items=[item], total=1),
        ):
            url = reverse("admin:ai_agent_agentmemory_changelist")
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Overall")
        self.assertContains(response, "overall-fact")

    def test_user_memory_page_groups_layers(self):
        grouped = {
            SUPERVISOR_LAYER: {
                "profile": [
                    MemoryRecord(
                        prefix="memories.1.supervisor.profile",
                        key="default",
                        layer=SUPERVISOR_LAYER,
                        kind="profile",
                        preview='{"name": "Ada"}',
                        value_json='{"name": "Ada"}',
                    )
                ],
                "semantic": [],
                "episodes": [],
                "other": [],
            },
            "wallet": {
                "profile": [],
                "semantic": [
                    MemoryRecord(
                        prefix="memories.1.wallet.semantic",
                        key="f1",
                        layer="wallet",
                        kind="semantic",
                        preview="likes gold",
                        value_json='"likes gold"',
                    )
                ],
                "episodes": [],
                "other": [],
            },
        }
        with patch(
            "ai_agent.admin.memories.memories_grouped_for_user",
            return_value=(grouped, None),
        ), patch(
            "ai_agent.admin.memories.list_prompt_overlays",
            return_value=QueryPage(items=[], total=0),
        ):
            url = reverse(
                "admin:ai_agent_agentmemory_user", args=[self.customer.pk]
            )
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Overall")
        self.assertContains(response, "wallet")
        self.assertContains(response, "likes gold")
        self.assertContains(response, "Ada User")

    def test_memory_user_page_links_to_threads_and_user_change(self):
        grouped = {"dummy": {"profile": [], "semantic": [], "episodes": [], "other": []}}
        with patch(
            "ai_agent.admin.memories.memories_grouped_for_user",
            return_value=(grouped, None),
        ), patch(
            "ai_agent.admin.memories.list_prompt_overlays",
            return_value=QueryPage(items=[], total=0),
        ):
            url = reverse(
                "admin:ai_agent_agentmemory_user", args=[self.customer.pk]
            )
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        threads_url = reverse("admin:ai_agent_agentthread_changelist")
        user_url = reverse("admin:auth_user_change", args=[self.customer.pk])
        prompts_url = reverse("admin:ai_agent_agentprompt_changelist")
        self.assertContains(response, f"{threads_url}?user={self.customer.pk}")
        self.assertContains(response, user_url)
        self.assertContains(response, prompts_url)

    def test_thread_changelist_shows_batch_delete_controls(self):
        thread = ThreadRecord(
            thread_id="thread-1",
            owner_id=str(self.customer.pk),
            user=self.customer,
        )
        with patch(
            "ai_agent.admin.threads.list_threads",
            return_value=QueryPage(items=[thread], total=1),
        ):
            response = self.client.get(
                reverse("admin:ai_agent_agentthread_changelist")
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Delete selected threads")
        self.assertContains(response, 'name="selected"')
        self.assertContains(response, 'value="thread-1"')

    def test_thread_batch_delete_confirms_before_deleting(self):
        url = reverse("admin:ai_agent_agentthread_changelist")
        with patch("ai_agent.admin.threads.delete_threads") as mocked:
            response = self.client.post(
                url,
                {"action": "delete", "selected": ["thread-1", "thread-2"]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "You are about to delete 2 threads")
        self.assertContains(response, "thread-1")
        self.assertContains(response, 'name="confirm"')
        mocked.assert_not_called()

    def test_thread_batch_delete_confirmed_calls_query_layer(self):
        url = reverse("admin:ai_agent_agentthread_changelist")
        with patch(
            "ai_agent.admin.threads.delete_threads",
            return_value=(2, None),
        ) as mocked:
            response = self.client.post(
                url,
                {
                    "action": "delete",
                    "selected": ["thread-1", "thread-2"],
                    "confirm": "1",
                    "next": "?user=4",
                },
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], f"{url}?user=4")
        mocked.assert_called_once_with(["thread-1", "thread-2"])

    def test_thread_batch_delete_empty_selection_errors(self):
        url = reverse("admin:ai_agent_agentthread_changelist")
        with patch("ai_agent.admin.threads.delete_threads") as mocked:
            response = self.client.post(url, {"action": "delete"}, follow=True)
        mocked.assert_not_called()
        self.assertContains(response, "Select at least one thread.")

    def test_memory_changelist_shows_batch_delete_controls(self):
        item = MemoryRecord(
            prefix="memories.1.supervisor.semantic",
            key="f1",
            user_id=str(self.customer.pk),
            layer=SUPERVISOR_LAYER,
            kind="semantic",
            user=self.customer,
        )
        with patch(
            "ai_agent.admin.memories.list_memories",
            return_value=QueryPage(items=[item], total=1),
        ):
            response = self.client.get(
                reverse("admin:ai_agent_agentmemory_changelist")
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Delete selected memories")
        self.assertContains(response, 'name="selected"')
        self.assertContains(response, f'value="{item.item_id}"')

    def test_memory_batch_delete_confirms_before_deleting(self):
        item_id = encode_memory_id("memories.1.supervisor.semantic", "f1")
        url = reverse("admin:ai_agent_agentmemory_changelist")
        with patch("ai_agent.admin.memories.delete_memories") as mocked:
            response = self.client.post(
                url,
                {"action": "delete", "selected": [item_id]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "You are about to delete 1 memory")
        mocked.assert_not_called()

    def test_memory_batch_delete_confirmed_calls_query_layer(self):
        item_id = encode_memory_id("memories.1.wallet.semantic", "f1")
        url = reverse("admin:ai_agent_agentmemory_changelist")
        with patch(
            "ai_agent.admin.memories.delete_memories",
            return_value=(1, None),
        ) as mocked:
            response = self.client.post(
                url,
                {"action": "delete", "selected": [item_id], "confirm": "1"},
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], url)
        mocked.assert_called_once_with([item_id])

    def test_memory_batch_delete_empty_selection_errors(self):
        url = reverse("admin:ai_agent_agentmemory_changelist")
        with patch("ai_agent.admin.memories.delete_memories") as mocked:
            response = self.client.post(url, {"action": "delete"}, follow=True)
        mocked.assert_not_called()
        self.assertContains(response, "Select at least one memory.")

    def test_user_memory_page_batch_delete_confirmed(self):
        item_id = encode_memory_id("memories.1.supervisor.profile", "default")
        url = reverse(
            "admin:ai_agent_agentmemory_user", args=[self.customer.pk]
        )
        with patch(
            "ai_agent.admin.memories.delete_memories",
            return_value=(1, None),
        ) as mocked:
            response = self.client.post(
                url,
                {"action": "delete", "selected": [item_id], "confirm": "1"},
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], url)
        mocked.assert_called_once_with([item_id])

    def test_prompt_overlay_changelist_shows_global_and_user(self):
        global_item = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix="prompts.global.wallet",
                key="default",
                layer="wallet",
                kind="prompt",
            ),
            current="Shared wallet overlay",
            updated_at="2026-01-02",
            history=[OverlayVersion(text="older global", updated_at="2026-01-01")],
        )
        local_item = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix=f"memories.{self.customer.pk}.wallet.prompt",
                key="default",
                user_id=str(self.customer.pk),
                layer="wallet",
                kind="prompt",
                user=self.customer,
            ),
            current="Always confirm withdraw",
            updated_at="2026-01-03",
            history=[OverlayVersion(text="older local")],
        )

        def fake_list(**kwargs):
            if kwargs.get("scope") == "global":
                return QueryPage(items=[global_item], total=1)
            return QueryPage(items=[local_item], total=1)

        with patch(
            "ai_agent.admin.prompts.list_prompt_overlays",
            side_effect=fake_list,
        ):
            response = self.client.get(
                reverse("admin:ai_agent_agentprompt_changelist")
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Global overlays")
        self.assertContains(response, "Shared wallet overlay")
        self.assertContains(response, "Always confirm withdraw")
        self.assertContains(response, "customer02")

    def test_prompt_overlay_changelist_shows_batch_delete_controls(self):
        global_item = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix="prompts.global.wallet",
                key="default",
                layer="wallet",
                kind="prompt",
            ),
            current="Shared wallet overlay",
        )
        local_item = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix=f"memories.{self.customer.pk}.wallet.prompt",
                key="default",
                user_id=str(self.customer.pk),
                layer="wallet",
                kind="prompt",
                user=self.customer,
            ),
            current="Always confirm withdraw",
        )

        def fake_list(**kwargs):
            if kwargs.get("scope") == "global":
                return QueryPage(items=[global_item], total=1)
            return QueryPage(items=[local_item], total=1)

        with patch(
            "ai_agent.admin.prompts.list_prompt_overlays",
            side_effect=fake_list,
        ):
            response = self.client.get(
                reverse("admin:ai_agent_agentprompt_changelist")
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Delete selected overlays")
        self.assertContains(response, 'name="selected"')
        self.assertContains(response, f'value="{global_item.item_id}"')
        self.assertContains(response, f'value="{local_item.item_id}"')

    def test_prompt_overlay_batch_delete_confirms_before_deleting(self):
        item_id = encode_memory_id("prompts.global.wallet", "default")
        url = reverse("admin:ai_agent_agentprompt_changelist")
        with patch("ai_agent.admin.prompts.delete_prompt_overlays") as mocked:
            response = self.client.post(
                url,
                {"action": "delete", "selected": [item_id]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "You are about to delete 1 overlay")
        mocked.assert_not_called()

    def test_prompt_overlay_batch_delete_confirmed_calls_query_layer(self):
        item_id = encode_memory_id("prompts.global.wallet", "default")
        url = reverse("admin:ai_agent_agentprompt_changelist")
        with patch(
            "ai_agent.admin.prompts.delete_prompt_overlays",
            return_value=(1, None),
        ) as mocked:
            response = self.client.post(
                url,
                {
                    "action": "delete",
                    "selected": [item_id],
                    "confirm": "1",
                    "next": "?layer=wallet",
                },
            )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], f"{url}?layer=wallet")
        mocked.assert_called_once_with([item_id])

    def test_prompt_overlay_batch_delete_empty_selection_errors(self):
        url = reverse("admin:ai_agent_agentprompt_changelist")
        with patch("ai_agent.admin.prompts.delete_prompt_overlays") as mocked:
            response = self.client.post(url, {"action": "delete"}, follow=True)
        mocked.assert_not_called()
        self.assertContains(response, "Select at least one overlay.")

    def test_prompt_overlay_detail_shows_history(self):
        overlay = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix="prompts.global.wallet",
                key="default",
                layer="wallet",
                kind="prompt",
            ),
            current="current overlay text",
            updated_at="2026-01-02",
            history=[
                OverlayVersion(text="previous overlay text", updated_at="2026-01-01")
            ],
        )
        item_id = overlay.item_id
        with patch(
            "ai_agent.admin.prompts.get_prompt_overlay",
            return_value=(overlay, None),
        ):
            response = self.client.get(
                reverse("admin:ai_agent_agentprompt_change", args=[item_id])
            )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "current overlay text")
        self.assertContains(response, "previous overlay text")
        self.assertContains(response, "History")

    def test_user_memory_page_shows_prompt_overlays(self):
        grouped = {
            "wallet": {
                "profile": [],
                "semantic": [],
                "episodes": [],
                "playbook": [],
                "prompt": [],
                "other": [],
            }
        }
        local_item = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix=f"memories.{self.customer.pk}.wallet.prompt",
                key="default",
                user_id=str(self.customer.pk),
                layer="wallet",
                kind="prompt",
                user=self.customer,
            ),
            current="Always confirm withdraw",
            history=[OverlayVersion(text="older addendum")],
        )
        global_item = PromptOverlayRecord(
            memory=MemoryRecord(
                prefix="prompts.global.wallet",
                key="default",
                layer="wallet",
                kind="prompt",
            ),
            current="Shared wallet overlay",
            history=[],
        )

        def fake_list(**kwargs):
            if kwargs.get("scope") == "global":
                return QueryPage(items=[global_item], total=1)
            return QueryPage(items=[local_item], total=1)

        with patch(
            "ai_agent.admin.memories.memories_grouped_for_user",
            return_value=(grouped, None),
        ), patch(
            "ai_agent.admin.memories.list_prompt_overlays",
            side_effect=fake_list,
        ):
            url = reverse(
                "admin:ai_agent_agentmemory_user", args=[self.customer.pk]
            )
            response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prompt overlays")
        self.assertContains(response, "Always confirm withdraw")
        self.assertContains(response, "Shared wallet overlay")
        self.assertContains(response, "older addendum")
