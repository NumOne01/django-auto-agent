"""Authenticated list/retrieve/delete APIs for user-facing agent memories."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from ai_agent.admin._query import (
    NOT_CONFIGURED,
    UNAVAILABLE,
    MemoryRecord,
    QueryPage,
    decode_memory_id,
    encode_memory_id,
)

User = get_user_model()


def _record(
    *,
    user_id: str,
    layer: str = "wallet",
    kind: str = "semantic",
    key: str = "k1",
    value=None,
):
    prefix = f"memories.{user_id}.{layer}.{kind}"
    if value is None:
        value = {
            "content": {
                "subject": "user",
                "predicate": "prefers",
                "object": "gold",
            }
        }
    return MemoryRecord(
        prefix=prefix,
        key=key,
        user_id=str(user_id),
        layer=layer,
        kind=kind,
        value=value,
    )


class AgentMemoryAPITests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="mem_user", password="pass12345"
        )
        self.other = User.objects.create_user(
            username="mem_other", password="pass12345"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_requires_auth(self):
        guest = APIClient()
        response = guest.get("/api/agent/memories/")
        self.assertEqual(response.status_code, 401)
        response = guest.get("/api/agent/memories/wallet/semantic/k1/")
        self.assertEqual(response.status_code, 401)
        response = guest.delete("/api/agent/memories/wallet/semantic/k1/")
        self.assertEqual(response.status_code, 401)

    def test_lists_own_profile_and_semantic_memories(self):
        own_semantic = _record(user_id=str(self.user.pk), key="fact-1")
        own_profile = _record(
            user_id=str(self.user.pk),
            layer="supervisor",
            kind="profile",
            key="default",
            value={"content": {"goals": "save more"}},
        )
        own_episode = _record(
            user_id=str(self.user.pk), kind="episodes", key="ep-1"
        )
        other_fact = _record(user_id=str(self.other.pk), key="other-fact")
        with patch("ai_agent.memory.user_memories.list_memories") as mocked:
            mocked.return_value = QueryPage(
                items=[own_semantic, own_profile, own_episode, other_fact],
                total=4,
            )
            response = self.client.get("/api/agent/memories/")
        self.assertEqual(response.status_code, 200)
        memories = response.json()["memories"]
        ids = {item["id"] for item in memories}
        self.assertEqual(
            ids,
            {
                "wallet/semantic/fact-1",
                "supervisor/profile/default",
            },
        )
        by_id = {item["id"]: item for item in memories}
        self.assertEqual(
            by_id["wallet/semantic/fact-1"]["content"],
            {"subject": "user", "predicate": "prefers", "object": "gold"},
        )
        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["user_id"], str(self.user.pk))
        self.assertEqual(set(kwargs["kinds"]), {"profile", "semantic"})
        self.assertEqual(kwargs["layer"], "")

    def test_list_passes_layer_filter(self):
        with patch("ai_agent.memory.user_memories.list_memories") as mocked:
            mocked.return_value = QueryPage(items=[], total=0)
            response = self.client.get("/api/agent/memories/?layer=wallet")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["memories"], [])
        self.assertEqual(mocked.call_args.kwargs["layer"], "wallet")

    def test_retrieves_own_memory(self):
        record = _record(user_id=str(self.user.pk), key="fact-1")
        with patch("ai_agent.memory.user_memories.get_memory") as get_mem:
            get_mem.return_value = (record, None)
            response = self.client.get(
                "/api/agent/memories/wallet/semantic/fact-1/"
            )
        self.assertEqual(response.status_code, 200)
        memory = response.json()["memory"]
        self.assertEqual(memory["id"], "wallet/semantic/fact-1")
        self.assertEqual(memory["layer"], "wallet")
        self.assertEqual(memory["kind"], "semantic")
        self.assertEqual(
            memory["content"]["object"],
            "gold",
        )
        prefix, key = decode_memory_id(get_mem.call_args[0][0])
        self.assertEqual(prefix, f"memories.{self.user.pk}.wallet.semantic")
        self.assertEqual(key, "fact-1")

    def test_deletes_own_memory(self):
        record = _record(user_id=str(self.user.pk), key="fact-1")
        with (
            patch("ai_agent.memory.user_memories.get_memory") as get_mem,
            patch("ai_agent.memory.user_memories.delete_memories") as delete_mem,
        ):
            get_mem.return_value = (record, None)
            delete_mem.return_value = (1, None)
            response = self.client.delete(
                "/api/agent/memories/wallet/semantic/fact-1/"
            )
        self.assertEqual(response.status_code, 204)
        expected_id = encode_memory_id(
            f"memories.{self.user.pk}.wallet.semantic", "fact-1"
        )
        delete_mem.assert_called_once_with([expected_id])

    def test_hides_other_users_memories_on_retrieve_and_delete(self):
        with (
            patch("ai_agent.memory.user_memories.get_memory") as get_mem,
            patch("ai_agent.memory.user_memories.delete_memories") as delete_mem,
        ):
            get_mem.return_value = (None, None)
            get_response = self.client.get(
                "/api/agent/memories/wallet/semantic/secret/"
            )
            delete_response = self.client.delete(
                "/api/agent/memories/wallet/semantic/secret/"
            )
        self.assertEqual(get_response.status_code, 404)
        self.assertEqual(delete_response.status_code, 404)
        delete_mem.assert_not_called()
        for call in get_mem.call_args_list:
            prefix, key = decode_memory_id(call[0][0])
            self.assertEqual(prefix, f"memories.{self.user.pk}.wallet.semantic")
            self.assertNotIn(str(self.other.pk), prefix)
            self.assertEqual(key, "secret")

    def test_rejects_foreign_record_even_if_store_returns_it(self):
        foreign = _record(user_id=str(self.other.pk), key="fact-1")
        with (
            patch("ai_agent.memory.user_memories.get_memory") as get_mem,
            patch("ai_agent.memory.user_memories.delete_memories") as delete_mem,
        ):
            get_mem.return_value = (foreign, None)
            response = self.client.delete(
                "/api/agent/memories/wallet/semantic/fact-1/"
            )
        self.assertEqual(response.status_code, 404)
        delete_mem.assert_not_called()

    def test_rejects_episode_kind(self):
        with (
            patch("ai_agent.memory.user_memories.get_memory") as get_mem,
            patch("ai_agent.memory.user_memories.delete_memories") as delete_mem,
        ):
            get_response = self.client.get(
                "/api/agent/memories/wallet/episodes/ep-1/"
            )
            delete_response = self.client.delete(
                "/api/agent/memories/wallet/episodes/ep-1/"
            )
        self.assertEqual(get_response.status_code, 404)
        self.assertEqual(delete_response.status_code, 404)
        get_mem.assert_not_called()
        delete_mem.assert_not_called()

    def test_list_unavailable_returns_503(self):
        with patch("ai_agent.memory.user_memories.list_memories") as mocked:
            mocked.return_value = QueryPage(
                items=[], total=0, error=NOT_CONFIGURED
            )
            response = self.client.get("/api/agent/memories/")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Agent memory store is unavailable",
        )

    def test_retrieve_unavailable_returns_503(self):
        with patch("ai_agent.memory.user_memories.get_memory") as get_mem:
            get_mem.return_value = (None, UNAVAILABLE)
            response = self.client.get(
                "/api/agent/memories/wallet/semantic/fact-1/"
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Agent memory store is unavailable",
        )
