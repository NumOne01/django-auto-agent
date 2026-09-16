"""Owner-scoped access to user-facing long-term memories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from ai_agent.admin._query import (
    MemoryRecord,
    delete_memories,
    encode_memory_id,
    get_memory,
    list_memories,
)

USER_MEMORY_KINDS = frozenset({"profile", "semantic"})
_LIST_LIMIT = 500


@dataclass
class UserMemory:
    layer: str
    kind: str
    key: str
    content: Any
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def id(self) -> str:
        return public_memory_id(self.layer, self.kind, self.key)


def public_memory_id(layer: str, kind: str, key: str) -> str:
    return f"{layer}/{kind}/{key}"


def memory_prefix(user_id: str, layer: str, kind: str) -> str:
    return f"memories.{user_id}.{layer}.{kind}"


def list_for_user(
    user_id: str, *, layer: str = ""
) -> tuple[Optional[list[UserMemory]], Optional[str]]:
    page = list_memories(
        user_id=str(user_id),
        layer=(layer or "").strip(),
        kinds=tuple(sorted(USER_MEMORY_KINDS)),
        limit=_LIST_LIMIT,
        offset=0,
    )
    if page.error:
        return None, page.error
    items = [
        _to_user_memory(item)
        for item in page.items
        if _owned_user_facing(item, user_id)
    ]
    return items, None


def get_for_user(
    user_id: str, *, layer: str, kind: str, key: str
) -> tuple[Optional[UserMemory], Optional[str]]:
    expected_prefix = _owned_prefix(user_id, layer, kind, key)
    if expected_prefix is None:
        return None, None
    record, error = get_memory(encode_memory_id(expected_prefix, key))
    if error:
        return None, error
    if record is None or not _owned_user_facing(
        record, user_id, expected_prefix=expected_prefix
    ):
        return None, None
    return _to_user_memory(record), None


def delete_for_user(
    user_id: str, *, layer: str, kind: str, key: str
) -> tuple[bool, Optional[str]]:
    item, error = get_for_user(user_id, layer=layer, kind=kind, key=key)
    if error:
        return False, error
    if item is None:
        return False, None
    prefix = memory_prefix(str(user_id), layer, kind)
    deleted, delete_error = delete_memories([encode_memory_id(prefix, key)])
    if delete_error:
        return False, delete_error
    return deleted > 0, None


def _owned_prefix(
    user_id: str, layer: str, kind: str, key: str
) -> Optional[str]:
    layer = (layer or "").strip()
    kind = (kind or "").strip()
    key = (key or "").strip()
    if not layer or not key or kind not in USER_MEMORY_KINDS:
        return None
    return memory_prefix(str(user_id), layer, kind)


def _owned_user_facing(
    record: MemoryRecord,
    user_id: str,
    *,
    expected_prefix: str | None = None,
) -> bool:
    if str(record.user_id) != str(user_id):
        return False
    if record.kind not in USER_MEMORY_KINDS:
        return False
    if record.scope == "global":
        return False
    prefix = expected_prefix or memory_prefix(
        str(user_id), record.layer, record.kind
    )
    return record.prefix == prefix


def _to_user_memory(record: MemoryRecord) -> UserMemory:
    return UserMemory(
        layer=record.layer,
        kind=record.kind,
        key=record.key,
        content=_unwrap_content(record.value),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _unwrap_content(value: Any) -> Any:
    if isinstance(value, dict) and "content" in value:
        payload = value.get("content")
        if isinstance(payload, dict):
            return payload
    return value if value is not None else {}
