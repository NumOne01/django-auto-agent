"""Deterministic per-user, per-layer memory reconcile (no LLM)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from ai_agent.conf import get_agent_settings
from ai_agent.memory.namespaces import (
    PLAYBOOK_KEY,
    PROFILE_KEY,
    bind_namespace,
    memory_namespaces,
)
from ai_agent.memory.spec import resolve_memory_spec
from ai_agent.memory.tools import (
    _content_of,
    _delete_extra_profile_keys,
    _fill_missing_profile,
    _merge_profile_items,
)

_FAILURE_MARKERS = (
    "fail",
    "failed",
    "failure",
    "prevent",
    "error",
    "http 4",
    "http 5",
    "declined",
    "rejected",
)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class MemoryRecord:
    key: str
    content: dict[str, Any]
    created_at: Any = None
    updated_at: Any = None
    kind: str = ""


@dataclass
class ReconcileResult:
    dirty: bool = False
    needs_llm: bool = False
    needs_optimize: bool = False
    signals: list[str] = field(default_factory=list)
    dump: str = ""
    fact_count: int = 0
    episode_count: int = 0
    collection_counts: dict[str, int] = field(default_factory=dict)


def list_layer_episodes(
    store, user_id: str, layer: str, *, limit: int | None = None
) -> list[MemoryRecord]:
    """Newest-first episodes for one user and layer (no AgentMessage scan)."""
    if store is None or not user_id or not layer:
        return []
    namespace = bind_namespace(
        memory_namespaces(layer)["episodes"], user_id=str(user_id)
    )
    fetch_limit = _list_limit(get_agent_settings())
    if limit is not None:
        fetch_limit = max(1, min(int(limit), fetch_limit))
    records = _records(_search(store, namespace, fetch_limit))
    records.sort(key=lambda record: (_timestamp(record), record.key), reverse=True)
    return records


def is_failure_episode(record: MemoryRecord) -> bool:
    return _is_failure_episode(record)


def reconcile_layer(store, user_id: str, layer: str) -> ReconcileResult:
    """List, merge, delete, and cap one user's one memory layer."""
    if store is None or not user_id or not layer:
        return ReconcileResult()
    spec = resolve_memory_spec(layer)
    settings = get_agent_settings()
    namespaces = memory_namespaces(layer)
    bound = {
        name: bind_namespace(template, user_id=str(user_id))
        for name, template in namespaces.items()
    }
    list_limit = _list_limit(settings)
    dirty = False

    profile_items = _search(store, bound["profile"], list_limit)
    if _collapse_singleton(
        store, bound["profile"], profile_items, spec.profile, PROFILE_KEY
    ):
        dirty = True
        profile_items = _search(store, bound["profile"], list_limit)
    profile_content = _singleton_content(profile_items, PROFILE_KEY)

    playbook_items = _search(store, bound["playbook"], list_limit)
    if _collapse_singleton(
        store,
        bound["playbook"],
        playbook_items,
        None,
        PLAYBOOK_KEY,
        playbook_cap=settings.memory_playbook_rule_cap,
    ):
        dirty = True
        playbook_items = _search(store, bound["playbook"], list_limit)

    semantic_records = _records(_search(store, bound["semantic"], list_limit))
    classified: dict[str, list[MemoryRecord]] = {}
    for record in semantic_records:
        kind = _classify_semantic(record.content, spec)
        record.kind = kind
        classified.setdefault(kind, []).append(record)

    fact_name = spec.collections[0].__name__ if spec.collections else "SemanticFact"
    extra_names = {schema.__name__ for schema in spec.collections[1:]}
    fields_by_kind = _natural_key_fields_by_kind(spec)

    for kind, group in list(classified.items()):
        merged, kind_dirty = _merge_natural_keys(
            store, bound["semantic"], group, fields_by_kind
        )
        dirty = dirty or kind_dirty
        merged, text_dirty = _merge_normalized_dupes(store, bound["semantic"], merged)
        dirty = dirty or text_dirty
        classified[kind] = merged

    facts = classified.get(fact_name, [])
    kept_facts, overlap_dirty = _drop_profile_overlap(
        store, bound["semantic"], facts, profile_content, spec.profile
    )
    dirty = dirty or overlap_dirty
    classified[fact_name] = kept_facts

    kept_facts, cap_dirty = _apply_cap(
        store,
        bound["semantic"],
        classified.get(fact_name, []),
        settings.memory_fact_cap,
        rank=_fact_rank,
    )
    dirty = dirty or cap_dirty
    classified[fact_name] = kept_facts

    collection_counts: dict[str, int] = {}
    for kind in extra_names:
        group = classified.get(kind, [])
        group, kind_cap_dirty = _apply_cap(
            store,
            bound["semantic"],
            group,
            settings.memory_collection_cap,
            rank=_recency_rank,
        )
        dirty = dirty or kind_cap_dirty
        classified[kind] = group
        collection_counts[kind] = len(group)

    episode_records = _records(_search(store, bound["episodes"], list_limit))
    episode_records, ep_text_dirty = _merge_normalized_dupes(
        store, bound["episodes"], episode_records
    )
    dirty = dirty or ep_text_dirty
    episode_records, ep_cap_dirty = _apply_cap(
        store,
        bound["episodes"],
        episode_records,
        settings.memory_episode_cap,
        rank=_episode_rank,
    )
    dirty = dirty or ep_cap_dirty

    fact_count = len(classified.get(fact_name, []))
    episode_count = len(episode_records)
    over_cap = (
        fact_count > settings.memory_fact_cap
        or episode_count > settings.memory_episode_cap
        or any(count > settings.memory_collection_cap for count in collection_counts.values())
    )
    signals = [item.key for item in episode_records if _is_failure_episode(item)]
    dump = _format_dump(
        profile_content,
        classified,
        episode_records,
        _singleton_content(playbook_items, PLAYBOOK_KEY),
        fact_name,
    )
    return ReconcileResult(
        dirty=dirty,
        needs_llm=over_cap,
        needs_optimize=bool(signals),
        signals=signals,
        dump=dump,
        fact_count=fact_count,
        episode_count=episode_count,
        collection_counts=collection_counts,
    )


def list_layer_dump(store, user_id: str, layer: str) -> str:
    """Full remaining dump for the curator LLM (no writes)."""
    if store is None or not user_id or not layer:
        return ""
    spec = resolve_memory_spec(layer)
    settings = get_agent_settings()
    namespaces = memory_namespaces(layer)
    bound = {
        name: bind_namespace(template, user_id=str(user_id))
        for name, template in namespaces.items()
    }
    list_limit = _list_limit(settings)
    profile_items = _search(store, bound["profile"], list_limit)
    playbook_items = _search(store, bound["playbook"], list_limit)
    semantic_records = _records(_search(store, bound["semantic"], list_limit))
    classified: dict[str, list[MemoryRecord]] = {}
    for record in semantic_records:
        kind = _classify_semantic(record.content, spec)
        record.kind = kind
        classified.setdefault(kind, []).append(record)
    fact_name = spec.collections[0].__name__ if spec.collections else "SemanticFact"
    episodes = _records(_search(store, bound["episodes"], list_limit))
    return _format_dump(
        _singleton_content(profile_items, PROFILE_KEY),
        classified,
        episodes,
        _singleton_content(playbook_items, PLAYBOOK_KEY),
        fact_name,
    )


def _list_limit(settings) -> int:
    return max(
        settings.memory_fact_cap * 4,
        settings.memory_episode_cap * 4,
        settings.memory_collection_cap * 4,
        80,
    )


def _search(store, namespace: tuple[str, ...], limit: int) -> list:
    try:
        return list(store.search(namespace, limit=limit) or [])
    except Exception:
        return []


def _records(items) -> list[MemoryRecord]:
    records = []
    for item in items:
        if item is None:
            continue
        key = str(getattr(item, "key", "") or "")
        if not key:
            continue
        records.append(
            MemoryRecord(
                key=key,
                content=_content_of(item),
                created_at=getattr(item, "created_at", None),
                updated_at=getattr(item, "updated_at", None),
            )
        )
    return records


def _collapse_singleton(
    store,
    namespace: tuple[str, ...],
    items,
    schema: Optional[type[BaseModel]],
    canonical_key: str,
    *,
    playbook_cap: int | None = None,
) -> bool:
    records = [item for item in items if item is not None]
    if not records:
        return False
    keys = [str(getattr(item, "key", "") or "") for item in records]
    extras = [key for key in keys if key and key != canonical_key]
    if not extras and keys == [canonical_key]:
        if playbook_cap is None:
            return False
        content = _content_of(records[0])
        rules = content.get("rules") if isinstance(content, dict) else None
        if isinstance(rules, list) and len(rules) > playbook_cap:
            trimmed = dict(content)
            trimmed["rules"] = rules[:playbook_cap]
            store.put(namespace, canonical_key, {"content": trimmed})
            return True
        return False
    if schema is not None:
        merged = _merge_profile_items(records, None, schema)
        store.put(namespace, canonical_key, {"content": merged})
        _delete_extra_profile_keys(store, namespace, records)
        return True
    merged: dict[str, Any] = {}
    for item in records:
        merged = _fill_missing_profile(merged, _content_of(item))
    if playbook_cap is not None:
        rules = merged.get("rules") if isinstance(merged.get("rules"), list) else []
        merged["rules"] = rules[:playbook_cap]
    store.put(namespace, canonical_key, {"content": merged})
    for item in records:
        key = getattr(item, "key", None)
        if key and key != canonical_key:
            store.delete(namespace, key=key)
    return True


def _singleton_content(items, canonical_key: str) -> dict[str, Any]:
    for item in items:
        if getattr(item, "key", None) == canonical_key:
            return _content_of(item)
    if items:
        return _content_of(items[0])
    return {}


def _classify_semantic(content: dict[str, Any], spec) -> str:
    fact_name = spec.collections[0].__name__ if spec.collections else "SemanticFact"
    if _is_fact(content):
        return fact_name
    extras = spec.collections[1:]
    best_name = None
    best_score = -1
    best_required = 0
    for schema in extras:
        keys = _schema_natural_key(schema) or tuple(schema.model_fields)
        present = sum(1 for key in keys if _filled(content.get(key)))
        if present > best_score or (
            present == best_score and len(keys) > best_required
        ):
            if present == 0:
                continue
            best_name = schema.__name__
            best_score = present
            best_required = len(keys)
            if present == len(keys):
                return schema.__name__
    return best_name or fact_name


def _schema_natural_key(schema) -> tuple[str, ...]:
    fields = getattr(schema, "natural_key", ())
    if not fields:
        return ()
    if isinstance(fields, str):
        return (fields,)
    return tuple(str(item) for item in fields if str(item))


def _natural_key_fields_by_kind(spec) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    for schema in spec.collections:
        fields = _schema_natural_key(schema)
        if fields:
            mapping[schema.__name__] = fields
    return mapping


def _is_fact(content: dict[str, Any]) -> bool:
    return _filled(content.get("subject")) and _filled(content.get("predicate"))


def _filled(value) -> bool:
    return value not in (None, "")


def _natural_key(
    kind: str,
    content: dict[str, Any],
    fields_by_kind: dict[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    fields = fields_by_kind.get(kind)
    if not fields:
        return None
    parts = []
    for field_name in fields:
        value = content.get(field_name)
        if not _filled(value):
            return None
        parts.append(_normalize_token(value))
    return tuple(parts)


def _normalize_token(value) -> str:
    return _WHITESPACE_RE.sub(" ", str(value).strip().casefold())


def _normalize_text(content: dict[str, Any]) -> str:
    try:
        dumped = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        dumped = str(content)
    return _WHITESPACE_RE.sub(" ", dumped.strip().casefold())


def _merge_natural_keys(
    store, namespace: tuple[str, ...], records: list[MemoryRecord],
    fields_by_kind: dict[str, tuple[str, ...]],
) -> tuple[list[MemoryRecord], bool]:
    groups: dict[tuple[str, ...], list[MemoryRecord]] = {}
    unique: list[MemoryRecord] = []
    for record in records:
        key = _natural_key(record.kind, record.content, fields_by_kind)
        if key is None:
            unique.append(record)
            continue
        groups.setdefault(key, []).append(record)
    dirty = False
    kept: list[MemoryRecord] = list(unique)
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        winner = max(group, key=_fact_rank)
        merged = dict(winner.content)
        for item in sorted(group, key=_fact_rank, reverse=True):
            if item.key == winner.key:
                continue
            merged = _fill_missing_profile(merged, item.content)
        if merged != winner.content:
            winner.content = merged
            store.put(namespace, winner.key, {"content": merged})
        for item in group:
            if item.key == winner.key:
                continue
            store.delete(namespace, key=item.key)
        kept.append(winner)
        dirty = True
    return kept, dirty


def _merge_normalized_dupes(
    store, namespace: tuple[str, ...], records: list[MemoryRecord]
) -> tuple[list[MemoryRecord], bool]:
    groups: dict[str, list[MemoryRecord]] = {}
    for record in records:
        groups.setdefault(_normalize_text(record.content), []).append(record)
    dirty = False
    kept: list[MemoryRecord] = []
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        winner = max(group, key=_recency_rank)
        for item in group:
            if item.key == winner.key:
                continue
            store.delete(namespace, key=item.key)
        kept.append(winner)
        dirty = True
    return kept, dirty


def _drop_profile_overlap(
    store,
    namespace: tuple[str, ...],
    facts: list[MemoryRecord],
    profile: dict[str, Any],
    schema: type[BaseModel],
) -> tuple[list[MemoryRecord], bool]:
    field_names = {name.casefold() for name in schema.model_fields}
    profile_values = {
        _normalize_token(value)
        for value in profile.values()
        if _filled(value)
    }
    kept = []
    dirty = False
    for record in facts:
        predicate = _normalize_token(record.content.get("predicate") or "")
        obj = _normalize_token(record.content.get("object") or "")
        if predicate in field_names or (obj and obj in profile_values):
            store.delete(namespace, key=record.key)
            dirty = True
            continue
        kept.append(record)
    return kept, dirty


def _apply_cap(
    store,
    namespace: tuple[str, ...],
    records: list[MemoryRecord],
    cap: int,
    *,
    rank,
) -> tuple[list[MemoryRecord], bool]:
    if len(records) <= cap:
        return records, False
    ordered = sorted(records, key=rank, reverse=True)
    keep = ordered[:cap]
    drop = ordered[cap:]
    keep_keys = {item.key for item in keep}
    for item in drop:
        if item.key not in keep_keys:
            store.delete(namespace, key=item.key)
    return keep, True


def _timestamp(record: MemoryRecord):
    value = record.updated_at or record.created_at
    if isinstance(value, datetime):
        return value
    return datetime.min


def _recency_rank(record: MemoryRecord):
    return (_timestamp(record), record.key)


def _richness(content: dict[str, Any]) -> int:
    return sum(len(str(content.get(key) or "")) for key in ("object", "context", "result"))


def _fact_rank(record: MemoryRecord):
    return (_richness(record.content), _timestamp(record), record.key)


def _is_failure_episode(record: MemoryRecord) -> bool:
    result = str(record.content.get("result") or "").casefold()
    return any(marker in result for marker in _FAILURE_MARKERS)


def _episode_rank(record: MemoryRecord):
    return (1 if _is_failure_episode(record) else 0, _timestamp(record), record.key)


def _format_dump(
    profile: dict[str, Any],
    classified: dict[str, list[MemoryRecord]],
    episodes: list[MemoryRecord],
    playbook: dict[str, Any],
    fact_name: str,
) -> str:
    parts = []
    if profile:
        parts.append("Profile:\n" + json.dumps(profile, ensure_ascii=False, default=str))
    if playbook:
        parts.append(
            "Playbook:\n" + json.dumps(playbook, ensure_ascii=False, default=str)
        )
    facts = classified.get(fact_name) or []
    if facts:
        parts.append("Facts:")
        for record in facts:
            parts.append(f"- [{record.key}] {json.dumps(record.content, ensure_ascii=False, default=str)}")
    for kind, group in sorted(classified.items()):
        if kind == fact_name or not group:
            continue
        parts.append(f"{kind}:")
        for record in group:
            parts.append(
                f"- [{record.key}] {json.dumps(record.content, ensure_ascii=False, default=str)}"
            )
    if episodes:
        parts.append("Episodes:")
        for record in episodes:
            parts.append(
                f"- [{record.key}] {json.dumps(record.content, ensure_ascii=False, default=str)}"
            )
    return "\n".join(parts)
