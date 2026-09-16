"""LangMem manage/search tools for the hot-path conversation agents."""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from ai_agent.memory.namespaces import PLAYBOOK_KEY, PROFILE_KEY, memory_namespaces
from ai_agent.memory.schemas import LayerPlaybook
from ai_agent.memory.spec import resolve_memory_spec

_CREATED_MEMORY_IDS: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "ai_agent_created_memory_ids", default=None
)
_CREATED_MEMORY_RE = re.compile(
    r"\b(?:created|updated) memory ([0-9a-fA-F-]{8,})\b", re.I
)
logger = logging.getLogger(__name__)
SAME_TURN_DELETE_REFUSAL = (
    "Refused to delete {id}: it was created in this same turn. "
    "Keep it, or update it. Only delete a different older duplicate."
)


def build_memory_tools(layer: str, *, store=None) -> list:
    from langmem import create_search_memory_tool

    spec = resolve_memory_spec(layer)
    namespaces = memory_namespaces(layer)
    semantic_schema = spec.collections[0]
    extra = {}
    if store is not None:
        extra["store"] = store
    tools = [
        _create_manage_profile_tool(
            layer,
            spec.profile,
            store=store,
            name=_tool_name(layer, "manage_profile"),
        ),
        _manage_memory_tool(
            namespace=namespaces["semantic"],
            schema=semantic_schema,
            actions_permitted=("create", "update", "delete"),
            name=_tool_name(layer, "manage_semantic_memory"),
            instructions=(
                "Insert, update, or delete durable facts. Search first. "
                "One fact per subject+predicate; reuse that predicate on update. "
                "Do not duplicate what already lives on the profile document. "
                "Skip one-off requests. Never delete a fact you created in this turn."
            ),
            **extra,
        ),
        create_search_memory_tool(
            namespace=namespaces["semantic"],
            name=_tool_name(layer, "search_semantic_memory"),
            instructions="Search durable facts for this agent before creating a new one.",
            **extra,
        ),
        _manage_memory_tool(
            namespace=namespaces["episodes"],
            schema=spec.episode,
            actions_permitted=("create", "update", "delete"),
            name=_tool_name(layer, "manage_episode"),
            instructions=(
                "Save a reusable problem-solving procedure: observation, "
                "thoughts, action, and result. If it failed, include that it "
                "failed, why, and how to prevent it next time. Skip one-off "
                "tool logs. Search before create. Never delete an episode you "
                "created in this turn, even if search now returns it."
            ),
            **extra,
        ),
        create_search_memory_tool(
            namespace=namespaces["episodes"],
            name=_tool_name(layer, "search_episodes"),
            instructions=(
                "Search past episodes that resemble the current task "
                "before creating a new one."
            ),
            **extra,
        ),
    ]
    for schema in spec.collections[1:]:
        blurb = (schema.__doc__ or "").strip().splitlines()[0] if schema.__doc__ else ""
        tools.append(
            _manage_memory_tool(
                namespace=namespaces["semantic"],
                schema=schema,
                actions_permitted=("create", "update", "delete"),
                name=_tool_name(layer, f"manage_{schema.__name__.lower()}"),
                instructions=(
                    f"Insert, update, or delete {schema.__name__} memories. "
                    f"{blurb} Search first. Prefer update over insert. "
                    "Do not duplicate fields already on the profile."
                ),
                **extra,
            )
        )
    return tools


def build_curator_tools(layer: str, *, store=None, optimize: bool = False) -> list:
    """Manage tools for the curator graph. No search; no semantic creates."""
    from langmem import create_manage_memory_tool

    spec = resolve_memory_spec(layer)
    namespaces = memory_namespaces(layer)
    extra = {}
    if store is not None:
        extra["store"] = store
    tools = [
        _create_manage_profile_tool(
            layer,
            spec.profile,
            store=store,
            name=_tool_name(layer, "manage_profile"),
        ),
        _create_manage_playbook_tool(
            layer,
            store=store,
            name=_tool_name(layer, "manage_playbook"),
        ),
        create_manage_memory_tool(
            namespace=namespaces["episodes"],
            schema=spec.episode,
            actions_permitted=("update", "delete"),
            name=_tool_name(layer, "manage_episode"),
            instructions=(
                "Update or delete an existing episode. Pass the memory id. "
                "Do not create a new episode."
            ),
            **extra,
        ),
        _create_list_layer_tool(layer, store=store),
    ]
    if optimize:
        return tools
    tools.insert(
        2,
        create_manage_memory_tool(
            namespace=namespaces["semantic"],
            schema=spec.collections[0],
            actions_permitted=("update", "delete"),
            name=_tool_name(layer, "manage_semantic_memory"),
            instructions=(
                "Update or delete an existing fact. Pass the memory id. "
                "Do not create a new fact."
            ),
            **extra,
        ),
    )
    for schema in spec.collections[1:]:
        blurb = (schema.__doc__ or "").strip().splitlines()[0] if schema.__doc__ else ""
        tools.append(
            create_manage_memory_tool(
                namespace=namespaces["semantic"],
                schema=schema,
                actions_permitted=("update", "delete"),
                name=_tool_name(layer, f"manage_{schema.__name__.lower()}"),
                instructions=(
                    f"Update or delete an existing {schema.__name__}. {blurb} "
                    "Pass the memory id. Do not create a new document."
                ),
                **extra,
            )
        )
    return tools


def curator_tool_names(layer: str) -> set[str]:
    names = {
        _tool_name(layer, "manage_profile"),
        _tool_name(layer, "manage_playbook"),
        _tool_name(layer, "manage_semantic_memory"),
        _tool_name(layer, "manage_episode"),
        _tool_name(layer, "list_layer_memories"),
    }
    spec = resolve_memory_spec(layer)
    for schema in spec.collections[1:]:
        names.add(_tool_name(layer, f"manage_{schema.__name__.lower()}"))
    return names


def memory_tool_names(layer: str) -> set[str]:
    names = {
        _tool_name(layer, "manage_profile"),
        _tool_name(layer, "manage_semantic_memory"),
        _tool_name(layer, "search_semantic_memory"),
        _tool_name(layer, "manage_episode"),
        _tool_name(layer, "search_episodes"),
    }
    spec = resolve_memory_spec(layer)
    for schema in spec.collections[1:]:
        names.add(_tool_name(layer, f"manage_{schema.__name__.lower()}"))
    return names


def _tool_name(layer: str, suffix: str) -> str:
    prefix = "memory" if layer.startswith("{") else layer
    return f"{prefix}_{suffix}"


def reset_created_memory_ids() -> None:
    """Start a new manage-memory turn so later deletes of older rows still work."""
    _CREATED_MEMORY_IDS.set(set())


def _created_memory_ids() -> set[str]:
    current = _CREATED_MEMORY_IDS.get()
    if current is None:
        current = set()
        _CREATED_MEMORY_IDS.set(current)
    return current


def _manage_memory_tool(**kwargs):
    from langmem import create_manage_memory_tool

    return _guard_same_turn_delete(create_manage_memory_tool(**kwargs))


def _guard_same_turn_delete(tool: StructuredTool) -> StructuredTool:
    inner = tool.func
    inner_coro = tool.coroutine

    def func(*args, **kwargs):
        refusal = _same_turn_delete_refusal(*args, **kwargs)
        if refusal:
            return refusal
        result = inner(*args, **kwargs)
        _note_created_memory(result)
        return result

    async def afunc(*args, **kwargs):
        refusal = _same_turn_delete_refusal(*args, **kwargs)
        if refusal:
            return refusal
        result = await inner_coro(*args, **kwargs)
        _note_created_memory(result)
        return result

    return StructuredTool.from_function(
        func,
        coroutine=afunc if inner_coro is not None else None,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _same_turn_delete_refusal(*args, **kwargs) -> str | None:
    action = kwargs.get("action")
    memory_id = kwargs.get("id")
    if action is None and len(args) >= 2:
        action = args[1]
    if str(action or "").lower() != "delete" or memory_id in (None, ""):
        return None
    key = str(memory_id).lower()
    if key in _created_memory_ids():
        return SAME_TURN_DELETE_REFUSAL.format(id=memory_id)
    return None


def _note_created_memory(result) -> None:
    text = result if isinstance(result, str) else str(result)
    match = _CREATED_MEMORY_RE.search(text)
    if match:
        _created_memory_ids().add(match.group(1).lower())


def _create_manage_profile_tool(layer: str, schema: type[BaseModel], *, store, name: str):
    from langmem.utils import NamespaceTemplate

    namespacer = NamespaceTemplate(memory_namespaces(layer)["profile"])
    args_schema = create_model(
        f"{name}_args",
        content=(Optional[schema], None),
        action=(Literal["create", "update"], "create"),
    )
    description = (
        "Patch this layer's singleton profile. Create and update both write the "
        "same document; a second profile is never created. Do not pass an id. "
        "Write only standing schema fields. Do not summarize the current turn, "
        "date range, chart, or in-progress task into the profile."
    )

    def manage_profile(content=None, action: str = "create"):
        resolved = _profile_store(store)
        namespace = namespacer()
        items = _list_namespace(resolved, namespace)
        merged = _merge_profile_items(items, content, schema)
        resolved.put(namespace, PROFILE_KEY, {"content": merged})
        _delete_extra_profile_keys(resolved, namespace, items)
        existed = any(getattr(item, "key", None) == PROFILE_KEY for item in items) or bool(
            items
        )
        verb = "updated" if existed else "created"
        return f"{verb} memory {PROFILE_KEY}"

    async def amanage_profile(content=None, action: str = "create"):
        resolved = _profile_store(store)
        namespace = namespacer()
        items = await _alist_namespace(resolved, namespace)
        merged = _merge_profile_items(items, content, schema)
        aput = getattr(resolved, "aput", None)
        if aput is not None:
            await aput(namespace, PROFILE_KEY, {"content": merged})
        else:
            resolved.put(namespace, PROFILE_KEY, {"content": merged})
        await _adelete_extra_profile_keys(resolved, namespace, items)
        existed = any(getattr(item, "key", None) == PROFILE_KEY for item in items) or bool(
            items
        )
        verb = "updated" if existed else "created"
        return f"{verb} memory {PROFILE_KEY}"

    return StructuredTool.from_function(
        manage_profile,
        coroutine=amanage_profile,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def _create_manage_playbook_tool(layer: str, *, store, name: str):
    from langmem.utils import NamespaceTemplate

    namespacer = NamespaceTemplate(memory_namespaces(layer)["playbook"])
    args_schema = create_model(
        f"{name}_args",
        content=(Optional[LayerPlaybook], None),
        action=(Literal["create", "update"], "create"),
    )
    description = (
        "Patch this layer's singleton local playbook. Create and update both "
        "write the same document. Do not pass an id. Write short standing "
        "rules only. Do not store the current turn or live balances."
    )

    def manage_playbook(content=None, action: str = "create"):
        resolved = _profile_store(store)
        namespace = namespacer()
        items = _list_namespace(resolved, namespace)
        merged = _merge_playbook_items(items, content)
        resolved.put(namespace, PLAYBOOK_KEY, {"content": merged})
        _delete_extra_profile_keys(resolved, namespace, items)
        existed = any(getattr(item, "key", None) == PLAYBOOK_KEY for item in items) or bool(
            items
        )
        verb = "updated" if existed else "created"
        return f"{verb} memory {PLAYBOOK_KEY}"

    async def amanage_playbook(content=None, action: str = "create"):
        resolved = _profile_store(store)
        namespace = namespacer()
        items = await _alist_namespace(resolved, namespace)
        merged = _merge_playbook_items(items, content)
        aput = getattr(resolved, "aput", None)
        if aput is not None:
            await aput(namespace, PLAYBOOK_KEY, {"content": merged})
        else:
            resolved.put(namespace, PLAYBOOK_KEY, {"content": merged})
        await _adelete_extra_profile_keys(resolved, namespace, items)
        existed = any(getattr(item, "key", None) == PLAYBOOK_KEY for item in items) or bool(
            items
        )
        verb = "updated" if existed else "created"
        return f"{verb} memory {PLAYBOOK_KEY}"

    return StructuredTool.from_function(
        manage_playbook,
        coroutine=amanage_playbook,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def _create_list_layer_tool(layer: str, *, store):
    from ai_agent.memory.namespaces import user_id_from_config
    from ai_agent.memory.reconcile import list_layer_dump

    name = _tool_name(layer, "list_layer_memories")

    def list_layer_memories() -> str:
        resolved = _profile_store(store)
        config = _tool_config()
        user_id = user_id_from_config(config) or ""
        bound_layer = (
            (config.get("configurable") or {}).get("memory_layer") if config else None
        ) or layer
        if bound_layer.startswith("{"):
            bound_layer = layer
        return list_layer_dump(resolved, user_id, bound_layer) or "No memories."

    return StructuredTool.from_function(
        list_layer_memories,
        name=name,
        description=(
            "List every remaining memory on this user and layer after the last "
            "change. Call this after a delete or merge to refresh ids."
        ),
    )


def _tool_config() -> dict:
    try:
        from langgraph.config import get_config

        live = get_config()
        if isinstance(live, dict):
            return live
    except Exception:
        logger.debug("langgraph.config.get_config unavailable for memory tools", exc_info=True)
    try:
        from langgraph.utils.config import get_config

        live = get_config()
        if isinstance(live, dict):
            return live
    except Exception:
        logger.debug(
            "langgraph.utils.config.get_config unavailable for memory tools",
            exc_info=True,
        )
    return {}


def _merge_playbook_items(items, incoming) -> dict[str, Any]:
    from ai_agent.conf import get_agent_settings

    cap = get_agent_settings().memory_playbook_rule_cap
    extras = []
    canonical = None
    for item in items:
        if getattr(item, "key", None) == PLAYBOOK_KEY:
            canonical = item
        else:
            extras.append(item)
    base: dict[str, Any] = _content_of(canonical) if canonical is not None else {}
    for item in extras:
        extra = _content_of(item)
        if canonical is None:
            base = _patch_playbook(base, extra)
        else:
            base = _fill_missing_playbook(base, extra)
    merged = _patch_playbook(base, _serialize_content(incoming))
    rules = merged.get("rules")
    if not isinstance(rules, list):
        rules = []
    merged = {"rules": rules[:cap]}
    return merged


def _patch_playbook(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    incoming_rules = incoming.get("rules") if isinstance(incoming, dict) else None
    if incoming_rules is not None:
        merged["rules"] = list(incoming_rules)
    return merged


def _fill_missing_playbook(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if merged.get("rules"):
        return merged
    incoming_rules = incoming.get("rules") if isinstance(incoming, dict) else None
    if incoming_rules:
        merged["rules"] = list(incoming_rules)
    return merged


def _profile_store(initial):
    if initial is not None:
        return initial
    from langgraph.utils.config import get_store

    return get_store()


def _list_namespace(store, namespace: tuple[str, ...]) -> list:
    try:
        return list(store.search(namespace, limit=50) or [])
    except Exception:
        item = store.get(namespace, PROFILE_KEY)
        return [item] if item is not None else []


async def _alist_namespace(store, namespace: tuple[str, ...]) -> list:
    asearch = getattr(store, "asearch", None)
    try:
        if asearch is not None:
            return list(await asearch(namespace, limit=50) or [])
        return _list_namespace(store, namespace)
    except Exception:
        aget = getattr(store, "aget", None)
        if aget is not None:
            item = await aget(namespace, PROFILE_KEY)
        else:
            item = store.get(namespace, PROFILE_KEY)
        return [item] if item is not None else []


def _delete_extra_profile_keys(store, namespace: tuple[str, ...], items) -> None:
    for item in items:
        key = getattr(item, "key", None)
        if key and key != PROFILE_KEY:
            store.delete(namespace, key=key)


async def _adelete_extra_profile_keys(store, namespace: tuple[str, ...], items) -> None:
    adelete = getattr(store, "adelete", None)
    for item in items:
        key = getattr(item, "key", None)
        if not key or key == PROFILE_KEY:
            continue
        if adelete is not None:
            await adelete(namespace, key=key)
        else:
            store.delete(namespace, key=key)


def _merge_profile_items(items, incoming, schema: type[BaseModel]) -> dict[str, Any]:
    extras = []
    canonical = None
    for item in items:
        if getattr(item, "key", None) == PROFILE_KEY:
            canonical = item
        else:
            extras.append(item)
    base: dict[str, Any] = _content_of(canonical) if canonical is not None else {}
    for item in extras:
        extra = _content_of(item)
        if canonical is None:
            base = _patch_profile(base, extra)
        else:
            base = _fill_missing_profile(base, extra)
    merged = _patch_profile(base, _serialize_content(incoming))
    allowed = set(schema.model_fields)
    return {key: value for key, value in merged.items() if key in allowed}


def _fill_missing_profile(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (incoming or {}).items():
        if value is None:
            continue
        if merged.get(key) in (None, ""):
            merged[key] = value
    return merged


def _content_of(item) -> dict[str, Any]:
    value = getattr(item, "value", item)
    if isinstance(value, dict) and "content" in value:
        payload = value.get("content")
    else:
        payload = value
    return dict(payload) if isinstance(payload, dict) else {}


def _serialize_content(content) -> dict[str, Any]:
    if content is None:
        return {}
    if hasattr(content, "model_dump"):
        try:
            dumped = content.model_dump(mode="json")
        except Exception:
            dumped = content.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(content, dict):
        return content
    return {}


def _patch_profile(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (incoming or {}).items():
        if value is None:
            continue
        merged[key] = value
    return merged
