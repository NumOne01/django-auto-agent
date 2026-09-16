"""Queries and deletes against the LangGraph Agent Server database.

Threads and long-term memory are not Django models. Customer-facing Agent
Server auth filters threads by owner, so admin reads and deletes langgraph-db
directly. This module must not import from ``ai_agent.admin`` registering
modules.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional, Sequence
from urllib.parse import quote, unquote

from django.conf import settings

from ai_agent.memory.namespaces import SUPERVISOR_LAYER

logger = logging.getLogger(__name__)

NOT_CONFIGURED = "LangGraph database is not configured"
UNAVAILABLE = "LangGraph database is unavailable"

_PREVIEW_CHARS = 180
_TOOL_CONTENT_CHARS = 2000
_SEARCH_MIN_VALUE_CHARS = 3
_SEARCH_TIMEOUT_MS = 5000
_STORE_TRGM_ATTEMPTED = False
_THREAD_TABLES = ("thread", "threads")
_THREAD_DELETE_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoint_delete_queue",
    "checkpoints",
    "run",
    "cron",
    "thread_ttl",
)
_ALLOWED_DELETE_TABLES = set(_THREAD_TABLES) | set(_THREAD_DELETE_TABLES)


@dataclass
class TranscriptMessage:
    role: str
    content: str
    name: str = ""
    truncated: bool = False


@dataclass
class ThreadRecord:
    thread_id: str
    owner_id: str = ""
    status: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: int = 0
    last_message_preview: str = ""
    is_child: bool = False
    parent_thread_id: str = ""
    child_label: str = ""
    messages: list[TranscriptMessage] = field(default_factory=list)
    related: list["ThreadRecord"] = field(default_factory=list)
    user: Any = None


@dataclass
class MemoryRecord:
    prefix: str
    key: str
    user_id: str = ""
    layer: str = ""
    kind: str = ""
    value: Any = None
    preview: str = ""
    value_json: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    user: Any = None

    @property
    def layer_display(self) -> str:
        return layer_label(self.layer)

    @property
    def item_id(self) -> str:
        return encode_memory_id(self.prefix, self.key)

    @property
    def scope(self) -> str:
        parts = [part for part in str(self.prefix or "").split(".") if part]
        if len(parts) >= 2 and parts[0] == "prompts" and parts[1] == "global":
            return "global"
        return "local"


@dataclass
class PromptOverlayRecord:
    memory: MemoryRecord
    current: str = ""
    previous: str = ""
    updated_at: str = ""
    item_count: int = 0
    history: list = field(default_factory=list)

    @property
    def item_id(self) -> str:
        return self.memory.item_id

    @property
    def scope(self) -> str:
        return self.memory.scope

    @property
    def layer(self) -> str:
        return self.memory.layer

    @property
    def layer_display(self) -> str:
        return self.memory.layer_display

    @property
    def user_id(self) -> str:
        return self.memory.user_id

    @property
    def user(self):
        return self.memory.user

    @property
    def version_count(self) -> int:
        return 1 + len(self.history) if self.current else len(self.history)


@dataclass
class QueryPage:
    items: list
    total: int
    error: Optional[str] = None


def langgraph_database_uri() -> str:
    if getattr(settings, "TESTING", False):
        return os.environ.get("LANGGRAPH_ADMIN_TEST_URI", "").strip()
    return (
        os.environ.get("DATABASE_URI") or os.environ.get("LANGGRAPH_STORE_URI") or ""
    ).strip()


def layer_label(layer: str) -> str:
    if layer == SUPERVISOR_LAYER:
        return "Overall"
    return layer or "—"


def parse_memory_prefix(prefix: str) -> tuple[str, str, str]:
    """Return ``(user_id, layer, kind)`` from a dotted store prefix."""
    parts = [part for part in str(prefix or "").split(".") if part]
    if len(parts) >= 3 and parts[0] == "prompts" and parts[1] == "global":
        return "", parts[2], "prompt"
    if len(parts) >= 4 and parts[0] == "memories":
        return parts[1], parts[2], parts[3]
    if len(parts) == 3 and parts[0] == "memories":
        return parts[1], parts[2], ""
    return "", "", ""


def owner_from_payload(metadata, config=None) -> str:
    blobs = [metadata, config]
    if isinstance(config, dict):
        blobs.append(config.get("configurable"))
    for blob in blobs:
        if not isinstance(blob, dict):
            continue
        for key in ("owner", "user_id", "langgraph_auth_user_id"):
            value = blob.get(key)
            if value not in (None, ""):
                return str(value)
        auth_user = blob.get("langgraph_auth_user")
        if isinstance(auth_user, dict):
            identity = auth_user.get("identity")
            if identity not in (None, ""):
                return str(identity)
    return ""


def child_thread_id(thread_id: str, app_label: str) -> str:
    try:
        parent = uuid.UUID(str(thread_id))
    except ValueError:
        return f"{thread_id}::{app_label}"
    return str(uuid.uuid5(parent, app_label))


def agent_layer_labels() -> list[str]:
    from django.apps import apps

    labels = []
    for config in apps.get_app_configs():
        if config.label == "ai_agent":
            continue
        if getattr(config, "agent_expose", False) or getattr(
            config, "agent_description", None
        ):
            labels.append(config.label)
    return labels


def decode_messages(raw) -> list[TranscriptMessage]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("messages", raw)
    if not isinstance(raw, (list, tuple)):
        return []
    messages = []
    for item in raw:
        decoded = _decode_one_message(item)
        if decoded is not None:
            messages.append(decoded)
    return messages


def encode_memory_id(prefix: str, key: str) -> str:
    return f"{quote(str(prefix or ''), safe='')}::{quote(str(key or ''), safe='')}"


def decode_memory_id(raw: str) -> tuple[str, str]:
    left, sep, right = str(raw or "").partition("::")
    if not sep:
        return "", ""
    return unquote(left), unquote(right)


def preview_text(value, limit: int = _PREVIEW_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text


def list_threads(
    *,
    search: str = "",
    owner_id: str = "",
    include_children: bool = False,
    limit: int = 25,
    offset: int = 0,
) -> QueryPage:
    if not langgraph_database_uri():
        return QueryPage(items=[], total=0, error=NOT_CONFIGURED)
    try:
        with _connection() as conn:
            rows = _fetch_thread_rows(conn)
    except Exception:
        logger.exception("Failed to list agent threads")
        return QueryPage(items=[], total=0, error=UNAVAILABLE)
    records = [_thread_from_row(row, include_messages=False) for row in rows]
    _mark_child_threads(records)
    records = _filter_threads(records, search=search, owner_id=owner_id)
    if not include_children:
        records = [item for item in records if not item.is_child]
    records.sort(key=lambda item: item.updated_at or item.created_at or datetime.min, reverse=True)
    total = len(records)
    page = records[offset : offset + limit]
    _attach_users(page, "owner_id")
    return QueryPage(items=page, total=total)


def get_thread(thread_id: str) -> tuple[Optional[ThreadRecord], Optional[str]]:
    if not langgraph_database_uri():
        return None, NOT_CONFIGURED
    try:
        with _connection() as conn:
            row = _fetch_thread_row(conn, thread_id)
            if row is None:
                return None, None
            record = _thread_from_row(row, include_messages=True)
            if not record.messages:
                record.messages = _messages_from_checkpoint(conn, thread_id)
                record.message_count = len(record.messages)
                if record.messages:
                    record.last_message_preview = preview_text(
                        record.messages[-1].content
                    )
            related_ids = [
                child_thread_id(thread_id, label) for label in agent_layer_labels()
            ]
            related_ids.append(f"{thread_id}::")
            related_rows = _fetch_thread_rows_by_ids(conn, related_ids, thread_id)
    except Exception:
        logger.exception("Failed to load agent thread %s", thread_id)
        return None, UNAVAILABLE
    _mark_child_threads([record])
    record.related = [
        _thread_from_row(item, include_messages=False) for item in related_rows
    ]
    for related in record.related:
        related.parent_thread_id = thread_id
        related.is_child = True
        related.child_label = _child_label_for(thread_id, related.thread_id)
    _attach_users([record, *record.related], "owner_id")
    return record, None


def list_memories(
    *,
    search: str = "",
    user_id: str = "",
    layer: str = "",
    kind: str = "",
    kinds: Sequence[str] = (),
    limit: int = 25,
    offset: int = 0,
) -> QueryPage:
    if not langgraph_database_uri():
        return QueryPage(items=[], total=0, error=NOT_CONFIGURED)
    try:
        with _connection() as conn:
            records, total = _fetch_memories(
                conn,
                search=search,
                user_id=user_id,
                layer=layer,
                kind=kind,
                kinds=kinds,
                limit=limit,
                offset=offset,
            )
    except Exception:
        logger.exception("Failed to list agent memories")
        return QueryPage(items=[], total=0, error=UNAVAILABLE)
    _attach_users(records, "user_id")
    return QueryPage(items=records, total=total)


def list_prompt_overlays(
    *,
    search: str = "",
    user_id: str = "",
    layer: str = "",
    scope: str = "",
    limit: int = 50,
    offset: int = 0,
) -> QueryPage:
    if not langgraph_database_uri():
        return QueryPage(items=[], total=0, error=NOT_CONFIGURED)
    try:
        with _connection() as conn:
            records, total = _fetch_prompt_overlays(
                conn,
                search=search,
                user_id=user_id,
                layer=layer,
                scope=scope,
                limit=limit,
                offset=offset,
            )
    except Exception:
        logger.exception("Failed to list agent prompt overlays")
        return QueryPage(items=[], total=0, error=UNAVAILABLE)
    memories = [item.memory for item in records]
    _attach_users(memories, "user_id")
    return QueryPage(items=records, total=total)


def get_memory(item_id: str) -> tuple[Optional[MemoryRecord], Optional[str]]:
    prefix, key = decode_memory_id(item_id)
    if not prefix or not key:
        return None, None
    if not langgraph_database_uri():
        return None, NOT_CONFIGURED
    try:
        with _connection() as conn:
            if "store" not in _table_names(conn):
                return None, None
            sql = """
                SELECT prefix, key, value, created_at, updated_at
                FROM store
                WHERE prefix = %s AND key = %s
                LIMIT 1
            """
            with conn.cursor() as cur:
                cur.execute(sql, (prefix, key))
                row = cur.fetchone()
    except Exception:
        logger.exception("Failed to load agent memory %s", item_id)
        return None, UNAVAILABLE
    if not row:
        return None, None
    record = _memory_from_row(row)
    _attach_users([record], "user_id")
    return record, None


def get_prompt_overlay(item_id: str) -> tuple[Optional[PromptOverlayRecord], Optional[str]]:
    prefix, key = decode_memory_id(item_id)
    if not prefix or not key:
        return None, None
    if not langgraph_database_uri():
        return None, NOT_CONFIGURED
    try:
        with _connection() as conn:
            if "store" not in _table_names(conn):
                return None, None
            sql = """
                SELECT prefix, key, value, created_at, updated_at
                FROM store
                WHERE prefix = %s AND key = %s
                LIMIT 1
            """
            with conn.cursor() as cur:
                cur.execute(sql, (prefix, key))
                row = cur.fetchone()
    except Exception:
        logger.exception("Failed to load agent prompt overlay %s", item_id)
        return None, UNAVAILABLE
    if not row:
        return None, None
    record = prompt_overlay_from_memory(_memory_from_row(row))
    _attach_users([record.memory], "user_id")
    return record, None


def prompt_overlay_from_memory(item: MemoryRecord) -> PromptOverlayRecord:
    from ai_agent.memory.prompt_optimize import overlay_doc_from_value

    doc = overlay_doc_from_value(item.value)
    return PromptOverlayRecord(
        memory=item,
        current=doc.text,
        previous=doc.previous,
        updated_at=doc.updated_at or str(item.updated_at or ""),
        item_count=doc.item_count,
        history=list(doc.history),
    )


def memories_grouped_for_user(user_id: str) -> tuple[dict[str, dict[str, list[MemoryRecord]]], Optional[str]]:
    page = list_memories(user_id=str(user_id), limit=500, offset=0)
    if page.error:
        return {}, page.error
    grouped: dict[str, dict[str, list[MemoryRecord]]] = {}
    for item in page.items:
        layer = item.layer or "unknown"
        kind = item.kind or "other"
        grouped.setdefault(
            layer,
            {
                "profile": [],
                "semantic": [],
                "episodes": [],
                "playbook": [],
                "prompt": [],
                "other": [],
            },
        )
        bucket = kind if kind in grouped[layer] else "other"
        grouped[layer][bucket].append(item)
    ordered = {}
    layers = sorted(grouped, key=lambda name: (name != SUPERVISOR_LAYER, name))
    for layer in layers:
        ordered[layer] = grouped[layer]
    return ordered, None


def resolve_user_query(raw: str) -> str:
    """Return a user id string from a pk or USERNAME_FIELD search."""
    value = (raw or "").strip()
    if not value:
        return ""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = User.objects.filter(**{User.USERNAME_FIELD: value}).first()
    if user is not None:
        return str(user.pk)
    if value.isdigit():
        user = User.objects.filter(pk=int(value)).first()
        if user is not None:
            return str(user.pk)
    return value


def delete_threads(thread_ids: list[str], *, include_children: bool = True) -> tuple[int, Optional[str]]:
    ids = [str(item).strip() for item in thread_ids if str(item).strip()]
    if not ids:
        return 0, None
    if not langgraph_database_uri():
        return 0, NOT_CONFIGURED
    try:
        with _connection() as conn:
            names = _table_names(conn)
            expanded = list(ids)
            if include_children:
                expanded = _expand_thread_delete_ids(ids)
            deleted = _delete_thread_ids(conn, names, expanded, like_parents=ids)
            conn.commit()
    except Exception:
        logger.exception("Failed to delete agent threads")
        return 0, UNAVAILABLE
    return deleted, None


def delete_memories(item_ids: list[str]) -> tuple[int, Optional[str]]:
    pairs = []
    for raw in item_ids:
        prefix, key = decode_memory_id(raw)
        if prefix and key:
            pairs.append((prefix, key))
    if not pairs:
        return 0, None
    if not langgraph_database_uri():
        return 0, NOT_CONFIGURED
    try:
        with _connection() as conn:
            names = _table_names(conn)
            deleted = _delete_memory_pairs(conn, names, pairs)
            conn.commit()
    except Exception:
        logger.exception("Failed to delete agent memories")
        return 0, UNAVAILABLE
    return deleted, None


def delete_prompt_overlays(item_ids: list[str]) -> tuple[int, Optional[str]]:
    return delete_memories(item_ids)


# --- connection / SQL -------------------------------------------------------


@contextmanager
def _connection() -> Iterator[Any]:
    uri = langgraph_database_uri()
    if not uri:
        yield None
        return
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(uri, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def _fetch_thread_rows(conn) -> list[dict]:
    table = _thread_table(conn)
    if table:
        columns = _table_columns(conn, table)
        select = _thread_select_sql(table, columns)
        with conn.cursor() as cur:
            cur.execute(select)
            return list(cur.fetchall() or [])
    if "checkpoints" not in _table_names(conn):
        return []
    sql = """
        SELECT c.thread_id::text AS thread_id,
               c.metadata AS metadata,
               NULL::text AS status,
               NULL::timestamptz AS created_at,
               NULL::timestamptz AS updated_at,
               NULL::jsonb AS config,
               c.checkpoint AS checkpoint,
               NULL::jsonb AS last_message,
               0 AS message_count
        FROM checkpoints c
        INNER JOIN (
            SELECT thread_id, MAX(checkpoint_id) AS checkpoint_id
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
        ) latest
          ON latest.thread_id = c.thread_id
         AND latest.checkpoint_id = c.checkpoint_id
        WHERE c.checkpoint_ns = ''
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return list(cur.fetchall() or [])


def _fetch_thread_row(conn, thread_id: str) -> Optional[dict]:
    table = _thread_table(conn)
    if table:
        columns = _table_columns(conn, table)
        select = _thread_select_sql(table, columns, where_id=True)
        with conn.cursor() as cur:
            cur.execute(select, (str(thread_id),))
            return cur.fetchone()
    if "checkpoints" not in _table_names(conn):
        return None
    sql = """
        SELECT c.thread_id::text AS thread_id,
               c.metadata AS metadata,
               NULL::text AS status,
               NULL::timestamptz AS created_at,
               NULL::timestamptz AS updated_at,
               NULL::jsonb AS config,
               c.checkpoint AS checkpoint,
               NULL::jsonb AS last_message,
               0 AS message_count
        FROM checkpoints c
        WHERE c.checkpoint_ns = '' AND c.thread_id = %s
        ORDER BY c.checkpoint_id DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (str(thread_id),))
        return cur.fetchone()


def _fetch_thread_rows_by_ids(conn, related_ids: list[str], parent_id: str) -> list[dict]:
    exact = [item for item in related_ids if not item.endswith("::")]
    table = _thread_table(conn)
    if not table:
        return []
    columns = _table_columns(conn, table)
    id_expr = "thread_id::text"
    clauses = []
    params: list[Any] = []
    if exact:
        clauses.append(f"{id_expr} = ANY(%s)")
        params.append(exact)
    clauses.append(f"{id_expr} LIKE %s")
    params.append(f"{parent_id}::%")
    where = " OR ".join(clauses)
    select = _thread_select_sql(table, columns) + f" WHERE {where}"
    with conn.cursor() as cur:
        cur.execute(select, params)
        rows = list(cur.fetchall() or [])
    return [row for row in rows if str(row.get("thread_id")) != str(parent_id)]


def _thread_select_sql(table: str, columns: set[str], *, where_id: bool = False) -> str:
    def col(name: str, alias: str, sql_default: str) -> str:
        if name in columns:
            return f"{name} AS {alias}"
        return f"{sql_default} AS {alias}"

    values_expr = "values" if "values" in columns else "NULL::jsonb"
    parts = [
        "thread_id::text AS thread_id",
        col("created_at", "created_at", "NULL::timestamptz"),
        col("updated_at", "updated_at", "NULL::timestamptz"),
        col("metadata", "metadata", "'{}'::jsonb"),
        col("status", "status", "NULL::text"),
        col("config", "config", "'{}'::jsonb"),
        f"""
        CASE
          WHEN jsonb_typeof({values_expr}->'messages') = 'array'
          THEN jsonb_array_length({values_expr}->'messages')
          ELSE 0
        END AS message_count
        """,
        f"{values_expr}->'messages'->-1 AS last_message",
    ]
    if where_id:
        parts.append(f"{values_expr} AS values")
        parts.append("NULL::jsonb AS checkpoint")
    else:
        parts.append("NULL::jsonb AS values")
        parts.append("NULL::jsonb AS checkpoint")
    sql = f"SELECT {', '.join(parts)} FROM {table}"
    if where_id:
        sql += " WHERE thread_id::text = %s"
    return sql


def _fetch_memories(
    conn,
    *,
    search: str,
    user_id: str,
    layer: str,
    kind: str,
    kinds: Sequence[str] = (),
    limit: int,
    offset: int,
) -> tuple[list[MemoryRecord], int]:
    if "store" not in _table_names(conn):
        return [], 0
    where, params = _memory_filters(
        search=search, user_id=user_id, layer=layer, kind=kind, kinds=kinds
    )
    rows, total = _run_store_search(
        conn, where, params, limit=limit, offset=offset, search=search
    )
    return [_memory_from_row(item) for item in rows], total


def _fetch_prompt_overlays(
    conn,
    *,
    search: str,
    user_id: str,
    layer: str,
    scope: str,
    limit: int,
    offset: int,
) -> tuple[list[PromptOverlayRecord], int]:
    if "store" not in _table_names(conn):
        return [], 0
    where, params = _prompt_filters(
        search=search, user_id=user_id, layer=layer, scope=scope
    )
    rows, total = _run_store_search(
        conn, where, params, limit=limit, offset=offset, search=search
    )
    return [
        prompt_overlay_from_memory(_memory_from_row(item)) for item in rows
    ], total


def _run_store_search(
    conn,
    where: str,
    params: list[Any],
    *,
    limit: int,
    offset: int,
    search: str,
) -> tuple[list[dict], int]:
    if search:
        _ensure_store_search_index(conn)
    count_sql = f"SELECT COUNT(*) AS n FROM store WHERE {where}"
    list_sql = f"""
        SELECT prefix, key, value, created_at, updated_at
        FROM store
        WHERE {where}
        ORDER BY updated_at DESC NULLS LAST, prefix, key
        LIMIT %s OFFSET %s
    """
    with conn.cursor() as cur:
        if search:
            _apply_search_timeout(cur)
        cur.execute(count_sql, params)
        row = cur.fetchone() or {}
        total = int(row.get("n") or 0)
        cur.execute(list_sql, [*params, limit, offset])
        rows = list(cur.fetchall() or [])
    return rows, total


def _apply_search_timeout(cur) -> None:
    try:
        cur.execute("SET LOCAL statement_timeout = %s", (_SEARCH_TIMEOUT_MS,))
    except Exception:
        logger.debug("Could not set statement_timeout on LangGraph store search")


def _ensure_store_search_index(conn) -> None:
    """Best-effort GIN trigram index on store.value. Not a Django migration."""
    global _STORE_TRGM_ATTEMPTED
    if _STORE_TRGM_ATTEMPTED:
        return
    _STORE_TRGM_ATTEMPTED = True
    if "store" not in _table_names(conn):
        return
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ai_agent_store_value_trgm
                  ON store USING gin ((COALESCE(value::text, '')) gin_trgm_ops)
                """
            )
        conn.commit()
    except Exception:
        logger.exception("Could not create LangGraph store trigram index")
        try:
            conn.rollback()
        except Exception:
            logger.debug("Could not roll back after trigram index failure", exc_info=True)


_ILIKE_PREFIX_KEY_SQL = (
    "(prefix ILIKE %s ESCAPE '\\' OR key ILIKE %s ESCAPE '\\')"
)
_ILIKE_VALUE_SQL = "COALESCE(value::text, '') ILIKE %s ESCAPE '\\'"


def _ilike_contains(term: str) -> str:
    """Wrap ``term`` for ILIKE substring match with literal ``%`` / ``_``."""
    escaped = (
        str(term).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return f"%{escaped}%"


def _memory_filters(
    *,
    search: str,
    user_id: str,
    layer: str,
    kind: str,
    kinds: Sequence[str] = (),
) -> tuple[str, list[Any]]:
    """Build a store WHERE clause. Wildcards are bound params for psycopg3."""
    clauses = ["prefix LIKE %s"]
    params: list[Any] = ["memories.%"]
    kind_list = tuple(
        str(item).strip() for item in (kinds or ()) if str(item).strip()
    )
    if kind_list:
        kind = ""
    if kind == "prompt":
        clauses = [
            "((prefix LIKE %s AND split_part(prefix, '.', 4) = %s) OR prefix LIKE %s)"
        ]
        params = ["memories.%", "prompt", "prompts.global.%"]
        kind = ""
    if user_id:
        clauses.append("split_part(prefix, '.', 2) = %s")
        params.append(str(user_id))
    if layer:
        clauses.append("split_part(prefix, '.', 3) = %s")
        params.append(layer)
    if kind_list:
        clauses.append("split_part(prefix, '.', 4) = ANY(%s)")
        params.append(list(kind_list))
    elif kind:
        clauses.append("split_part(prefix, '.', 4) = %s")
        params.append(kind)
    _append_search_clause(clauses, params, search)
    return " AND ".join(clauses), params


def _prompt_filters(
    *, search: str, user_id: str, layer: str, scope: str
) -> tuple[str, list[Any]]:
    """WHERE clause for local prompt addenda and shared global overlays."""
    local_sql = "(prefix LIKE %s AND split_part(prefix, '.', 4) = %s)"
    global_sql = "prefix LIKE %s"
    scope = str(scope or "").strip().lower()
    if scope == "global":
        clauses = [global_sql]
        params: list[Any] = ["prompts.global.%"]
    elif scope == "local":
        clauses = [local_sql]
        params = ["memories.%", "prompt"]
    else:
        clauses = [f"({local_sql} OR {global_sql})"]
        params = ["memories.%", "prompt", "prompts.global.%"]
    if user_id:
        clauses.append("split_part(prefix, '.', 2) = %s")
        params.append(str(user_id))
    if layer:
        clauses.append("split_part(prefix, '.', 3) = %s")
        params.append(layer)
    _append_search_clause(clauses, params, search)
    return " AND ".join(clauses), params


def _append_search_clause(
    clauses: list[str], params: list[Any], search: str
) -> None:
    term = (search or "").strip()
    if not term:
        return
    like = _ilike_contains(term)
    if len(term) >= _SEARCH_MIN_VALUE_CHARS:
        clauses.append(f"({_ILIKE_PREFIX_KEY_SQL} OR {_ILIKE_VALUE_SQL})")
        params.extend([like, like, like])
        return
    clauses.append(_ILIKE_PREFIX_KEY_SQL)
    params.extend([like, like])


def _messages_from_checkpoint(conn, thread_id: str) -> list[TranscriptMessage]:
    if "checkpoints" not in _table_names(conn):
        return []
    sql = """
        SELECT checkpoint
        FROM checkpoints
        WHERE checkpoint_ns = '' AND thread_id = %s
        ORDER BY checkpoint_id DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (str(thread_id),))
        row = cur.fetchone()
    if not row:
        return []
    checkpoint = row.get("checkpoint") or {}
    if not isinstance(checkpoint, dict):
        return []
    channel_values = checkpoint.get("channel_values") or {}
    return decode_messages(channel_values.get("messages"))


def _thread_table(conn) -> Optional[str]:
    names = _table_names(conn)
    for name in _THREAD_TABLES:
        if name in names:
            return name
    return None


def _table_names(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        return {row["table_name"] for row in cur.fetchall()}


def _table_columns(conn, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table,),
        )
        return {row["column_name"] for row in cur.fetchall()}


def _expand_thread_delete_ids(ids: list[str]) -> list[str]:
    expanded = set(ids)
    for thread_id in ids:
        for label in agent_layer_labels():
            expanded.add(child_thread_id(thread_id, label))
    return list(expanded)


def _delete_thread_ids(
    conn, names: set[str], ids: list[str], *, like_parents: list[str]
) -> int:
    if not ids:
        return 0
    like_patterns = [f"{parent}::%" for parent in like_parents]
    with conn.cursor() as cur:
        for table in _THREAD_DELETE_TABLES:
            if table not in names:
                continue
            _delete_by_thread_id(cur, table, ids, like_patterns)
        deleted = 0
        thread_table = _thread_table(conn)
        if thread_table:
            deleted = _delete_by_thread_id(cur, thread_table, ids, like_patterns)
        elif "checkpoints" in names:
            deleted = _delete_by_thread_id(cur, "checkpoints", ids, like_patterns)
    return deleted


def _delete_by_thread_id(cur, table: str, ids: list[str], like_patterns: list[str]) -> int:
    if table not in _ALLOWED_DELETE_TABLES:
        raise ValueError(f"Refusing to delete from unexpected table {table!r}")
    if like_patterns:
        cur.execute(
            f"DELETE FROM {table} WHERE thread_id::text = ANY(%s) OR thread_id::text LIKE ANY(%s)",
            (ids, like_patterns),
        )
    else:
        cur.execute(
            f"DELETE FROM {table} WHERE thread_id::text = ANY(%s)",
            (ids,),
        )
    return int(cur.rowcount or 0)


def _delete_memory_pairs(conn, names: set[str], pairs: list[tuple[str, str]]) -> int:
    if "store" not in names or not pairs:
        return 0
    deleted = 0
    with conn.cursor() as cur:
        for prefix, key in pairs:
            if "store_vectors" in names:
                cur.execute(
                    "DELETE FROM store_vectors WHERE prefix = %s AND key = %s",
                    (prefix, key),
                )
            cur.execute(
                "DELETE FROM store WHERE prefix = %s AND key = %s",
                (prefix, key),
            )
            if cur.rowcount and cur.rowcount > 0:
                deleted += 1
    return deleted


def _thread_from_row(row: dict, *, include_messages: bool) -> ThreadRecord:
    metadata = row.get("metadata") or {}
    config = row.get("config") or {}
    owner_id = owner_from_payload(metadata, config)
    messages = []
    if include_messages:
        values = row.get("values") or {}
        messages = decode_messages(values)
        if not messages:
            checkpoint = row.get("checkpoint") or {}
            if isinstance(checkpoint, dict):
                channel_values = checkpoint.get("channel_values") or {}
                messages = decode_messages(channel_values.get("messages"))
    preview_source = row.get("last_message")
    preview = ""
    if messages:
        preview = preview_text(messages[-1].content)
    elif preview_source is not None:
        preview = preview_text(_message_content(preview_source) or preview_source)
    count = int(row.get("message_count") or 0)
    if messages:
        count = len(messages)
    thread_id = str(row.get("thread_id") or "")
    is_child = False
    parent_thread_id = ""
    child_label = ""
    if "::" in thread_id:
        is_child = True
        parent_thread_id, child_label = thread_id.split("::", 1)
    return ThreadRecord(
        thread_id=thread_id,
        owner_id=owner_id,
        status=str(row.get("status") or ""),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        message_count=count,
        last_message_preview=preview,
        messages=messages,
        is_child=is_child,
        parent_thread_id=parent_thread_id,
        child_label=child_label,
    )


def _memory_from_row(row: dict) -> MemoryRecord:
    prefix = str(row.get("prefix") or "")
    user_id, layer, kind = parse_memory_prefix(prefix)
    value = row.get("value")
    return MemoryRecord(
        prefix=prefix,
        key=str(row.get("key") or ""),
        user_id=user_id,
        layer=layer,
        kind=kind,
        value=value,
        preview=preview_text(value),
        value_json=_value_json(value),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _value_json(value) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def _filter_threads(
    records: list[ThreadRecord], *, search: str, owner_id: str
) -> list[ThreadRecord]:
    owner_id = (owner_id or "").strip()
    search = (search or "").strip()
    if owner_id:
        records = [item for item in records if item.owner_id == owner_id]
    if not search:
        return records
    resolved = resolve_user_query(search)
    needle = search.lower()
    matched = []
    for item in records:
        haystacks = [
            item.thread_id,
            item.owner_id,
            item.status,
            item.last_message_preview,
        ]
        if any(needle in str(part).lower() for part in haystacks):
            matched.append(item)
            continue
        if resolved and item.owner_id == resolved:
            matched.append(item)
            continue
        user = getattr(item, "user", None)
        username = ""
        if user is not None:
            getter = getattr(user, "get_username", None)
            username = str(getter() or "") if callable(getter) else ""
        if needle in username.lower():
            matched.append(item)
    return matched


def _mark_child_threads(records: list[ThreadRecord]) -> None:
    by_id = {item.thread_id: item for item in records}
    labels = agent_layer_labels()
    for parent in list(records):
        for label in labels:
            child_id = child_thread_id(parent.thread_id, label)
            child = by_id.get(child_id)
            if child is None or child.thread_id == parent.thread_id:
                continue
            child.is_child = True
            child.parent_thread_id = parent.thread_id
            child.child_label = label
        prefix = f"{parent.thread_id}::"
        for child in records:
            if child.thread_id.startswith(prefix):
                child.is_child = True
                child.parent_thread_id = parent.thread_id
                child.child_label = child.thread_id.split("::", 1)[-1]


def _child_label_for(parent_id: str, child_id: str) -> str:
    for label in agent_layer_labels():
        if child_thread_id(parent_id, label) == child_id:
            return label
    if child_id.startswith(f"{parent_id}::"):
        return child_id.split("::", 1)[-1]
    return ""


def _attach_users(records: list, id_attr: str) -> None:
    from django.contrib.auth import get_user_model

    User = get_user_model()

    raw_ids = []
    for record in records:
        value = getattr(record, id_attr, "") or ""
        if str(value).isdigit():
            raw_ids.append(int(value))
    if not raw_ids:
        return
    users = {str(user.pk): user for user in User.objects.filter(pk__in=raw_ids)}
    for record in records:
        record.user = users.get(str(getattr(record, id_attr, "") or ""))


def _decode_one_message(item) -> Optional[TranscriptMessage]:
    if item is None:
        return None
    role = _message_role(item)
    content = _message_content(item)
    name = _message_name(item)
    truncated = False
    if role == "tool" and len(content) > _TOOL_CONTENT_CHARS:
        content = content[:_TOOL_CONTENT_CHARS].rstrip() + "…"
        truncated = True
    return TranscriptMessage(role=role, content=content, name=name, truncated=truncated)


def _message_role(item) -> str:
    if isinstance(item, dict):
        if item.get("type") == "constructor" and item.get("id"):
            last = str(item["id"][-1])
            lowered = last.lower()
            if "human" in lowered:
                return "human"
            if "ai" in lowered or "assistant" in lowered:
                return "ai"
            if "tool" in lowered:
                return "tool"
            if "system" in lowered:
                return "system"
        for key in ("type", "role"):
            value = item.get(key)
            if value:
                value = str(value).lower()
                if value in {"human", "user"}:
                    return "human"
                if value in {"ai", "assistant"}:
                    return "ai"
                if value == "tool":
                    return "tool"
                if value == "system":
                    return "system"
                return value
        kwargs = item.get("kwargs")
        if isinstance(kwargs, dict):
            return _message_role(kwargs)
    type_name = type(item).__name__.lower()
    if "human" in type_name:
        return "human"
    if "ai" in type_name:
        return "ai"
    if "tool" in type_name:
        return "tool"
    role = getattr(item, "type", None) or getattr(item, "role", None)
    if role:
        return str(role).lower()
    return "unknown"


def _message_content(item) -> str:
    if isinstance(item, str):
        return item
    content = None
    if isinstance(item, dict):
        kwargs = item.get("kwargs")
        if isinstance(kwargs, dict) and "content" in kwargs:
            content = kwargs.get("content")
        else:
            content = item.get("content")
    else:
        content = getattr(item, "content", None)
    return _flatten_content(content)


def _message_name(item) -> str:
    if isinstance(item, dict):
        kwargs = item.get("kwargs") if isinstance(item.get("kwargs"), dict) else item
        return str(kwargs.get("name") or kwargs.get("tool") or "")
    return str(getattr(item, "name", "") or "")


def _flatten_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif "text" in block:
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(preview_text(block, limit=400))
            else:
                parts.append(str(block))
        return "".join(parts)
    return preview_text(content, limit=2000)
