"""LangGraph Store factory for long-term memory."""

from __future__ import annotations

import logging
import os

from django.conf import settings

from ai_agent.conf import AgentSettings, get_agent_settings

logger = logging.getLogger(__name__)


def build_memory_store(agent_settings: AgentSettings | None = None):
    """Return a BaseStore, or None when memory is off / platform-injected."""
    agent_settings = agent_settings or get_agent_settings()
    if not agent_settings.memory_enabled:
        return None
    backend = agent_settings.memory_store
    if backend == "platform" and not _force_in_memory_store():
        return None
    if _force_in_memory_store() or backend == "memory":
        return _in_memory_store(agent_settings)
    if backend == "postgres":
        return _postgres_store(agent_settings)
    if backend == "redis":
        return _redis_store(agent_settings)
    logger.warning("Unknown MEMORY_STORE %s; using in-memory store", backend)
    return _in_memory_store(agent_settings)


def _force_in_memory_store() -> bool:
    if getattr(settings, "TESTING", False):
        return True
    engine = settings.DATABASES["default"]["ENGINE"]
    return engine.endswith("sqlite3")


def _index_config(agent_settings: AgentSettings) -> dict:
    if getattr(settings, "TESTING", False):
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return {
            "dims": 8,
            "embed": DeterministicFakeEmbedding(size=8),
            "fields": ["$"],
        }
    return {
        "dims": agent_settings.memory_embedding_dims,
        "embed": agent_settings.memory_embeddings,
        "fields": ["$"],
    }


def _in_memory_store(agent_settings: AgentSettings):
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore(index=_index_config(agent_settings))


def _postgres_store(agent_settings: AgentSettings):
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from langgraph.store.postgres import PostgresStore

    uri = _postgres_uri()
    pool = ConnectionPool(
        conninfo=uri,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=1,
        max_size=4,
    )
    store = PostgresStore(conn=pool, index=_index_config(agent_settings))
    store.setup()
    return store


def _postgres_uri() -> str:
    uri = (os.environ.get("DATABASE_URI") or os.environ.get("LANGGRAPH_STORE_URI") or "").strip()
    if uri:
        return uri
    raise RuntimeError(
        "MEMORY_STORE=postgres requires DATABASE_URI (LangGraph Agent Server "
        "database, not the host Django DATABASES default)."
    )


def _redis_store(agent_settings: AgentSettings):
    try:
        from langgraph.store.redis import RedisStore
    except ImportError as exc:
        raise RuntimeError(
            "MEMORY_STORE=redis requires langgraph-checkpoint-redis. "
            "The supported production path is postgres/platform with pgvector."
        ) from exc
    uri = (os.environ.get("REDIS_URI") or "").strip()
    if not uri:
        raise RuntimeError("MEMORY_STORE=redis requires REDIS_URI")
    store = RedisStore.from_conn_string(uri, index=_index_config(agent_settings))
    if hasattr(store, "setup"):
        store.setup()
    return store
