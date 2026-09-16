"""Per-user memory curator: deterministic reconcile, optional LLM curate/optimize."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ai_agent.conf import (
    AgentSettings,
    get_agent_settings,
)
from ai_agent.memory.namespaces import (
    normalize_memory_config,
    user_id_from_config,
)
from ai_agent.memory.reconcile import list_layer_dump, reconcile_layer
from ai_agent.memory.tools import build_curator_tools

logger = logging.getLogger(__name__)

CURATOR_ASSISTANT_ID = "memory_curator"

_CURATE_PROMPT = """
You are a memory curator. You only merge or delete memories that are already listed.

Protocol:
1. The dump in the user message is the source of truth. Call list_layer_memories after a change if you need fresh ids.
2. Update or delete only. Never create a new fact, collection document, or episode.
3. Prefer one canonical fact per subject+predicate.
4. Consolidate similar episodes. Keep failure plus how to prevent it.
5. Do not invent information that is not in the dump.
6. If nothing is redundant, make no tool calls.
"""

_OPTIMIZE_PROMPT = """
You are a local optimizer for one customer and one domain layer.

Protocol:
1. Use only the dump in the user message (profile, episodes, existing playbook).
2. If episodes show a failure and a prevention lesson, patch the standing profile field for constraints/preferences when that is a durable how-to for this customer.
3. Upsert the singleton playbook with short rules (trigger, do, dont, why). At most eight rules.
4. You may update or delete an existing episode to merge a lesson. Do not create episodes or facts.
5. Do not dump the current task, balances, or one-off requests into the profile.
6. If nothing should change, make no tool calls.
"""

CURATOR_ASSISTANT_ID = "memory_curator"


class CuratorState(TypedDict, total=False):
    user_id: str
    layer: str
    dirty: bool
    needs_llm: bool
    needs_optimize: bool
    dump: str
    signals: list[str]


def build_curator_graph(*, store=None):
    """Compiled curator for LangGraph Agent Server (graph id ``memory_curator``)."""
    builder = StateGraph(CuratorState)
    builder.add_node("load_layer", _load_layer)
    builder.add_node("deterministic_reconcile", _deterministic_reconcile)
    builder.add_node("llm_curate", _llm_curate)
    builder.add_node("local_optimize", _local_optimize)
    builder.add_node("prompt_optimize", _prompt_optimize)
    builder.add_edge(START, "load_layer")
    builder.add_edge("load_layer", "deterministic_reconcile")
    builder.add_edge("deterministic_reconcile", "llm_curate")
    builder.add_edge("llm_curate", "local_optimize")
    builder.add_edge("local_optimize", "prompt_optimize")
    builder.add_edge("prompt_optimize", END)
    kwargs: dict[str, Any] = {}
    if store is not None:
        kwargs["store"] = store
    return builder.compile(**kwargs)


def curator_thread_id(user_id: str, layer: str) -> str:
    return f"curator:{user_id}:{layer}"


def curator_after_seconds(settings: AgentSettings | None = None) -> int:
    agent_settings = settings or get_agent_settings()
    return max(
        0,
        agent_settings.memory_debounce_seconds
        + agent_settings.memory_curator_delay_seconds,
    )


def submit_curator(
    layer: str,
    config,
    *,
    store=None,
    settings: AgentSettings | None = None,
    cache: dict | None = None,
    graphs: dict | None = None,
    curate_agents: dict | None = None,
    optimize_agents: dict | None = None,
) -> None:
    agent_settings = settings or get_agent_settings()
    if not agent_settings.memory_curator_enabled:
        return
    bound = normalize_memory_config(config, layer)
    user_id = user_id_from_config(bound)
    if not user_id:
        return
    thread_id = curator_thread_id(user_id, layer)
    payload = {"user_id": str(user_id), "layer": layer}
    executor = _curator_executor(
        layer,
        store=store,
        settings=agent_settings,
        cache=cache,
        graphs=graphs,
        curate_agents=curate_agents,
        optimize_agents=optimize_agents,
    )
    executor.submit(
        payload,
        config=bound,
        after_seconds=curator_after_seconds(agent_settings),
        thread_id=thread_id,
    )


def run_curator(
    user_id: str,
    layer: str,
    *,
    store=None,
    config=None,
    settings: AgentSettings | None = None,
    graphs: dict | None = None,
    curate_agents: dict | None = None,
    optimize_agents: dict | None = None,
) -> dict:
    """Invoke the curator immediately (management command / tests)."""
    graph = _local_curator_graph(
        store=store,
        graphs=graphs,
        curate_agents=curate_agents,
        optimize_agents=optimize_agents,
    )
    bound = normalize_memory_config(config, layer)
    bound = normalize_memory_config(
        {**bound, "configurable": {**(bound.get("configurable") or {}), "user_id": str(user_id)}},
        layer,
    )
    return graph.invoke({"user_id": str(user_id), "layer": layer}, bound)


def _load_layer(state: CuratorState) -> dict:
    return {
        "user_id": str(state.get("user_id") or ""),
        "layer": str(state.get("layer") or ""),
    }


def _deterministic_reconcile(state: CuratorState, config=None) -> dict:
    store = _store_from_runtime(config)
    user_id = str(state.get("user_id") or "")
    layer = str(state.get("layer") or "")
    if not user_id or not layer or store is None:
        return {
            "dirty": False,
            "needs_llm": False,
            "needs_optimize": False,
            "dump": "",
            "signals": [],
        }
    result = reconcile_layer(store, user_id, layer)
    return {
        "dirty": result.dirty,
        "needs_llm": result.needs_llm,
        "needs_optimize": result.needs_optimize,
        "dump": result.dump,
        "signals": result.signals,
    }


def _llm_curate(state: CuratorState, config=None) -> dict:
    if not state.get("needs_llm"):
        return {}
    _run_llm_curate(state, config)
    store = _store_from_runtime(config)
    dump = list_layer_dump(store, state.get("user_id") or "", state.get("layer") or "")
    return {"dump": dump, "dirty": True}


def _local_optimize(state: CuratorState, config=None) -> dict:
    if not state.get("needs_optimize"):
        return {}
    _run_local_optimize(state, config)
    return {"dirty": True}


def _prompt_optimize(state: CuratorState, config=None) -> dict:
    """Gradient prompt overlay; failures must not fail the curator."""
    try:
        from ai_agent.memory.prompt_optimize import maybe_run_local_prompt_optimize

        store = _store_from_runtime(config)
        maybe_run_local_prompt_optimize(
            str(state.get("user_id") or ""),
            str(state.get("layer") or ""),
            store=store,
        )
    except Exception:
        logger.exception("Local prompt optimizer failed")
    return {}


def _run_llm_curate(state: CuratorState, config=None) -> None:
    layer = str(state.get("layer") or "")
    store = _store_from_runtime(config)
    agent = _curate_agent(layer, store=store)
    dump = state.get("dump") or "No memories."
    bound = normalize_memory_config(config, layer)
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Reconcile remaining near-duplicates that could not be "
                        "merged automatically.\n\n"
                        f"{dump}"
                    )
                )
            ]
        },
        bound,
    )


def _run_local_optimize(state: CuratorState, config=None) -> None:
    layer = str(state.get("layer") or "")
    store = _store_from_runtime(config)
    agent = _optimize_agent(layer, store=store)
    dump = state.get("dump") or "No memories."
    bound = normalize_memory_config(config, layer)
    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Write a local playbook and/or profile patch from these "
                        "failure lessons.\n\n"
                        f"{dump}"
                    )
                )
            ]
        },
        bound,
    )


def _curate_agent(
    layer: str,
    *,
    store=None,
    settings: AgentSettings | None = None,
    curate_agents: dict | None = None,
):
    if curate_agents is not None:
        cached = curate_agents.get(layer)
        if cached is not None:
            return cached
    agent_settings = settings or get_agent_settings()
    extra = {}
    if store is not None:
        extra["store"] = store
    agent = create_agent(
        model=agent_settings.memory_model,
        tools=build_curator_tools(layer, store=store, optimize=False),
        system_prompt=_CURATE_PROMPT,
        **extra,
    )
    if curate_agents is not None:
        curate_agents[layer] = agent
    return agent


def _optimize_agent(
    layer: str,
    *,
    store=None,
    settings: AgentSettings | None = None,
    optimize_agents: dict | None = None,
):
    if optimize_agents is not None:
        cached = optimize_agents.get(layer)
        if cached is not None:
            return cached
    agent_settings = settings or get_agent_settings()
    extra = {}
    if store is not None:
        extra["store"] = store
    agent = create_agent(
        model=agent_settings.memory_model,
        tools=build_curator_tools(layer, store=store, optimize=True),
        system_prompt=_OPTIMIZE_PROMPT,
        **extra,
    )
    if optimize_agents is not None:
        optimize_agents[layer] = agent
    return agent


def _local_curator_graph(
    store=None,
    *,
    graphs: dict | None = None,
    curate_agents: dict | None = None,
    optimize_agents: dict | None = None,
):
    if graphs is not None:
        cached = graphs.get("local")
        if cached is not None:
            return cached
    graph = build_curator_graph(store=store)
    if graphs is not None:
        graphs["local"] = graph
    return graph


def _curator_executor(
    layer: str,
    store=None,
    *,
    settings: AgentSettings | None = None,
    cache: dict | None = None,
    graphs: dict | None = None,
    curate_agents: dict | None = None,
    optimize_agents: dict | None = None,
):
    if cache is not None:
        cached = cache.get(layer)
        if cached is not None:
            return cached
    from ai_agent.memory.managers import (
        _RemoteMemoryExecutor,
        _remote_executor_url,
        _use_remote_executor,
    )

    agent_settings = settings or get_agent_settings()
    if _use_remote_executor(agent_settings):
        executor = _RemoteMemoryExecutor(
            _remote_executor_url(), assistant_id=CURATOR_ASSISTANT_ID
        )
        if cache is not None:
            cache[layer] = executor
        return executor
    executor = _LocalCuratorExecutor(
        _local_curator_graph(
            store=store,
            graphs=graphs,
            curate_agents=curate_agents,
            optimize_agents=optimize_agents,
        ),
        store=store,
    )
    if cache is not None:
        cache[layer] = executor
    return executor


def _store_from_runtime(config=None):
    try:
        from langgraph.config import get_store

        store = get_store()
        if store is not None:
            return store
    except Exception:
        logger.debug("langgraph.config.get_store unavailable for curator", exc_info=True)
    if isinstance(config, dict):
        configurable = config.get("configurable") or {}
        nested = configurable.get("store")
        if nested is not None:
            return nested
    return None


class _LocalCuratorExecutor:
    def __init__(self, graph, store=None):
        self._graph = graph
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, payload, /, config=None, *, after_seconds: int = 0, thread_id=None):
        return self._pool.submit(self._run, payload, config, after_seconds)

    def shutdown(self, wait=False, cancel_futures=True):
        try:
            self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            self._pool.shutdown(wait=wait)

    def _run(self, payload, config, after_seconds):
        if after_seconds:
            time.sleep(after_seconds)
        try:
            self._graph.invoke(payload, config or {})
        except Exception:
            logger.exception("Local memory curator failed")
