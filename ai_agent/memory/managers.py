"""Background memory agent and ReflectionExecutor wiring."""

from __future__ import annotations

import os

from langchain.agents import create_agent

from ai_agent.conf import AgentSettings, get_agent_settings
from ai_agent.memory.namespaces import normalize_memory_config
from ai_agent.memory.tools import build_memory_tools

_MEMORY_PROMPT = """
You are a memory manager. Write only what will help a later session.

Protocol:
1. Existing memories may already be in this prompt. Search semantic and episode tools if needed to find an existing match *before* creating.
2. Prefer update or delete over insert, except for profile: never a second document.
3. Never insert a near-duplicate.
4. Never delete a document you created in this same turn. If a search now returns only that new document, keep it.
5. If nothing durable changed, make no tool calls.
6. Store only information supported by the conversation or tool results. Do not speculate.

Profile
- Exactly one document per layer. Never a second document.
- Write only standing field values defined by this layer's profile schema.
- Do not summarize the current turn into the profile.
- Do not store the current request, a chart, a date range, an allocation mix, or other in-progress work on the profile.
- Domain-specific details belong on that domain's memory, not on the supervisor profile.
- When new information arrives, patch the existing document.
- Preserve existing field values unless the conversation explicitly updates or invalidates them.
- Do not overwrite a known value with an uncertain, weaker, or missing value.
- Do not duplicate profile fields into semantic memory.

Semantic Memories
- Store durable facts that should accumulate and be searched later.
- These are collections: many documents, not a singleton.
- Extra application collection schemas are also collections, not profiles.
- Prefer one canonical fact per subject + predicate.
- Reuse the same subject and predicate when updating a fact.
- If a matching subject + predicate exists:
  - update it if the value changed
  - delete it if it is no longer true
  - do nothing if it is already correct
- Never create a paraphrased duplicate.
- Store facts, not conversation summaries.

Episodes
- Store reusable procedures, workflows, strategies, or lessons that would be useful to replay in a similar future situation, including what failed.
- An episode should capture:
  - the situation or trigger
  - the useful approach
  - relevant constraints
  - the outcome or lesson, when known
- If the attempt failed, also capture:
  - that it failed
  - the reason it failed
  - how a similar failure could be prevented next time
- A unique failure episode must be kept unless you are merging it into a
  different existing episode. Do not delete the only copy.
- Search for a similar episode before creating. After creating, do not delete
  it because search now finds it.
- Do not store a chronological log of the current turn.
- Do not store raw tool calls or incidental debugging steps.
- Prefer updating or consolidating a similar episode over creating another.

Routing rules
For each candidate memory, choose exactly one primary destination:

1. Profile:
   Use only for a standing field on this layer's profile schema (name, language, reply style, constraints, or this domain's standing knobs).

2. Semantic memory:
   Use for a durable fact that should accumulate and does not belong on the profile.

3. Episode:
   Use for a reusable way of solving or handling a situation, including a failure, its cause, and how to avoid it.

Do not store the same information in multiple places.

Never store:
- the current turn's request merely because it was asked
- greetings or filler
- temporary search scopes or date ranges
- raw tool outputs
- intermediate reasoning
- speculative conclusions
- one-off in-progress work that will not matter next session
- facts already represented accurately in another memory layer
- a playbook document. Playbooks are curator-owned; do not write them.

Before writing memory, apply these tests:

Durability:
"Is this likely to matter in a later session, after this task is done?"

Profile fit:
"Is this a standing field on the profile schema, not a summary of this turn?"

Novelty:
"Is this materially new or changed?"

Conflict:
"Would this create two competing representations of the same fact?"

If nothing durable and new remains after these checks, make no memory tool calls.
"""

_REFLECTOR_NAMESPACE = ("memories", "{user_id}")


def build_memory_graph(*, store=None, settings: AgentSettings | None = None):
    """Compiled memory agent for LangGraph Agent Server (graph id ``memory``)."""
    return build_layer_memory_agent(
        "{memory_layer}", store=store, settings=settings
    )


def build_layer_memory_agent(
    layer: str,
    *,
    store=None,
    settings: AgentSettings | None = None,
):
    from ai_agent.memory.middleware import MemoryRecallMiddleware

    agent_settings = settings or get_agent_settings()
    agent_kwargs = {
        "model": agent_settings.memory_model,
        "tools": build_memory_tools(layer),
        "system_prompt": _MEMORY_PROMPT,
        "middleware": [MemoryRecallMiddleware(layer, settings=agent_settings)],
    }
    if store is not None:
        agent_kwargs["store"] = store
    return create_agent(**agent_kwargs)


def submit_memory(
    layer: str,
    messages,
    config,
    *,
    store=None,
    settings: AgentSettings | None = None,
    cache: dict | None = None,
    agents: dict | None = None,
) -> None:
    agent_settings = settings or get_agent_settings()
    if not agent_settings.is_memory_background():
        return
    if not messages:
        return
    bound = normalize_memory_config(config, layer)
    thread_id = _thread_id_from_config(bound)
    kwargs = {
        "config": bound,
        "after_seconds": max(0, agent_settings.memory_debounce_seconds),
    }
    if thread_id:
        kwargs["thread_id"] = thread_id
    executor = _reflection_executor(
        layer,
        store=store,
        settings=agent_settings,
        cache=cache,
        agents=agents,
    )
    executor.submit({"messages": list(messages)}, **kwargs)


def _reflection_executor(
    layer: str,
    store=None,
    *,
    settings: AgentSettings | None = None,
    cache: dict | None = None,
    agents: dict | None = None,
):
    agent_settings = settings or get_agent_settings()
    if cache is not None:
        cached = cache.get(layer)
        if cached is not None:
            return cached
    if _use_remote_executor(agent_settings):
        executor = _RemoteMemoryExecutor(_remote_executor_url())
        if cache is not None:
            cache[layer] = executor
        return executor
    from langmem import ReflectionExecutor

    agent = _NamespacedReflector(
        _local_memory_agent(
            layer, store=store, settings=agent_settings, agents=agents
        )
    )
    executor = ReflectionExecutor(agent, store=store)
    if cache is not None:
        cache[layer] = executor
    return executor


def _local_memory_agent(
    layer: str,
    *,
    store=None,
    settings: AgentSettings | None = None,
    agents: dict | None = None,
):
    if agents is not None:
        cached = agents.get(layer)
        if cached is not None:
            return cached
    agent = build_layer_memory_agent(layer, store=store, settings=settings)
    if agents is not None:
        agents[layer] = agent
    return agent


def _use_remote_executor(settings: AgentSettings | None = None) -> bool:
    agent_settings = settings or get_agent_settings()
    if not agent_settings.memory_enabled:
        return False
    if agent_settings.memory_store != "platform":
        return False
    return bool(_remote_executor_url())


def _remote_executor_url() -> str:
    return (
        os.environ.get("LANGGRAPH_API_URL")
        or os.environ.get("LANGGRAPH_RUNTIME_URL")
        or ""
    ).strip()


def _thread_id_from_config(config) -> str | None:
    if not isinstance(config, dict):
        return None
    value = (config.get("configurable") or {}).get("thread_id")
    if value in (None, ""):
        return None
    return str(value)


class _NamespacedReflector:
    """Adapter so LocalReflectionExecutor can wrap create_agent graphs.

    LangMem requires ``reflector.namespace`` to be a NamespaceTemplate. The
    compiled supervisor/memory agent does not define that attribute.
    """

    def __init__(self, agent):
        from langmem.utils import NamespaceTemplate

        self._agent = agent
        self.namespace = NamespaceTemplate(_REFLECTOR_NAMESPACE)

    def invoke(self, payload, config=None, **kwargs):
        if config is None:
            return self._agent.invoke(payload)
        return self._agent.invoke(payload, config, **kwargs)

    async def ainvoke(self, payload, config=None, **kwargs):
        if config is None:
            return await self._agent.ainvoke(payload)
        return await self._agent.ainvoke(payload, config, **kwargs)


class _RemoteMemoryExecutor:
    """Remote ReflectionExecutor that keeps user_id and memory_layer in config."""

    def __init__(self, url: str, *, assistant_id: str = "memory"):
        from concurrent.futures import ThreadPoolExecutor

        from langgraph_sdk import get_sync_client

        self._assistant_id = assistant_id
        self._client = get_sync_client(url=url)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def shutdown(self, wait=False, cancel_futures=True):
        try:
            self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            self._pool.shutdown(wait=wait)

    def submit(self, payload, /, config=None, *, after_seconds: int = 0, thread_id=None):
        configurable = dict((config or {}).get("configurable") or {})
        resolved_thread = thread_id or configurable.get("thread_id")
        if resolved_thread:
            configurable["thread_id"] = str(resolved_thread)
        return self._pool.submit(
            self._run,
            payload,
            configurable,
            after_seconds,
            str(resolved_thread) if resolved_thread else None,
        )

    def _run(self, payload, configurable, after_seconds, thread_id):
        self._client.runs.create(
            thread_id=thread_id,
            assistant_id=self._assistant_id,
            input=payload,
            config={"configurable": configurable},
            multitask_strategy="rollback",
            after_seconds=after_seconds,
            if_not_exists="create",
        )
        return None
