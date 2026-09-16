"""Owned agent collaborators: settings, endpoints, checkpointer, store, executors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Sequence

from ai_agent.conf import AgentSettings, get_agent_settings
from ai_agent.discovery import DiscoveredEndpoint, discover_endpoints

logger = logging.getLogger(__name__)

_MISSING = object()


def _shutdown_executor(executor) -> None:
    shutdown = getattr(executor, "shutdown", None)
    if not shutdown:
        return
    try:
        shutdown(wait=False, cancel_futures=True)
    except TypeError:
        logger.debug("Executor shutdown without cancel_futures support")
        shutdown(wait=False)


@dataclass
class AgentRuntime:
    settings: AgentSettings
    endpoints: list[DiscoveredEndpoint]
    checkpointer: object | None
    store: object | None
    extra_middleware: tuple = ()
    extra_agents: tuple = ()

    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    _supervisor: object | None = field(default=None, repr=False, compare=False)
    _memory_executors: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _memory_agents: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _curator_executors: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _curator_graphs: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _curate_agents: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )
    _optimize_agents: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        *,
        settings: AgentSettings | None = None,
        endpoints: Sequence[DiscoveredEndpoint] | None = None,
        checkpointer: Any = _MISSING,
        store: Any = _MISSING,
        extra_middleware: Sequence | None = None,
        extra_agents: Sequence | None = None,
    ) -> AgentRuntime:
        from ai_agent.agents import resolve_model_agents
        from ai_agent.graph import build_checkpointer
        from ai_agent.memory.store import build_memory_store

        agent_settings = settings or get_agent_settings()
        resolved_endpoints = list(
            endpoints if endpoints is not None else discover_endpoints()
        )
        resolved_checkpointer = (
            build_checkpointer() if checkpointer is _MISSING else checkpointer
        )
        resolved_store = (
            build_memory_store(agent_settings) if store is _MISSING else store
        )
        resolved_extra_agents = tuple(
            resolve_model_agents(
                settings=agent_settings,
                extra_agents=extra_agents,
                endpoints=resolved_endpoints,
            )
        )
        return cls(
            settings=agent_settings,
            endpoints=resolved_endpoints,
            checkpointer=resolved_checkpointer,
            store=resolved_store,
            extra_middleware=tuple(extra_middleware or ()),
            extra_agents=resolved_extra_agents,
        )

    def supervisor(self, **kwargs):
        from ai_agent.graph import build_supervisor

        if kwargs:
            return self._build_supervisor(**kwargs)
        with self._lock:
            if self._supervisor is None:
                self._supervisor = self._build_supervisor()
            return self._supervisor

    def _build_supervisor(self, **kwargs):
        from ai_agent.graph import build_supervisor

        kwargs.setdefault("settings", self.settings)
        kwargs.setdefault("endpoints", self.endpoints)
        kwargs.setdefault("checkpointer", self.checkpointer)
        kwargs.setdefault("store", self.store)
        kwargs.setdefault("extra_middleware", self.extra_middleware)
        kwargs.setdefault("extra_agents", self.extra_agents)
        kwargs.setdefault("agent_runtime", self)
        return build_supervisor(**kwargs)

    def domain_agent(self, app_label: str, **kwargs):
        from ai_agent.graph import build_domain_agent

        kwargs.setdefault("settings", self.settings)
        kwargs.setdefault("endpoints", self.endpoints)
        kwargs.setdefault("checkpointer", self.checkpointer)
        kwargs.setdefault("store", self.store)
        kwargs.setdefault("extra_middleware", self.extra_middleware)
        kwargs.setdefault("extra_agents", self.extra_agents)
        kwargs.setdefault("agent_runtime", self)
        return build_domain_agent(app_label, **kwargs)

    def memory_graph(self, **kwargs):
        from ai_agent.memory.managers import build_memory_graph

        kwargs.setdefault("settings", self.settings)
        kwargs.setdefault("store", self.store)
        return build_memory_graph(**kwargs)

    def curator_graph(self, **kwargs):
        from ai_agent.memory.curator import build_curator_graph

        kwargs.setdefault("store", self.store)
        return build_curator_graph(**kwargs)

    def submit_memory(self, layer: str, messages, config, *, store=None) -> None:
        from ai_agent.memory.managers import submit_memory

        resolved = self.store if store is None else store
        with self._lock:
            submit_memory(
                layer,
                messages,
                config,
                store=resolved,
                settings=self.settings,
                cache=self._memory_executors,
                agents=self._memory_agents,
            )

    def submit_curator(self, layer: str, config, *, store=None) -> None:
        from ai_agent.memory.curator import submit_curator

        resolved = self.store if store is None else store
        with self._lock:
            submit_curator(
                layer,
                config,
                store=resolved,
                settings=self.settings,
                cache=self._curator_executors,
                graphs=self._curator_graphs,
                curate_agents=self._curate_agents,
                optimize_agents=self._optimize_agents,
            )

    def run_curator(self, user_id: str, layer: str, *, store=None, config=None) -> dict:
        from ai_agent.memory.curator import run_curator

        resolved = self.store if store is None else store
        return run_curator(
            user_id,
            layer,
            store=resolved,
            config=config,
            settings=self.settings,
            graphs=self._curator_graphs,
            curate_agents=self._curate_agents,
            optimize_agents=self._optimize_agents,
        )

    def close(self) -> None:
        with self._lock:
            for executor in list(self._memory_executors.values()) + list(
                self._curator_executors.values()
            ):
                _shutdown_executor(executor)
            self._memory_executors.clear()
            self._memory_agents.clear()
            self._curator_executors.clear()
            self._curator_graphs.clear()
            self._curate_agents.clear()
            self._optimize_agents.clear()
            self._supervisor = None
