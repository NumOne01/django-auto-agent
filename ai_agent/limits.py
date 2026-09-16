"""Per-run max-turn and timeout guards for create_agent loops."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue

from ai_agent.conf import get_agent_settings

TIMEOUT_MESSAGE = (
    "I stopped because this request took too long. "
    "Please try again with a narrower question."
)


class RunTimeoutState(AgentState):
    run_started_at: NotRequired[Annotated[float, UntrackedValue, PrivateStateAttr]]


class RunTimeoutMiddleware(AgentMiddleware[RunTimeoutState]):
    """End the agent run after a wall-clock budget, before the next model call."""

    state_schema = RunTimeoutState

    def __init__(
        self,
        timeout_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self._clock = clock

    def before_agent(self, state: RunTimeoutState, runtime) -> dict[str, Any] | None:
        if state.get("run_started_at") is None:
            return {"run_started_at": self._clock()}
        return None

    async def abefore_agent(
        self, state: RunTimeoutState, runtime
    ) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: RunTimeoutState, runtime) -> dict[str, Any] | None:
        started = state.get("run_started_at")
        now = self._clock()
        if started is None:
            return {"run_started_at": now}
        if now - started >= self.timeout_seconds:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=TIMEOUT_MESSAGE)],
            }
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: RunTimeoutState, runtime
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)


def build_limit_middleware(agent_settings=None) -> list:
    agent_settings = agent_settings or get_agent_settings()
    items = []
    if agent_settings.max_turn:
        items.append(
            ModelCallLimitMiddleware(
                run_limit=agent_settings.max_turn,
                exit_behavior="end",
            )
        )
    if agent_settings.timeout_seconds:
        items.append(RunTimeoutMiddleware(agent_settings.timeout_seconds))
    return items


# create_agent compiles with this when max_turn is off. Nested tool invokes
# must set it explicitly: parent tool config defaults to LangChain's 25.
_CREATE_AGENT_RECURSION_LIMIT = 9_999
# Each agent turn is more than model+tools: middleware hooks are graph nodes.
_GRAPH_STEPS_PER_TURN = 8
_GRAPH_STEPS_OVERHEAD = 16


def recursion_limit_for_max_turn(max_turn: int) -> int:
    return max_turn * _GRAPH_STEPS_PER_TURN + _GRAPH_STEPS_OVERHEAD


def recursion_limit_for_settings(agent_settings=None) -> int:
    max_turn = (agent_settings or get_agent_settings()).max_turn
    if max_turn:
        return recursion_limit_for_max_turn(max_turn)
    return _CREATE_AGENT_RECURSION_LIMIT


def bind_recursion_limit(config: dict, agent_settings=None) -> dict:
    """Give a nested invoke its own graph step budget.

    Parent tool configs carry LangChain's default ``recursion_limit`` of 25,
    which overrides the child graph's ``with_config`` and raises
    ``GraphRecursionError`` after a few middleware-heavy turns.
    """
    return {
        **config,
        "recursion_limit": recursion_limit_for_settings(agent_settings),
    }


def apply_run_limits(graph, agent_settings=None):
    agent_settings = agent_settings or get_agent_settings()
    if not agent_settings.max_turn:
        return graph
    with_config = getattr(graph, "with_config", None)
    if not callable(with_config):
        return graph
    return with_config(
        {"recursion_limit": recursion_limit_for_settings(agent_settings)}
    )
