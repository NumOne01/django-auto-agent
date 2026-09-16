"""Isolated AgentRuntime helpers for live evals (and tests)."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from contextlib import contextmanager

from ai_agent.runtime import AgentRuntime


class _ImmediateExecutor(Executor):
    """Run ToolNode work on this thread so Django TestCase SQLite stays visible."""

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def map(self, fn, *iterables):
        return [fn(*args) for args in zip(*iterables, strict=False)]


@contextmanager
def _immediate_executor(_config=None):
    executor = _ImmediateExecutor()
    try:
        yield executor
    finally:
        executor.shutdown(wait=False)


def make_test_runtime(**kwargs) -> AgentRuntime:
    """Build an isolated AgentRuntime from the current Django AI_AGENT settings."""
    from langgraph.checkpoint.memory import InMemorySaver

    from ai_agent.conf import get_agent_settings

    kwargs.setdefault("checkpointer", InMemorySaver())
    if "settings" not in kwargs:
        kwargs["settings"] = get_agent_settings()
    return AgentRuntime.create(**kwargs)
