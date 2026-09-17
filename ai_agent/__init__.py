"""Opt-in DRF-to-LangGraph agent bridge.

Extractable surface: expose, discovery, schema, executor, tools, graph, context.
Product wiring (chat view, Django settings) stays in this app.

``ModelAgent`` / ``agent_tool`` are lazy so importing this package during
Django ``AppConfig`` population does not pull DRF views before apps are ready.
"""

from ai_agent.expose import agent_exclude, agent_expose

__all__ = ["agent_expose", "agent_exclude", "ModelAgent", "agent_tool"]


def __getattr__(name):
    if name in {"ModelAgent", "agent_tool"}:
        from ai_agent.agents import ModelAgent, agent_tool

        mapping = {"ModelAgent": ModelAgent, "agent_tool": agent_tool}
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
