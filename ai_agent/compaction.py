"""Conversation compaction via LangChain SummarizationMiddleware.

LangChain stores the retry-wrapped summarizer on the private ``_summary_model``
attribute after ``with_retry()``. There is no public quiet-summarizer hook, so
this module subclasses the middleware and wraps that runnable after init.

The wrapper injects CopilotKit ``emit_messages=False`` / ``emit_tool_calls=False``
(public ``copilotkit_customize_config``) and the unprefixed AG-UI metadata keys
so summarizer tokens are not streamed into chat UIs.
"""

from __future__ import annotations

import logging

from copilotkit.langgraph import copilotkit_customize_config
from langchain.agents.middleware import SummarizationMiddleware

from ai_agent.conf import get_agent_settings

logger = logging.getLogger(__name__)

AGUI_EMIT_MESSAGES = "emit-messages"
AGUI_EMIT_TOOL_CALLS = "emit-tool-calls"


def build_compaction_middleware(agent_settings=None):
    agent_settings = agent_settings or get_agent_settings()
    return _QuietSummarizationMiddleware(
        model=agent_settings.compaction_model,
        trigger=(
            agent_settings.compaction_trigger_type,
            agent_settings.compaction_trigger,
        ),
        keep=(
            agent_settings.compaction_keep_type,
            agent_settings.compaction_keep,
        ),
    )


class _QuietSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that suppresses CopilotKit/AG-UI token streaming.

    Must wrap ``_summary_model`` *after* ``super().__init__`` so the quiet
    wrapper sits on the retry runnable LangChain actually invokes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _silence_summarizer_stream(self)


def _silence_summarizer_stream(middleware):
    """Hide summarizer tokens from CopilotKit/AG-UI chat streaming."""
    inner = getattr(middleware, "_summary_model", None)
    if inner is None:
        logger.warning(
            "SummarizationMiddleware has no _summary_model; "
            "summarizer tokens may stream to CopilotKit/AG-UI"
        )
        return middleware
    middleware._summary_model = _QuietSummaryModel(inner)
    return middleware


def _quiet_summary_config(config=None):
    """Mark the summarizer call so CopilotKit and ag-ui-langgraph drop it.

    CopilotKit's wrapper filters ``copilotkit:emit-messages``; ag-ui-langgraph
    filters the unprefixed ``emit-messages`` key. Set both.
    """
    base = dict(config) if isinstance(config, dict) else {}
    metadata = dict(base.get("metadata") or {})
    base["metadata"] = metadata
    customized = copilotkit_customize_config(
        base,
        emit_messages=False,
        emit_tool_calls=False,
    )
    metadata = dict(customized.get("metadata") or {})
    metadata[AGUI_EMIT_MESSAGES] = False
    metadata[AGUI_EMIT_TOOL_CALLS] = False
    return {**customized, "metadata": metadata}


class _QuietSummaryModel:
    def __init__(self, inner):
        self._inner = inner

    def invoke(self, prompt, config=None, **kwargs):
        return self._inner.invoke(
            prompt, config=_quiet_summary_config(config), **kwargs
        )

    async def ainvoke(self, prompt, config=None, **kwargs):
        return await self._inner.ainvoke(
            prompt, config=_quiet_summary_config(config), **kwargs
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)
