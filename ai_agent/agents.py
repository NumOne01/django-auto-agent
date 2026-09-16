"""Host-defined specialists that do not require DRF endpoints."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from asgiref.sync import async_to_sync, sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt

from ai_agent.conf import AgentSettings, get_agent_settings
from ai_agent.tools import _resume_is_approved

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_RESERVED_NAMES = frozenset({"supervisor"})
_DECLINED = "The user declined this action."


@dataclass(frozen=True)
class ModelToolSpec:
    name: str
    description: str
    confirm: bool
    method_name: str


def agent_tool(func=None, *, name=None, description=None, confirm=False):
    """Mark a ``ModelAgent`` method as a specialist tool.

    ``confirm`` opts this tool into mutation HITL. ``CONFIRM_MUTATIONS`` does
    not apply because there is no HTTP method to infer.
    """

    def apply(method):
        method._agent_tool = True
        method._agent_tool_name = name
        method._agent_tool_description = description
        method._agent_tool_confirm = bool(confirm)
        return method

    if func is not None:
        return apply(func)
    return apply


class ModelAgent:
    """Base class for host specialists backed by Python tools (including Django ORM).

    Subclass, set ``name`` and ``description``, decorate methods with
    ``@agent_tool``, and list the class in ``AI_AGENT["EXTRA_AGENTS"]``.
    Instances are process-long; do not store per-request state on ``self``.
    Use ``ai_agent.context.get_current_user()`` inside tools.
    """

    name: str = ""
    description: str = ""
    memory_profile = None
    memory_collections = None
    memory_episode = None

    def validated_name(self) -> str:
        label = str(self.name or "").strip()
        if not label:
            raise ImproperlyConfigured(
                f"{type(self).__name__} must set a non-empty name."
            )
        if label in _RESERVED_NAMES:
            raise ImproperlyConfigured(
                f"ModelAgent name {label!r} is reserved."
            )
        if not _NAME_RE.fullmatch(label):
            raise ImproperlyConfigured(
                f"ModelAgent name {label!r} must match {_NAME_RE.pattern}."
            )
        return label

    def validated_description(self) -> str:
        blurb = str(self.description or "").strip()
        if not blurb:
            raise ImproperlyConfigured(
                f"{type(self).__name__} must set a non-empty description."
            )
        return blurb

    def tool_specs(self) -> list[ModelToolSpec]:
        specs: list[ModelToolSpec] = []
        seen_names: set[str] = set()
        for method_name, func in _tool_methods(type(self)).items():
            spec = _spec_for_method(method_name, func)
            if spec.name in seen_names:
                raise ImproperlyConfigured(
                    f"ModelAgent {self.validated_name()!r} has duplicate tool "
                    f"name {spec.name!r}."
                )
            seen_names.add(spec.name)
            specs.append(spec)
        return specs

    def build_tools(self) -> list[StructuredTool]:
        tools: list[StructuredTool] = []
        for spec in self.tool_specs():
            bound = getattr(self, spec.method_name)
            tools.append(_structured_tool(bound, spec))
        return tools


def resolve_model_agents(
    *,
    settings: Optional[AgentSettings] = None,
    extra_agents: Optional[Sequence] = None,
    endpoints: Optional[Sequence] = None,
    check_endpoint_collisions: bool = True,
) -> list[ModelAgent]:
    """Instantiate and validate ``AI_AGENT.EXTRA_AGENTS`` (or an explicit list)."""
    agent_settings = settings or get_agent_settings()
    raw = extra_agents if extra_agents is not None else agent_settings.extra_agents
    agents = [_instantiate_model_agent(item) for item in raw]
    names: list[str] = []
    for agent in agents:
        label = agent.validated_name()
        agent.validated_description()
        if label in names:
            raise ImproperlyConfigured(
                f"Duplicate ModelAgent name {label!r} in AI_AGENT.EXTRA_AGENTS."
            )
        names.append(label)
        agent.tool_specs()
    if check_endpoint_collisions:
        if endpoints is None:
            from ai_agent.discovery import discover_endpoints

            endpoints = discover_endpoints()
        _reject_collisions(agents, endpoints)
    return agents


def model_agent_names(*, settings: Optional[AgentSettings] = None) -> list[str]:
    return [
        agent.validated_name()
        for agent in resolve_model_agents(
            settings=settings, check_endpoint_collisions=False
        )
    ]


def model_agent_for_layer(
    layer: str, *, settings: Optional[AgentSettings] = None
) -> ModelAgent | None:
    for agent in resolve_model_agents(
        settings=settings, check_endpoint_collisions=False
    ):
        if agent.validated_name() == layer:
            return agent
    return None


def _instantiate_model_agent(item) -> ModelAgent:
    if isinstance(item, ModelAgent):
        return item
    loaded = item
    if isinstance(item, str):
        path = item.strip()
        if not path:
            raise ImproperlyConfigured(
                "AI_AGENT.EXTRA_AGENTS contains an empty path."
            )
        try:
            loaded = import_string(path)
        except ImportError as exc:
            raise ImproperlyConfigured(
                f"AI_AGENT.EXTRA_AGENTS {path!r} could not be imported."
            ) from exc
    if inspect.isclass(loaded):
        if not issubclass(loaded, ModelAgent):
            raise ImproperlyConfigured(
                f"AI_AGENT.EXTRA_AGENTS {loaded!r} must subclass ModelAgent."
            )
        try:
            return loaded()
        except ImproperlyConfigured:
            raise
        except Exception as exc:
            raise ImproperlyConfigured(
                f"AI_AGENT.EXTRA_AGENTS {loaded!r} could not be instantiated."
            ) from exc
    raise ImproperlyConfigured(
        f"AI_AGENT.EXTRA_AGENTS {item!r} must be a ModelAgent subclass, "
        "instance, or dotted path."
    )


def _reject_collisions(agents: Sequence[ModelAgent], endpoints: Sequence) -> None:
    exposed = {item.app_label for item in endpoints}
    operation_ids = {item.operation_id for item in endpoints}
    for agent in agents:
        label = agent.validated_name()
        if label in exposed:
            raise ImproperlyConfigured(
                f"ModelAgent name {label!r} collides with an exposed Django app."
            )
        for spec in agent.tool_specs():
            if spec.name in operation_ids:
                raise ImproperlyConfigured(
                    f"ModelAgent tool {spec.name!r} collides with API operationId "
                    f"{spec.name!r}."
                )


def _tool_methods(cls: type) -> dict[str, object]:
    found: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        for name, value in base.__dict__.items():
            if inspect.isfunction(value):
                found[name] = value
    return {
        name: value
        for name, value in found.items()
        if getattr(value, "_agent_tool", False)
    }


def _spec_for_method(method_name: str, func) -> ModelToolSpec:
    raw_name = getattr(func, "_agent_tool_name", None) or method_name
    name = str(raw_name).strip()
    if not _NAME_RE.fullmatch(name):
        raise ImproperlyConfigured(
            f"ModelAgent tool name {name!r} must match {_NAME_RE.pattern}."
        )
    description = str(
        getattr(func, "_agent_tool_description", None) or inspect.getdoc(func) or name
    ).strip()
    confirm = bool(getattr(func, "_agent_tool_confirm", False))
    return ModelToolSpec(
        name=name,
        description=description,
        confirm=confirm,
        method_name=method_name,
    )


def _structured_tool(bound, spec: ModelToolSpec) -> StructuredTool:
    is_async = inspect.iscoroutinefunction(bound)

    @functools.wraps(bound)
    def _run(*args, **kwargs):
        arguments = _bound_arguments(bound, args, kwargs)
        declined = _confirm_model_tool(spec, arguments)
        if declined is not None:
            return declined
        if is_async:
            result = async_to_sync(bound)(*args, **kwargs)
        else:
            result = bound(*args, **kwargs)
        return _truncate_tool_result(result)

    @functools.wraps(bound)
    async def _arun(*args, **kwargs):
        arguments = _bound_arguments(bound, args, kwargs)
        declined = _confirm_model_tool(spec, arguments)
        if declined is not None:
            return declined
        if is_async:
            result = await bound(*args, **kwargs)
        else:
            result = await sync_to_async(bound, thread_sensitive=True)(*args, **kwargs)
        return _truncate_tool_result(result)

    _run.__name__ = spec.name
    _run.__doc__ = spec.description
    _arun.__name__ = spec.name
    _arun.__doc__ = spec.description
    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=spec.name,
        description=spec.description,
    )


def _bound_arguments(bound, args, kwargs) -> dict:
    try:
        bound_args = inspect.signature(bound).bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)
    except TypeError:
        return dict(kwargs)


def _confirm_model_tool(spec: ModelToolSpec, arguments: dict) -> str | None:
    if not spec.confirm:
        return None
    approved = interrupt(
        {
            "action": spec.name,
            "args": arguments,
            "summary": spec.description,
        }
    )
    if not _resume_is_approved(approved):
        return _DECLINED
    return None


def _truncate_tool_result(result) -> str:
    text = result if isinstance(result, str) else str(result)
    max_chars = get_agent_settings().max_tool_response_chars
    total = len(text)
    if total > max_chars:
        return (
            text[:max_chars]
            + f"…[truncated, showing {max_chars} of {total} chars; "
            "do not invent omitted fields]"
        )
    return text
