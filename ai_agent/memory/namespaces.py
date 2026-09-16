"""LangGraph Store namespaces for supervisor and per-app memory."""

from __future__ import annotations

SUPERVISOR_LAYER = "supervisor"
PROFILE_KEY = "default"
PLAYBOOK_KEY = PROFILE_KEY
PROMPT_KEY = PROFILE_KEY

_PROFILE = "profile"
_SEMANTIC = "semantic"
_EPISODES = "episodes"
_PLAYBOOK = "playbook"
_PROMPT = "prompt"


def memory_namespaces(layer: str) -> dict[str, tuple[str, ...]]:
    """Return profile/semantic/episode/playbook/prompt namespace templates.

    ``layer`` is either an app label, ``supervisor``, or ``{memory_layer}``
    for the shared remote memory graph.
    """
    return {
        "profile": ("memories", "{user_id}", layer, _PROFILE),
        "semantic": ("memories", "{user_id}", layer, _SEMANTIC),
        "episodes": ("memories", "{user_id}", layer, _EPISODES),
        "playbook": ("memories", "{user_id}", layer, _PLAYBOOK),
        "prompt": ("memories", "{user_id}", layer, _PROMPT),
    }


def global_prompt_namespace(layer: str) -> tuple[str, ...]:
    """Shared overlay for every user on ``layer`` (no ``user_id``)."""
    return ("prompts", "global", layer)


def agent_memory_layers() -> list[str]:
    """Supervisor plus every app that exposes agent APIs or a memory profile.

    ``ModelAgent`` names from ``AI_AGENT.EXTRA_AGENTS`` are included as well.
    """
    from django.apps import apps

    from ai_agent.agents import model_agent_names

    labels = [SUPERVISOR_LAYER]
    for config in apps.get_app_configs():
        if config.label == "ai_agent":
            continue
        if getattr(config, "agent_expose", False) or getattr(
            config, "agent_memory_profile", None
        ):
            labels.append(config.label)
    for name in model_agent_names():
        if name not in labels:
            labels.append(name)
    return labels


def bind_namespace(namespace: tuple[str, ...], *, user_id: str) -> tuple[str, ...]:
    return tuple(
        str(user_id) if part == "{user_id}" else part for part in namespace
    )


def user_id_from_config(config) -> str | None:
    """Trusted LangGraph auth identity, then test-only ``user_id``.

    Client-supplied ``configurable.user_id`` is ignored outside Django
    ``TESTING`` so a forged id cannot select another user's memory.
    """
    if not isinstance(config, dict):
        return None
    cfg = dict(config.get("configurable") or {})
    extra = config.get("context")
    if isinstance(extra, dict):
        cfg = {**cfg, **extra}

    auth_id = _auth_user_id(cfg)
    if auth_id is not None:
        return auth_id

    from django.conf import settings

    if not getattr(settings, "TESTING", False):
        return None
    user_id = cfg.get("user_id")
    if user_id not in (None, ""):
        return str(user_id)
    return None


def _auth_user_id(cfg: dict) -> str | None:
    user_id = cfg.get("langgraph_auth_user_id")
    if user_id not in (None, ""):
        return str(user_id)
    auth_user = cfg.get("langgraph_auth_user")
    if isinstance(auth_user, dict):
        identity = auth_user.get("identity")
        if identity not in (None, ""):
            return str(identity)
    identity = getattr(auth_user, "identity", None)
    if identity not in (None, ""):
        return str(identity)
    return None


def normalize_memory_config(config, layer: str) -> dict:
    """Ensure ``user_id`` and ``memory_layer`` are set for LangMem templates."""
    if not isinstance(config, dict):
        config = {}
    configurable = dict(config.get("configurable") or {})
    user_id = user_id_from_config({**config, "configurable": configurable})
    if user_id:
        configurable["user_id"] = user_id
    else:
        configurable.pop("user_id", None)
    auth_id = _auth_user_id(configurable)
    if auth_id:
        configurable["langgraph_auth_user_id"] = auth_id
    configurable["memory_layer"] = layer
    return {**config, "configurable": configurable}
