"""Read ``settings.AI_AGENT`` into a frozen ``AgentSettings`` at call time.

``get_agent_settings()`` rebuilds from Django's lazy settings (or an explicit
``raw`` dict) on every call. Long-lived graphs still snapshot the object they
were built with.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

from django.conf import settings

_VALID_MEMORY_MODES = {"hot", "background"}
_VALID_MEMORY_STORES = {"platform", "memory", "postgres", "redis"}
_VALID_CONTEXT_KINDS = {"tokens", "messages", "fraction"}
_DEFAULT_TRIGGER_TYPE = "tokens"
_DEFAULT_TRIGGER_TOKENS = 8000
_DEFAULT_TRIGGER_FRACTION = 0.8
_DEFAULT_KEEP_TYPE = "messages"
_DEFAULT_KEEP_MESSAGES = 20
_DEFAULT_KEEP_TOKENS = 4000
_DEFAULT_KEEP_FRACTION = 0.3
_DEFAULT_MAX_TURN = 25
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_HTTP_THROTTLE = "60/minute"
_DEFAULT_PLATFORM_NAME = "this product"
_DEFAULT_ENTITY_TERMS = "values, records, or identifiers"


@dataclass(frozen=True)
class PlatformConfig:
    """Host product vocabulary injected into prompts and overlay guardrails."""

    platform_name: str = _DEFAULT_PLATFORM_NAME
    domains: tuple[str, ...] = ()
    entity_terms: str = _DEFAULT_ENTITY_TERMS
    markets_out_of_scope: tuple[str, ...] = ()
    mutation_terms: tuple[str, ...] = ()
    pii_terms: tuple[str, ...] = ()
    currency_tokens: tuple[str, ...] = ()
    phone_patterns: tuple[str, ...] = ()
    extra_pii_patterns: tuple[str, ...] = ()
    extra_downgrade_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentSettings:
    supervisor_model: str
    subagent_model: str
    confirm_mutations: bool
    max_tool_response_chars: int
    studio_user_phone: str
    memory_enabled: bool
    memory_mode: str
    memory_store: str
    memory_model: str
    memory_query_model: str
    memory_embeddings: str
    memory_embedding_dims: int
    memory_debounce_seconds: int
    memory_search_limit: int
    memory_curator_enabled: bool
    memory_curator_delay_seconds: int
    memory_fact_cap: int
    memory_episode_cap: int
    memory_collection_cap: int
    memory_playbook_rule_cap: int
    memory_prompt_optimizer_enabled: bool
    memory_prompt_optimizer_model: str
    memory_prompt_optimizer_local_max_chars: int
    memory_prompt_optimizer_global_max_chars: int
    memory_prompt_optimizer_trajectory_cap: int
    memory_prompt_optimizer_min_new_episodes: int
    memory_prompt_optimizer_min_global_users: int
    memory_prompt_optimizer_max_namespaces: int
    compaction_enabled: bool
    compaction_model: str
    compaction_trigger_type: str
    compaction_trigger: float
    compaction_keep_type: str
    compaction_keep: float
    max_turn: int | None
    timeout_seconds: float | None
    http_throttle: str
    http_trusted_proxy_count: int
    authenticate_token: object
    eval_module: str
    middleware: tuple
    extra_agents: tuple
    platform: PlatformConfig

    def is_memory_hot(self) -> bool:
        return self.memory_enabled and self.memory_mode == "hot"

    def is_memory_background(self) -> bool:
        return self.memory_enabled and self.memory_mode == "background"


def get_agent_settings(*, raw: dict | None = None) -> AgentSettings:
    if raw is None:
        raw = getattr(settings, "AI_AGENT", None) or {}
    supervisor_model = str(raw.get("SUPERVISOR_MODEL") or "").strip()
    subagent_model = str(raw.get("SUBAGENT_MODEL") or "").strip()
    memory_mode = str(raw.get("MEMORY_MODE") or "background").strip().lower()
    if memory_mode not in _VALID_MEMORY_MODES:
        memory_mode = "background"
    memory_store = str(raw.get("MEMORY_STORE") or "platform").strip().lower()
    if memory_store not in _VALID_MEMORY_STORES:
        memory_store = "platform"
    trigger_type = _as_context_kind(
        raw.get("COMPACTION_TRIGGER_TYPE"), _DEFAULT_TRIGGER_TYPE
    )
    keep_type = _as_context_kind(raw.get("COMPACTION_KEEP_TYPE"), _DEFAULT_KEEP_TYPE)
    return AgentSettings(
        supervisor_model=supervisor_model,
        subagent_model=subagent_model,
        confirm_mutations=raw.get("CONFIRM_MUTATIONS", True),
        max_tool_response_chars=int(raw.get("MAX_TOOL_RESPONSE_CHARS", 8000)),
        studio_user_phone=str(raw.get("STUDIO_USER_PHONE") or "").strip(),
        memory_enabled=_as_bool(raw.get("MEMORY_ENABLED", False)),
        memory_mode=memory_mode,
        memory_store=memory_store,
        memory_model=str(raw.get("MEMORY_MODEL") or subagent_model).strip(),
        memory_query_model=str(raw.get("MEMORY_QUERY_MODEL") or "").strip(),
        memory_embeddings=str(raw.get("MEMORY_EMBEDDINGS") or "").strip(),
        memory_embedding_dims=int(raw.get("MEMORY_EMBEDDING_DIMS", 1536)),
        memory_debounce_seconds=int(raw.get("MEMORY_DEBOUNCE_SECONDS", 30)),
        memory_search_limit=int(raw.get("MEMORY_SEARCH_LIMIT", 5)),
        memory_curator_enabled=_curator_enabled(raw),
        memory_curator_delay_seconds=int(
            raw.get("MEMORY_CURATOR_DELAY_SECONDS", 120)
        ),
        memory_fact_cap=_as_positive_int(raw.get("MEMORY_FACT_CAP"), 30),
        memory_episode_cap=_as_positive_int(raw.get("MEMORY_EPISODE_CAP"), 15),
        memory_collection_cap=_as_positive_int(
            raw.get("MEMORY_COLLECTION_CAP"), 20
        ),
        memory_playbook_rule_cap=_as_positive_int(
            raw.get("MEMORY_PLAYBOOK_RULE_CAP"), 8
        ),
        memory_prompt_optimizer_enabled=_optimizer_enabled(raw),
        memory_prompt_optimizer_model=str(
            raw.get("MEMORY_PROMPT_OPTIMIZER_MODEL") or ""
        ).strip(),
        memory_prompt_optimizer_local_max_chars=_as_positive_int(
            raw.get("MEMORY_PROMPT_OPTIMIZER_LOCAL_MAX_CHARS"), 1500
        ),
        memory_prompt_optimizer_global_max_chars=_as_positive_int(
            raw.get("MEMORY_PROMPT_OPTIMIZER_GLOBAL_MAX_CHARS"), 4000
        ),
        memory_prompt_optimizer_trajectory_cap=_as_positive_int(
            raw.get("MEMORY_PROMPT_OPTIMIZER_TRAJECTORY_CAP"), 20
        ),
        memory_prompt_optimizer_min_new_episodes=_as_positive_int(
            raw.get("MEMORY_PROMPT_OPTIMIZER_MIN_NEW_EPISODES"), 3
        ),
        memory_prompt_optimizer_min_global_users=_as_positive_int(
            raw.get("MEMORY_PROMPT_OPTIMIZER_MIN_GLOBAL_USERS"), 2
        ),
        memory_prompt_optimizer_max_namespaces=_as_positive_int(
            raw.get("MEMORY_PROMPT_OPTIMIZER_MAX_NAMESPACES"), 200
        ),
        compaction_enabled=_as_bool(raw.get("COMPACTION_ENABLED", False)),
        compaction_model=str(raw.get("COMPACTION_MODEL") or subagent_model).strip(),
        compaction_trigger_type=trigger_type,
        compaction_trigger=_as_context_value(
            trigger_type,
            raw.get("COMPACTION_TRIGGER", _DEFAULT_TRIGGER_TOKENS),
            tokens_default=_DEFAULT_TRIGGER_TOKENS,
            messages_default=_DEFAULT_TRIGGER_TOKENS,
            fraction_default=_DEFAULT_TRIGGER_FRACTION,
        ),
        compaction_keep_type=keep_type,
        compaction_keep=_as_context_value(
            keep_type,
            raw.get("COMPACTION_KEEP", _DEFAULT_KEEP_MESSAGES),
            tokens_default=_DEFAULT_KEEP_TOKENS,
            messages_default=_DEFAULT_KEEP_MESSAGES,
            fraction_default=_DEFAULT_KEEP_FRACTION,
        ),
        max_turn=_as_optional_positive_int(
            raw.get("MAX_TURN"), _DEFAULT_MAX_TURN
        ),
        timeout_seconds=_as_optional_positive_float(
            raw.get("TIMEOUT_SECONDS"), _DEFAULT_TIMEOUT_SECONDS
        ),
        http_throttle=_as_http_throttle(raw.get("HTTP_THROTTLE")),
        http_trusted_proxy_count=_as_non_negative_int(
            raw.get("HTTP_TRUSTED_PROXY_COUNT"), 0
        ),
        authenticate_token=_as_authenticate_token(raw.get("AUTHENTICATE_TOKEN")),
        eval_module=str(raw.get("EVAL_MODULE") or "").strip(),
        middleware=_as_middleware_tuple(raw.get("MIDDLEWARE")),
        extra_agents=_as_middleware_tuple(raw.get("EXTRA_AGENTS")),
        platform=_platform_config(raw),
    )


def join_en(items: Sequence[str]) -> str:
    """Join terms as ``a``, ``a and b``, or ``a, b, and c``."""
    parts = [str(item).strip() for item in items if str(item).strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def labeled_terms(label: str, terms: Sequence[str]) -> str:
    """Return ``label`` or ``label (a, b, and c)`` when terms are present."""
    joined = join_en(terms)
    if not joined:
        return label
    return f"{label} ({joined})"


def memory_enabled() -> bool:
    return get_agent_settings().memory_enabled


def memory_curator_enabled() -> bool:
    return get_agent_settings().memory_curator_enabled


def memory_prompt_optimizer_enabled() -> bool:
    return get_agent_settings().memory_prompt_optimizer_enabled


def compaction_enabled() -> bool:
    return get_agent_settings().compaction_enabled


def resolve_eval_module(*, raw: dict | None = None):
    """Return the host eval module from ``AI_AGENT.EVAL_MODULE``, or ``None``."""
    from importlib import import_module

    from django.core.exceptions import ImproperlyConfigured

    path = str(get_agent_settings(raw=raw).eval_module or "").strip()
    if not path:
        return None
    try:
        return import_module(path)
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"AI_AGENT.EVAL_MODULE {path!r} could not be imported."
        ) from exc


def resolve_authenticate_token(*, raw: dict | None = None):
    """Return the host JWT/user callable from ``AI_AGENT.AUTHENTICATE_TOKEN``."""
    from django.core.exceptions import ImproperlyConfigured
    from django.utils.module_loading import import_string

    value = get_agent_settings(raw=raw).authenticate_token
    if not isinstance(value, str) and callable(value):
        return value
    path = str(value or "").strip()
    if not path:
        raise ImproperlyConfigured(
            "AI_AGENT.AUTHENTICATE_TOKEN is required to validate access tokens."
        )
    loaded = import_string(path)
    if not callable(loaded):
        raise ImproperlyConfigured(
            "AI_AGENT.AUTHENTICATE_TOKEN must be a callable or dotted path to one."
        )
    return loaded


def resolve_agent_middleware(*, raw: dict | None = None, settings=None):
    """Instantiate ``AI_AGENT.MIDDLEWARE`` entries at graph-build time."""
    agent_settings = settings or get_agent_settings(raw=raw)
    return [_instantiate_middleware(item) for item in agent_settings.middleware]


def resolve_studio_django_settings_module() -> str:
    """Settings module for Studio / Agent Server entrypoints.

    ``AI_AGENT_STUDIO_DJANGO_SETTINGS`` wins when set; otherwise the existing
    ``DJANGO_SETTINGS_MODULE`` is kept. One of the two must be present.
    """
    override = os.environ.get("AI_AGENT_STUDIO_DJANGO_SETTINGS", "").strip()
    if override:
        return override
    current = os.environ.get("DJANGO_SETTINGS_MODULE", "").strip()
    if current:
        return current
    raise RuntimeError(
        "Set DJANGO_SETTINGS_MODULE or AI_AGENT_STUDIO_DJANGO_SETTINGS "
        "before starting the AI agent."
    )


def assert_production_studio_safe() -> None:
    """Refuse Studio-only fallbacks when the Agent Server is not in debug.

    ``disable_studio_auth`` in langgraph.json / the Agent Server image disables
    LangSmith login so the product JWT in ``ai_agent.auth`` is the gate. That
    is required in production. What must not ship is LangGraph desktop mode or
    the Studio phone impersonation fallback.
    """
    desktop = os.environ.get("LANGSMITH_LANGGRAPH_DESKTOP", "").strip().lower()
    if desktop in {"1", "true", "yes", "on"} and not getattr(
        settings, "DEBUG", False
    ):
        raise RuntimeError(
            "LangGraph Studio desktop mode is not allowed when DEBUG is false."
        )
    if getattr(settings, "TESTING", False) or getattr(settings, "DEBUG", False):
        return
    if get_agent_settings().studio_user_phone:
        raise RuntimeError(
            "AI_AGENT_STUDIO_USER_PHONE is a Studio impersonation fallback "
            "and is not allowed in production."
        )


def memory_is_hot() -> bool:
    return get_agent_settings().is_memory_hot()


def memory_is_background() -> bool:
    return get_agent_settings().is_memory_background()


def _platform_config(raw: dict) -> PlatformConfig:
    return PlatformConfig(
        platform_name=_as_nonempty_str(
            raw.get("PLATFORM_NAME"), _DEFAULT_PLATFORM_NAME
        ),
        domains=_as_str_tuple(raw.get("PLATFORM_DOMAINS")),
        entity_terms=_as_nonempty_str(
            raw.get("PLATFORM_ENTITY_TERMS"), _DEFAULT_ENTITY_TERMS
        ),
        markets_out_of_scope=_as_str_tuple(
            raw.get("PLATFORM_MARKETS_OUT_OF_SCOPE")
        ),
        mutation_terms=_as_str_tuple(raw.get("PLATFORM_MUTATION_TERMS")),
        pii_terms=_as_str_tuple(raw.get("PLATFORM_PII_TERMS")),
        currency_tokens=_as_str_tuple(raw.get("PLATFORM_CURRENCY_TOKENS")),
        phone_patterns=_as_pattern_tuple(raw.get("PLATFORM_PHONE_PATTERNS")),
        extra_pii_patterns=_as_pattern_tuple(
            raw.get("PLATFORM_EXTRA_PII_PATTERNS")
        ),
        extra_downgrade_patterns=_as_pattern_tuple(
            raw.get("PLATFORM_EXTRA_DOWNGRADE_PATTERNS")
        ),
    )


def _as_nonempty_str(raw, default: str) -> str:
    text = str(raw or "").strip()
    return text or default


def _as_str_tuple(raw) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def _as_pattern_tuple(raw) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def _as_authenticate_token(raw):
    if not isinstance(raw, str) and callable(raw):
        return raw
    return str(raw or "").strip()


def _as_middleware_tuple(raw) -> tuple:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(item for item in raw if item not in (None, ""))
    return ()


def _instantiate_middleware(item):
    import inspect

    from django.core.exceptions import ImproperlyConfigured
    from django.utils.module_loading import import_string

    if isinstance(item, str):
        path = item.strip()
        if not path:
            raise ImproperlyConfigured("AI_AGENT.MIDDLEWARE contains an empty path.")
        try:
            item = import_string(path)
        except ImportError as exc:
            raise ImproperlyConfigured(
                f"AI_AGENT.MIDDLEWARE {path!r} could not be imported."
            ) from exc
    if inspect.isclass(item):
        try:
            return item()
        except Exception as exc:
            raise ImproperlyConfigured(
                f"AI_AGENT.MIDDLEWARE {item!r} could not be instantiated."
            ) from exc
    if inspect.isfunction(item) or inspect.ismethod(item):
        try:
            return item()
        except Exception as exc:
            raise ImproperlyConfigured(
                f"AI_AGENT.MIDDLEWARE {item!r} could not be called."
            ) from exc
    if item is None:
        raise ImproperlyConfigured("AI_AGENT.MIDDLEWARE contains a null entry.")
    return item


def _as_http_throttle(raw) -> str:
    if raw is None or raw == "":
        return _DEFAULT_HTTP_THROTTLE
    text = str(raw).strip()
    if text.lower() in {"0", "none", "off", "false"}:
        return ""
    return text


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _curator_enabled(raw: dict) -> bool:
    if not _as_bool(raw.get("MEMORY_ENABLED", False)):
        return False
    flag = raw.get("MEMORY_CURATOR_ENABLED")
    if flag is None or flag == "":
        return True
    return _as_bool(flag)


def _optimizer_enabled(raw: dict) -> bool:
    if not _as_bool(raw.get("MEMORY_ENABLED", False)):
        return False
    flag = raw.get("MEMORY_PROMPT_OPTIMIZER_ENABLED")
    if flag is None or flag == "":
        from django.conf import settings as django_settings

        if getattr(django_settings, "TESTING", False):
            return False
        return True
    return _as_bool(flag)


def _as_non_negative_int(raw, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    if value < 0:
        return default
    return value


def _as_positive_int(raw, default: int) -> int:
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    if value < 1:
        return default
    return value


def _as_context_kind(value, default: str) -> str:
    kind = str(value or "").strip().lower()
    if kind not in _VALID_CONTEXT_KINDS:
        return default
    return kind


def _as_optional_positive_int(raw, default: int) -> int | None:
    if raw is None or raw == "":
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    if value == 0:
        return None
    if value < 0:
        return default
    return value


def _as_optional_positive_float(raw, default: float) -> float | None:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value == 0:
        return None
    if value < 0:
        return default
    return value


def _as_context_value(
    kind: str,
    raw,
    *,
    tokens_default: int,
    messages_default: int,
    fraction_default: float,
):
    if kind == "fraction":
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return fraction_default
        if not 0 < value <= 1:
            return fraction_default
        return value
    default = tokens_default if kind == "tokens" else messages_default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    if value < 1:
        return default
    return value
