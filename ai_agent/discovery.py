"""Walk the Django URLconf and collect endpoints marked for the AI agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Optional

from django.apps import apps
from django.urls import URLPattern, URLResolver, get_resolver
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.views import APIView

from ai_agent.conf import get_agent_settings
from ai_agent.expose import get_view_attr

HARD_SKIP_PATH_FRAGMENTS = ("/internal/", "/webhooks/", "/callback/")
SKIP_HTTP_METHODS = frozenset({"options", "head"})
_PATH_PARAM_RE = re.compile(r"<([^:>]+:)?([^>]+)>")

_cache: Optional[list["DiscoveredEndpoint"]] = None
_cache_lock = Lock()


@dataclass
class DiscoveredEndpoint:
    app_label: str
    url_name: str
    method: str
    path_template: str
    django_path: str
    callback: Callable
    confirm: bool
    operation_id: str
    summary: str = ""
    description: str = ""
    path_params: list[str] = field(default_factory=list)
    query_params: list[str] = field(default_factory=list)
    body_fields: list[str] = field(default_factory=list)
    converters: dict[str, Any] = field(default_factory=dict)
    args_properties: dict[str, Any] = field(default_factory=dict)
    args_required: list[str] = field(default_factory=list)
    response_serializer: Any = None


def reset_discovery_cache():
    global _cache
    with _cache_lock:
        _cache = None


def discover_endpoints(*, use_cache: bool = True) -> list[DiscoveredEndpoint]:
    global _cache
    if use_cache:
        with _cache_lock:
            if _cache is not None:
                return list(_cache)
            endpoints = _discover()
            _cache = list(endpoints)
            return list(endpoints)
    endpoints = _discover()
    with _cache_lock:
        _cache = list(endpoints)
    return endpoints


def _discover() -> list[DiscoveredEndpoint]:
    from ai_agent.schema import enrich_endpoints

    raw: list[DiscoveredEndpoint] = []
    for django_path, pattern in _iter_url_patterns():
        callback = pattern.callback
        if not _is_drf_view(callback):
            continue
        normalized = _normalize_path(django_path)
        app_label = _app_label_for(callback)
        if not app_label:
            continue
        url_name = pattern.name or ""
        methods = _http_methods(callback)
        if not methods:
            continue

        explicit_expose = bool(get_view_attr(callback, "_agent_expose", False))
        if not _is_exposed(callback, app_label, url_name, normalized, explicit_expose):
            continue

        converters = _converters_for(pattern)
        path_params = list(converters.keys())
        path_template = _django_path_to_openapi(normalized)
        confirm_override = get_view_attr(callback, "_agent_confirm", None)
        view_description = get_view_attr(callback, "_agent_description", None)
        view_serializer = get_view_attr(callback, "_agent_response_serializer", None)

        for method in methods:
            confirm = _resolve_confirm(method, confirm_override)
            operation_id = url_name or f"{app_label}_{method.lower()}"
            if len(methods) > 1:
                operation_id = f"{operation_id}_{method.lower()}"
            raw.append(
                DiscoveredEndpoint(
                    app_label=app_label,
                    url_name=url_name,
                    method=method,
                    path_template=path_template,
                    django_path=normalized,
                    callback=callback,
                    confirm=confirm,
                    operation_id=_safe_operation_id(operation_id),
                    description=view_description or "",
                    path_params=path_params,
                    converters=converters,
                    response_serializer=view_serializer,
                )
            )

    return enrich_endpoints(raw)


def _iter_url_patterns(resolver=None, prefix=""):
    if resolver is None:
        resolver = get_resolver()
    for pattern in resolver.url_patterns:
        if isinstance(pattern, URLResolver):
            yield from _iter_url_patterns(pattern, prefix + str(pattern.pattern))
        elif isinstance(pattern, URLPattern):
            yield prefix + str(pattern.pattern), pattern


def _is_drf_view(callback) -> bool:
    cls = getattr(callback, "cls", None)
    return isinstance(cls, type) and issubclass(cls, APIView)


def _http_methods(callback) -> list[str]:
    cls = getattr(callback, "cls", None)
    if cls is None:
        return []
    methods = []
    for name in getattr(cls, "http_method_names", []) or []:
        lowered = name.lower()
        if lowered in SKIP_HTTP_METHODS:
            continue
        if hasattr(cls, lowered):
            methods.append(lowered.upper())
    return methods


def _app_label_for(callback) -> str:
    cls = getattr(callback, "cls", None)
    module = getattr(cls, "__module__", "") if cls is not None else ""
    if not module:
        module = getattr(callback, "__module__", "") or ""
    app_label = module.split(".", 1)[0]
    if not app_label or app_label in {"django", "rest_framework", "drf_spectacular"}:
        return ""
    try:
        apps.get_app_config(app_label)
    except LookupError:
        return ""
    return app_label


def _converters_for(pattern) -> dict[str, Any]:
    route = getattr(pattern, "pattern", None)
    converters = getattr(route, "converters", None) or {}
    return dict(converters)


def _normalize_path(django_path: str) -> str:
    path = "/" + str(django_path).lstrip("/")
    return path.replace("//", "/")


def _django_path_to_openapi(path: str) -> str:
    converted = _PATH_PARAM_RE.sub(r"{\2}", path)
    if len(converted) > 1:
        converted = converted.rstrip("/")
    return converted or "/"


def _safe_operation_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value or "endpoint")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"op_{cleaned}"
    return cleaned or "endpoint"


def _is_hard_skip_path(path: str) -> bool:
    haystack = path if path.endswith("/") else path + "/"
    return any(fragment in haystack for fragment in HARD_SKIP_PATH_FRAGMENTS)


def _has_multipart_parser(callback) -> bool:
    """True when the view cannot accept JSON (upload-only parsers)."""
    cls = getattr(callback, "cls", None)
    parsers = list(getattr(cls, "parser_classes", None) or [])
    if not parsers:
        return False
    names = {getattr(parser, "__name__", "") for parser in parsers}
    has_multipart = MultiPartParser in parsers or "MultiPartParser" in names
    has_json = JSONParser in parsers or "JSONParser" in names
    return has_multipart and not has_json


def _app_config(app_label: str):
    try:
        return apps.get_app_config(app_label)
    except LookupError:
        return None


def _is_exposed(callback, app_label: str, url_name: str, path: str, explicit_expose: bool) -> bool:
    if get_view_attr(callback, "_agent_exclude", False):
        return False

    hard_skip = _is_hard_skip_path(path) or _has_multipart_parser(callback)
    if hard_skip and not explicit_expose:
        return False

    if explicit_expose:
        return True

    config = _app_config(app_label)
    if config is None:
        return False
    excluded_names = set(getattr(config, "agent_exclude", ()) or ())
    if url_name and url_name in excluded_names:
        return False
    return bool(getattr(config, "agent_expose", False))


def _resolve_confirm(method: str, override) -> bool:
    if override is not None:
        return bool(override)
    if not get_agent_settings().confirm_mutations:
        return False
    return method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
