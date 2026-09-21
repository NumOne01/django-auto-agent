"""Invoke a discovered DRF view in-process as a given user."""

from __future__ import annotations

import json
import logging
from urllib.parse import urlencode

from django.core.exceptions import ValidationError as DjangoValidationError
from django.http import Http404
from rest_framework.test import APIRequestFactory, force_authenticate

from ai_agent.conf import get_agent_settings
from ai_agent.discovery import DiscoveredEndpoint

logger = logging.getLogger(__name__)


def invoke_endpoint(endpoint: DiscoveredEndpoint, user, arguments: dict | None = None) -> str:
    from ai_agent.db import refresh_db_connections

    refresh_db_connections()
    arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
    try:
        path_kwargs = _coerce_path_kwargs(endpoint, arguments)
    except (TypeError, ValueError, DjangoValidationError) as exc:
        return f"HTTP 400: {exc}"
    query = {
        name: arguments[name]
        for name in endpoint.query_params
        if name in arguments
    }
    body = _body_payload(endpoint, arguments)
    path = _format_path(endpoint, path_kwargs)

    factory = APIRequestFactory()
    method = endpoint.method.lower()
    factory_method = getattr(factory, method)
    query_string = urlencode(query, doseq=True) if query else ""

    if method in {"get", "delete", "head"}:
        request_path = f"{path}?{query_string}" if query_string else path
        request = factory_method(request_path)
    else:
        request_path = f"{path}?{query_string}" if query_string else path
        request = factory_method(request_path, data=body, format="json")

    if user is not None and getattr(user, "is_authenticated", False):
        force_authenticate(request, user=user)

    try:
        response = endpoint.callback(request, **path_kwargs)
    except Http404:
        return "HTTP 404: Not found."
    except DjangoValidationError as exc:
        return f"HTTP 400: {exc}"
    except Exception as exc:
        logger.exception(
            "AI agent endpoint %s %s raised during invoke",
            endpoint.url_name,
            endpoint.method,
        )
        return f"HTTP 500: {type(exc).__name__}: {exc}"
    if hasattr(response, "render"):
        response.render()

    payload = _serialize_response(response, endpoint)
    max_chars = get_agent_settings().max_tool_response_chars
    total = len(payload)
    if total > max_chars:
        payload = (
            payload[:max_chars]
            + f"…[truncated, showing {max_chars} of {total} chars; "
            "do not invent omitted fields]"
        )
    return f"HTTP {response.status_code}: {payload}"


def _coerce_path_kwargs(endpoint: DiscoveredEndpoint, arguments: dict) -> dict:
    kwargs = {}
    for name in endpoint.path_params:
        if name not in arguments:
            continue
        value = arguments[name]
        converter = endpoint.converters.get(name)
        if converter is not None:
            kwargs[name] = converter.to_python(str(value))
        else:
            kwargs[name] = value
    return kwargs


def _format_path(endpoint: DiscoveredEndpoint, path_kwargs: dict) -> str:
    path = endpoint.path_template
    for name, value in path_kwargs.items():
        path = path.replace("{" + name + "}", str(value))
    if not path.startswith("/"):
        path = "/" + path
    return path


def _body_payload(endpoint: DiscoveredEndpoint, arguments: dict) -> dict:
    if endpoint.body_fields:
        return {name: arguments[name] for name in endpoint.body_fields if name in arguments}
    reserved = set(endpoint.path_params) | set(endpoint.query_params)
    return {k: v for k, v in arguments.items() if k not in reserved}


def _serialize_response(response, endpoint: DiscoveredEndpoint | None = None) -> str:
    data = getattr(response, "data", None)
    payload_data = _reshape_with_serializer(response, endpoint, data)
    if payload_data is not None:
        try:
            return json.dumps(payload_data, default=str, ensure_ascii=False)
        except TypeError:
            logger.warning("AI agent response JSON dump failed; using str()")
            return str(payload_data)
    content = getattr(response, "content", b"") or b""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _reshape_with_serializer(response, endpoint, data):
    """Apply an optional 2xx response serializer; fall back to the raw payload."""
    serializer_cls = getattr(endpoint, "response_serializer", None) if endpoint else None
    status_code = getattr(response, "status_code", 0) or 0
    if serializer_cls is None or data is None or not (200 <= int(status_code) < 300):
        return data
    try:
        many = isinstance(data, list)
        return serializer_cls(instance=data, many=many).data
    except Exception:
        logger.warning(
            "AI agent response_serializer %s failed; using raw payload",
            getattr(serializer_cls, "__name__", serializer_cls),
        )
        return data
