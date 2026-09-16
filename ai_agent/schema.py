"""Map drf-spectacular OpenAPI operations onto discovered endpoints."""

from __future__ import annotations

import re
from decimal import Decimal
from threading import Lock
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, create_model

from ai_agent.discovery import DiscoveredEndpoint

_PATH_PARAM_RE = re.compile(r"\{[^}]+\}")

_openapi_cache: Optional[dict[str, Any]] = None
_openapi_cache_lock = Lock()


def reset_schema_cache():
    global _openapi_cache
    with _openapi_cache_lock:
        _openapi_cache = None


def get_openapi_schema() -> dict[str, Any]:
    global _openapi_cache
    with _openapi_cache_lock:
        if _openapi_cache is None:
            from drf_spectacular.generators import SchemaGenerator

            generator = SchemaGenerator()
            _openapi_cache = generator.get_schema(request=None, public=True) or {}
        return _openapi_cache


def enrich_endpoints(endpoints: list[DiscoveredEndpoint]) -> list[DiscoveredEndpoint]:
    schema = get_openapi_schema()
    paths = schema.get("paths") or {}
    used_names: set[str] = set()

    for endpoint in endpoints:
        found = _find_operation(paths, endpoint)
        if found:
            spec_path, operation = found
            op_id = operation.get("operationId") or endpoint.operation_id
            endpoint.operation_id = _unique_name(op_id, used_names)
            endpoint.summary = operation.get("summary") or endpoint.summary
            endpoint.description = (
                endpoint.description
                or operation.get("description")
                or endpoint.summary
            )
            _apply_parameters(endpoint, schema, operation, spec_path=spec_path)
            for name in endpoint.path_params:
                endpoint.args_properties.setdefault(name, {"type": "string"})
                if name not in endpoint.args_required:
                    endpoint.args_required.append(name)
        else:
            endpoint.operation_id = _unique_name(endpoint.operation_id, used_names)
            if not endpoint.description:
                endpoint.description = endpoint.summary or endpoint.operation_id
            for name in endpoint.path_params:
                endpoint.args_properties.setdefault(name, {"type": "string"})
                if name not in endpoint.args_required:
                    endpoint.args_required.append(name)
    return endpoints


def build_args_model(endpoint: DiscoveredEndpoint) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    required = set(endpoint.args_required)
    for name, spec in endpoint.args_properties.items():
        py_type = _json_type_to_python(spec)
        description = _field_description(spec)
        if name in required:
            fields[name] = (py_type, Field(description=description))
        else:
            fields[name] = (Optional[py_type], Field(default=None, description=description))
    model_name = f"AgentArgs_{endpoint.operation_id}"
    if not fields:
        return create_model(model_name)
    return create_model(model_name, **fields)


def _unique_name(name: str, used: set[str]) -> str:
    from ai_agent.discovery import _safe_operation_id

    base = _safe_operation_id(name)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _find_operation(
    paths: dict, endpoint: DiscoveredEndpoint
) -> Optional[tuple[str, dict]]:
    method = endpoint.method.lower()
    candidates = {
        endpoint.path_template,
        endpoint.path_template.rstrip("/") + "/",
        endpoint.path_template.rstrip("/"),
    }
    for path in candidates:
        item = paths.get(path)
        if not item:
            continue
        operation = item.get(method)
        if operation:
            return path, operation
    # Spectacular coerces {pk} → {id} and may keep a trailing slash.
    for spec_path, item in paths.items():
        if _paths_equivalent(spec_path, endpoint.path_template) and item.get(method):
            return spec_path, item.get(method)
    return None


def _paths_equivalent(left: str, right: str) -> bool:
    """Treat OpenAPI `{id}` and Django `{pk}` (etc.) as the same path slot."""

    def normalize(path: str) -> str:
        return _PATH_PARAM_RE.sub("{}", path.rstrip("/"))

    return normalize(left) == normalize(right)


def _path_param_names(path: str) -> list[str]:
    return [match[1:-1] for match in _PATH_PARAM_RE.findall(path)]


def _path_param_rename(django_names: list[str], openapi_names: list[str]) -> dict[str, str]:
    if len(django_names) != len(openapi_names):
        return {}
    return {
        spec: django
        for spec, django in zip(openapi_names, django_names)
        if spec != django
    }


def _apply_parameters(
    endpoint: DiscoveredEndpoint,
    schema: dict,
    operation: dict,
    spec_path: str | None = None,
):
    properties: dict[str, Any] = {}
    required: list[str] = []
    query_params: list[str] = []
    body_fields: list[str] = []
    openapi_path_names: list[str] = []

    for param in operation.get("parameters") or []:
        param = _resolve_ref(schema, param)
        name = param.get("name")
        if not name:
            continue
        location = param.get("in")
        spec = param.get("schema") or {"type": "string"}
        spec = _resolve_ref(schema, spec)
        if param.get("description") and "description" not in spec:
            spec = {**spec, "description": param["description"]}
        properties[name] = spec
        if location == "path" or param.get("required"):
            if name not in required:
                required.append(name)
        if location == "query":
            query_params.append(name)
        elif location == "path":
            openapi_path_names.append(name)

    rename = _path_param_rename(endpoint.path_params, openapi_path_names)
    if not rename and spec_path:
        rename = _path_param_rename(
            _path_param_names(endpoint.path_template),
            _path_param_names(spec_path),
        )
    if rename:
        properties = {rename.get(name, name): spec for name, spec in properties.items()}
        required = [rename.get(name, name) for name in required]
        openapi_path_names = [rename.get(name, name) for name in openapi_path_names]

    for name in openapi_path_names:
        if name not in endpoint.path_params:
            endpoint.path_params.append(name)

    body = operation.get("requestBody") or {}
    content = body.get("content") or {}
    json_body = (
        content.get("application/json")
        or content.get("application/x-www-form-urlencoded")
        or content.get("multipart/form-data")
        or next(iter(content.values()), {})
        or {}
    )
    body_schema = json_body.get("schema")
    if body_schema:
        body_schema = _resolve_ref(schema, body_schema)
        body_props = body_schema.get("properties") or {}
        for name, spec in body_props.items():
            spec = _resolve_ref(schema, spec)
            if _is_binary_upload(spec):
                continue
            properties[name] = spec
            body_fields.append(name)
        for name in body_schema.get("required") or []:
            if name not in required and name in properties:
                required.append(name)

    endpoint.args_properties = properties
    endpoint.args_required = required
    endpoint.query_params = query_params
    endpoint.body_fields = body_fields


def _resolve_ref(schema: dict, spec: Any) -> dict:
    if not isinstance(spec, dict):
        return {"type": "string"}
    if "$ref" in spec:
        node: Any = schema
        for part in spec["$ref"].lstrip("#/").split("/"):
            if not isinstance(node, dict):
                return {"type": "object"}
            node = node.get(part)
            if node is None:
                return {"type": "object"}
        merged = _resolve_ref(schema, node)
        extras = {k: v for k, v in spec.items() if k != "$ref"}
        if extras:
            return _merge_subschemas([merged, extras])
        return merged
    if "allOf" in spec:
        parts = [_resolve_ref(schema, item) for item in spec["allOf"]]
        extras = {k: v for k, v in spec.items() if k != "allOf"}
        if extras:
            parts.append(extras)
        return _merge_subschemas(parts)
    if "anyOf" in spec or "oneOf" in spec:
        key = "anyOf" if "anyOf" in spec else "oneOf"
        parts = [_resolve_ref(schema, item) for item in spec[key]]
        extras = {k: v for k, v in spec.items() if k != key}
        non_null = [part for part in parts if part.get("type") != "null"]
        chosen = non_null or parts
        if extras:
            chosen = [*chosen, extras]
        return _merge_subschemas(chosen)
    return spec


def _merge_subschemas(parts: list[dict]) -> dict:
    """Flatten spectacular allOf/anyOf. Enums stay scalars, not objects."""
    merged: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    enums: list[Any] = []
    types: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        props = part.get("properties")
        if isinstance(props, dict) and props:
            properties.update(props)
        required.extend(part.get("required") or [])
        if part.get("enum"):
            enums.extend(part["enum"])
        type_name = part.get("type")
        if type_name:
            types.append(type_name)
        for key, value in part.items():
            if key in {
                "properties",
                "required",
                "enum",
                "type",
                "allOf",
                "anyOf",
                "oneOf",
            }:
                continue
            if key not in merged or not merged[key]:
                merged[key] = value
    if properties:
        merged["properties"] = {**properties, **(merged.get("properties") or {})}
        merged["required"] = list(dict.fromkeys(required))
        merged.setdefault("type", "object")
        return merged
    if enums:
        merged["enum"] = list(dict.fromkeys(enums))
    scalar = _scalar_type(types)
    if scalar:
        merged["type"] = scalar
    elif enums:
        merged["type"] = "string"
    return merged or {"type": "string"}


def _scalar_type(types: list[str]) -> str | None:
    non_null = [item for item in types if item and item != "null"]
    non_object = [item for item in non_null if item != "object"]
    if non_object:
        return non_object[0]
    if non_null:
        return non_null[0]
    return None


def _field_description(spec: dict) -> str:
    description = spec.get("description") or ""
    enum_values = spec.get("enum") or []
    if enum_values:
        allowed = ", ".join(str(value) for value in enum_values)
        extra = f"Allowed: {allowed}."
        description = f"{description} {extra}".strip() if description else extra
    return description


def _coerce_decimal_input(value):
    """LLMs send decimals as JSON numbers; DRF still validates decimal_places.

    ``format(1.5, "f")`` becomes ``1.500000``, which fails a 2-place price field.
    Normalize via Decimal so ints/floats stay compact fixed-point strings.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, float):
        decimal_value = Decimal(str(value))
    elif isinstance(value, int):
        return str(value)
    else:
        return value
    return format(decimal_value, "f").rstrip("0").rstrip(".") or "0"


DecimalString = Annotated[str, BeforeValidator(_coerce_decimal_input)]


def _is_binary_upload(spec: dict) -> bool:
    if spec.get("format") in {"binary", "byte"}:
        return True
    items = spec.get("items")
    if not isinstance(items, dict):
        return False
    if items.get("format") in {"binary", "byte"}:
        return True
    # drf-spectacular often emits FileField lists as string/uri, not binary.
    return spec.get("type") == "array" and items.get("format") == "uri"


def _json_type_to_python(spec: dict):
    enum_values = spec.get("enum") or []
    if enum_values and all(isinstance(value, str) for value in enum_values):
        return Literal[*tuple(enum_values)]
    if enum_values:
        return str
    type_name = spec.get("type")
    if spec.get("format") == "decimal":
        return DecimalString
    if spec.get("format") == "binary":
        return str
    if type_name == "integer":
        return int
    if type_name == "number":
        return float
    if type_name == "boolean":
        return bool
    if type_name == "array":
        return list
    if type_name == "object":
        return dict
    return str
