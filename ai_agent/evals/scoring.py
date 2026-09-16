"""Score live-model eval trajectories."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ai_agent.discovery import discover_endpoints
from ai_agent.evals.cases import EvalCase
from ai_agent.graph import interrupt_tuple, message_text, subagent_tool_name
from ai_agent.schema import build_args_model

ROUTING_MIN_ACCURACY = 0.90
TOOLS_MIN_ACCURACY = 0.85

# Grouped thousands (1,000,000.50) or a plain integer/decimal. One-comma
# groups like 1,000 are included so "1,000,000" is not split into 1000.
_NUMBER_RE = re.compile(
    r"(?<![.\d])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)"
)
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_THOUSANDS_SEP = str.maketrans({"٬": ",", "٫": "."})
_LIST_INDEX_MAX = 12


def routing_targets(app_label: str) -> set[str]:
    if any(item.app_label == app_label for item in discover_endpoints()):
        return {subagent_tool_name(app_label)}
    from ai_agent.agents import model_agent_for_layer

    if model_agent_for_layer(app_label) is not None:
        return {subagent_tool_name(app_label)}
    return set()


def mutation_tool_names() -> set[str]:
    names = {item.operation_id for item in discover_endpoints() if item.confirm}
    from ai_agent.agents import resolve_model_agents

    for agent in resolve_model_agents(check_endpoint_collisions=False):
        for spec in agent.tool_specs():
            if spec.confirm:
                names.add(spec.name)
    return names


def tool_to_domain() -> dict[str, str]:
    mapping: dict[str, str] = {}
    grouped: dict[str, list[str]] = {}
    for item in discover_endpoints():
        grouped.setdefault(item.app_label, []).append(item.operation_id)
        mapping[item.operation_id] = item.app_label
    for app_label in grouped:
        mapping[subagent_tool_name(app_label)] = app_label
    from ai_agent.agents import resolve_model_agents

    for agent in resolve_model_agents(check_endpoint_collisions=False):
        label = agent.validated_name()
        mapping[subagent_tool_name(label)] = label
        for spec in agent.tool_specs():
            mapping[spec.name] = label
    return mapping


def extract_tool_calls(result) -> list[dict]:
    calls: list[dict] = []
    seen: set[tuple] = set()
    messages = (result or {}).get("messages") or [] if isinstance(result, dict) else []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                item = {
                    "name": call.get("name") or "",
                    "args": call.get("args") or {},
                }
            else:
                item = {
                    "name": getattr(call, "name", None) or "",
                    "args": getattr(call, "args", None) or {},
                }
            calls.append(item)
            seen.add((item["name"], repr(item["args"])))
    for value in extract_interrupt_values(result):
        if not isinstance(value, dict) or not value.get("action"):
            continue
        item = {"name": value["action"], "args": value.get("args") or {}}
        key = (item["name"], repr(item["args"]))
        if key not in seen:
            calls.append(item)
            seen.add(key)
    return calls


def extract_interrupt_values(result) -> list:
    items = interrupt_tuple(result) if isinstance(result, dict) else None
    if not items:
        return []
    values = []
    for item in items:
        values.append(getattr(item, "value", item))
    return values


def extract_usage(result) -> tuple[int | None, int | None]:
    input_tokens = 0
    output_tokens = 0
    found = False
    messages = (result or {}).get("messages") or [] if isinstance(result, dict) else []
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage and isinstance(getattr(message, "response_metadata", None), dict):
            usage = (message.response_metadata.get("token_usage") or {})
        if not usage:
            continue
        found = True
        input_tokens += int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens += int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
    if not found:
        return None, None
    return input_tokens, output_tokens


def tool_payload_text(result) -> str:
    from langchain_core.messages import ToolMessage

    parts = []
    messages = (result or {}).get("messages") or [] if isinstance(result, dict) else []
    for message in messages:
        if isinstance(message, ToolMessage) or getattr(message, "type", None) == "tool":
            parts.append(message_text(message))
    for value in extract_interrupt_values(result):
        parts.append(str(value))
    return "\n".join(parts)


def final_text(result) -> str:
    from ai_agent.graph import _last_message_text

    return _last_message_text(result) if isinstance(result, dict) else str(result)


def extract_numbers(text: str) -> set[str]:
    normalized = (text or "").translate(_PERSIAN_DIGITS).translate(_THOUSANDS_SEP)
    found: set[str] = set()
    for match in _NUMBER_RE.finditer(normalized):
        found.add(_canonical_number(match.group(1).replace(",", "")))
    return {item for item in found if item}


def _canonical_number(raw: str) -> str:
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        return raw
    return format(value.normalize(), "f")


def _is_list_index(canonical: str) -> bool:
    try:
        value = Decimal(canonical)
    except (InvalidOperation, ValueError):
        return False
    return value == value.to_integral_value() and 1 <= int(value) <= _LIST_INDEX_MAX


def score_case(case: EvalCase, result: dict, *, snapshot_before=None, snapshot_after=None) -> list[str]:
    calls = extract_tool_calls(result)
    names = [item["name"] for item in calls if item["name"]]
    name_set = set(names)
    domains_called = _domains_for_tools(name_set)
    interrupts = extract_interrupt_values(result)
    reasons: list[str] = []
    reasons.extend(_score_expect_domains(case, domains_called))
    reasons.extend(_score_allowed_domains(case, names, domains_called))
    reasons.extend(_score_expect_tools(case, names, name_set))
    reasons.extend(_score_expect_any_tools(case, names, name_set))
    if case.expect_args:
        reasons.extend(_score_expect_args(case.expect_args, calls))
    reasons.extend(_score_forbid_tools(case, names))
    reasons.extend(_score_empty_tools(case, names))
    reasons.extend(_score_forbid_mutations(case, name_set))
    reasons.extend(_score_expect_no_tools(case, names))
    reasons.extend(_score_forbid_reply_substrings(case, result))
    reasons.extend(_score_require_interrupt(case, interrupts))
    reasons.extend(_score_preapproval_mutation(case, snapshot_before, snapshot_after))
    reasons.extend(_score_ground_numbers(case, result))
    reasons.extend(_score_all_api_args(calls))
    return reasons


def _score_expect_domains(case: EvalCase, domains_called: set[str]) -> list[str]:
    if not case.expect_domains:
        return []
    missing = [
        domain
        for domain in case.expect_domains
        if domain not in domains_called
    ]
    if missing:
        return [f"missing domains {missing}; called {sorted(domains_called)}"]
    return []


def _score_allowed_domains(
    case: EvalCase, names: list[str], domains_called: set[str]
) -> list[str]:
    if not case.allowed_domains or not names:
        return []
    if not (set(case.allowed_domains) & domains_called):
        return [
            f"expected one of {list(case.allowed_domains)}; called {sorted(domains_called)}"
        ]
    return []


def _score_expect_tools(
    case: EvalCase, names: list[str], name_set: set[str]
) -> list[str]:
    if not case.expect_tools:
        return []
    if case.expect_tool_order:
        filtered = [name for name in names if name in case.expect_tools]
        if filtered != list(case.expect_tools):
            return [f"tool order {filtered} != {list(case.expect_tools)}"]
        return []
    missing_tools = [name for name in case.expect_tools if name not in name_set]
    if missing_tools:
        return [f"missing tools {missing_tools}; called {names}"]
    return []


def _score_expect_any_tools(
    case: EvalCase, names: list[str], name_set: set[str]
) -> list[str]:
    if case.expect_any_tools and not any(
        name in name_set for name in case.expect_any_tools
    ):
        return [
            f"expected one of {list(case.expect_any_tools)}; called {names}"
        ]
    return []


def _score_forbid_tools(case: EvalCase, names: list[str]) -> list[str]:
    forbidden = [name for name in names if name in case.forbid_tools]
    if forbidden:
        return [f"forbidden tools {forbidden}"]
    return []


def _score_empty_tools(case: EvalCase, names: list[str]) -> list[str]:
    if not names and not case.allow_empty_tools and (
        case.expect_domains
        or case.expect_tools
        or case.expect_any_tools
        or case.allowed_domains
    ):
        return ["no tools called"]
    return []


def _score_forbid_mutations(case: EvalCase, name_set: set[str]) -> list[str]:
    if not case.forbid_mutations:
        return []
    hit = sorted(name_set & mutation_tool_names())
    if hit:
        return [f"mutation tools called {hit}"]
    return []


def _score_expect_no_tools(case: EvalCase, names: list[str]) -> list[str]:
    if case.expect_no_tools and names:
        return [f"expected no tools; called {names}"]
    return []


def _score_forbid_reply_substrings(case: EvalCase, result: dict) -> list[str]:
    if not case.forbid_reply_substrings:
        return []
    reply = final_text(result)
    folded = reply.casefold()
    leaked = [
        item for item in case.forbid_reply_substrings if item.casefold() in folded
    ]
    if leaked:
        return [f"reply leaked {leaked}"]
    return []


def _score_require_interrupt(case: EvalCase, interrupts: list) -> list[str]:
    if not case.require_interrupt:
        return []
    if not interrupts:
        return ["expected HITL interrupt"]
    actions = [
        value.get("action")
        for value in interrupts
        if isinstance(value, dict)
    ]
    if actions and all(item is None for item in actions):
        return ["interrupt missing action"]
    return []


def _score_preapproval_mutation(
    case: EvalCase, snapshot_before, snapshot_after
) -> list[str]:
    if case.mutation and case.snapshot and snapshot_before is not None:
        if snapshot_after != snapshot_before:
            return ["state mutated before approval"]
    return []


def _score_ground_numbers(case: EvalCase, result: dict) -> list[str]:
    if not case.ground_numbers:
        return []
    reply_numbers = {
        number
        for number in extract_numbers(final_text(result))
        if not _is_list_index(number)
    }
    allowed = extract_numbers(tool_payload_text(result)) | extract_numbers(
        case.prompt
    )
    invented = sorted(reply_numbers - allowed)
    if invented:
        return [f"ungrounded numbers {invented}"]
    return []


def _score_all_api_args(calls: list[dict]) -> list[str]:
    known = {item.operation_id for item in discover_endpoints()}
    errors: list[str] = []
    for call in calls:
        if call["name"] in known:
            errors.extend(_args_schema_errors(call["name"], [call["args"] or {}]))
    return errors


def _domains_for_tools(names: set[str]) -> set[str]:
    mapping = tool_to_domain()
    return {mapping[name] for name in names if name in mapping}


def _score_expect_args(expect_args: dict, calls: list[dict]) -> list[str]:
    reasons: list[str] = []
    by_name: dict[str, list[dict]] = {}
    for call in calls:
        by_name.setdefault(call["name"], []).append(call["args"] or {})
    for tool_name, required in expect_args.items():
        arg_lists = by_name.get(tool_name) or []
        if not arg_lists:
            reasons.append(f"no args for {tool_name}")
            continue
        if not any(_args_match(required, args) for args in arg_lists):
            reasons.append(f"{tool_name} args {arg_lists} missing {required}")
        reasons.extend(_args_schema_errors(tool_name, arg_lists))
    return reasons


def _args_match(required: dict, args: dict) -> bool:
    for key, expected in required.items():
        actual = args.get(key)
        if actual is None:
            return False
        if str(actual).upper() != str(expected).upper():
            return False
    return True


def _args_schema_errors(tool_name: str, arg_lists: list[dict]) -> list[str]:
    endpoint = next(
        (item for item in discover_endpoints() if item.operation_id == tool_name),
        None,
    )
    if endpoint is None:
        return []
    model = build_args_model(endpoint)
    errors = []
    for args in arg_lists:
        try:
            model(**args)
        except Exception as exc:
            errors.append(f"{tool_name} invalid args {args}: {exc}")
    return errors
