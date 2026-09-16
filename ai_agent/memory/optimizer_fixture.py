"""Fixture memories for testing reconcile, local/global overlays, and PII leaks.

Dev/test only. Fake PII is hardcoded so optimizer evals can assert redaction.
Do not run against production data; the ``seed_memory_optimizer_fixture``
command refuses to seed unless DEBUG, TESTING, or
``AI_AGENT_ALLOW_OPTIMIZER_FIXTURE=1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_agent.memory.namespaces import (
    SUPERVISOR_LAYER,
    bind_namespace,
    global_prompt_namespace,
    memory_namespaces,
)

LAYER = "dummy"

USER_A_LOGIN = "fixture_a"
USER_B_LOGIN = "fixture_b"


@dataclass(frozen=True)
class FixturePerson:
    label: str
    login: str
    first_name: str
    last_name: str
    full_name: str
    episode_phone: str
    email: str
    account_id: str
    language: str
    standing_pref: str


PERSON_A = FixturePerson(
    label="A",
    login=USER_A_LOGIN,
    first_name="Alex",
    last_name="Example",
    full_name="Alex Example",
    episode_phone="+15550101111",
    email="alex@example.com",
    account_id="GB82WEST12345698765432",
    language="English",
    standing_pref="Always confirm catalog creates before calling tools.",
)
PERSON_B = FixturePerson(
    label="B",
    login=USER_B_LOGIN,
    first_name="Sam",
    last_name="Sample",
    full_name="Sam Sample",
    episode_phone="+15550102222",
    email="sam@example.net",
    account_id="GB29NWBK60161331926819",
    language="English",
    standing_pref="Keep replies short; never guess identifiers.",
)


def pii_tokens(*people: FixturePerson) -> list[str]:
    """Strings that must not appear in a published global overlay."""
    tokens: list[str] = []
    for person in people or (PERSON_A, PERSON_B):
        tokens.extend(
            [
                person.full_name,
                person.first_name + " " + person.last_name,
                person.episode_phone,
                person.email,
                person.account_id,
            ]
        )
    return tokens


def seed_optimizer_fixture(store, *, user_a_id: str, user_b_id: str) -> dict[str, dict[str, int]]:
    """Write duplicate facts/episodes plus learnable failures with PII."""
    counts = {}
    counts[f"{user_a_id}/{LAYER}"] = _seed_layer(store, user_a_id, PERSON_A)
    counts[f"{user_b_id}/{LAYER}"] = _seed_layer(store, user_b_id, PERSON_B)
    counts[f"{user_a_id}/{SUPERVISOR_LAYER}"] = _seed_supervisor(
        store, user_a_id, PERSON_A
    )
    return counts


def wipe_optimizer_fixture(store, *, user_a_id: str, user_b_id: str) -> None:
    for user_id in (user_a_id, user_b_id):
        for layer in (LAYER, SUPERVISOR_LAYER):
            _wipe_layer(store, user_id, layer)
    _wipe_namespace(store, global_prompt_namespace(LAYER))
    _wipe_namespace(store, global_prompt_namespace(SUPERVISOR_LAYER))


def count_layer(store, user_id: str, layer: str) -> dict[str, int]:
    namespaces = memory_namespaces(layer)
    return {
        name: _len(store, bind_namespace(template, user_id=str(user_id)))
        for name, template in namespaces.items()
        if name in {"semantic", "episodes", "profile", "playbook", "prompt"}
    }


def _seed_layer(store, user_id: str, person: FixturePerson) -> dict[str, int]:
    semantic = bind_namespace(
        memory_namespaces(LAYER)["semantic"], user_id=str(user_id)
    )
    episodes = bind_namespace(
        memory_namespaces(LAYER)["episodes"], user_id=str(user_id)
    )
    facts = _layer_facts(person)
    traces = _layer_episodes(person)
    for index, content in enumerate(facts, start=1):
        store.put(semantic, f"fact-{person.label}-{index}", {"content": content})
    for index, content in enumerate(traces, start=1):
        store.put(episodes, f"ep-{person.label}-{index}", {"content": content})
    return count_layer(store, user_id, LAYER)


def _seed_supervisor(store, user_id: str, person: FixturePerson) -> dict[str, int]:
    semantic = bind_namespace(
        memory_namespaces(SUPERVISOR_LAYER)["semantic"], user_id=str(user_id)
    )
    episodes = bind_namespace(
        memory_namespaces(SUPERVISOR_LAYER)["episodes"], user_id=str(user_id)
    )
    facts = [
        {
            "subject": "User",
            "predicate": "language",
            "object": person.language,
            "context": "",
        },
        {
            "subject": "user",
            "predicate": "Language",
            "object": person.language.lower(),
            "context": "replies",
        },
        {
            "subject": "User",
            "predicate": "prefers",
            "object": "dummy catalog",
            "context": "typical domains",
        },
    ]
    traces = [
        _episode(
            observation=(
                f"{person.full_name} asked about two catalog items in one message."
            ),
            thoughts="I should route the dummy specialist instead of guessing.",
            action="Called the dummy specialist.",
            result="Success when the domain used its own tools.",
        ),
        _episode(
            observation=(
                f"{person.full_name} asked about two catalog items in one message."
            ),
            thoughts="I should route the dummy specialist instead of guessing.",
            action="Called the dummy specialist.",
            result="Success when the domain used its own tools.",
        ),
    ]
    for index, content in enumerate(facts, start=1):
        store.put(semantic, f"sup-fact-{index}", {"content": content})
    for index, content in enumerate(traces, start=1):
        store.put(episodes, f"sup-ep-{index}", {"content": content})
    return count_layer(store, user_id, SUPERVISOR_LAYER)


def _layer_facts(person: FixturePerson) -> list[dict[str, Any]]:
    return [
        {
            "subject": "User",
            "predicate": "prefers",
            "object": "Acme",
            "context": "",
        },
        {
            "subject": "user",
            "predicate": "PREFERS",
            "object": "Acme account for catalog orders",
            "context": "standing rail",
        },
        {
            "subject": "User",
            "predicate": "prefers",
            "object": "Acme for create",
            "context": "dummy",
        },
        {
            "subject": "User",
            "predicate": "language",
            "object": person.language,
            "context": "",
        },
        {
            "subject": "User",
            "predicate": "language",
            "object": person.language,
            "context": "spoken in chat",
        },
        {
            "bank_name": "Acme",
            "label": "primary",
            "use_for": "create",
        },
        {
            "bank_name": "Acme",
            "label": "primary",
            "use_for": "both",
        },
        {
            "subject": "User",
            "predicate": "constraint",
            "object": person.standing_pref,
            "context": "standing preference, not a live value",
        },
    ]


def _layer_episodes(person: FixturePerson) -> list[dict[str, Any]]:
    confirm_lesson = _episode(
        observation=(
            f"{person.full_name} ({person.email}, {person.episode_phone}) asked to "
            f"create an item billed to {person.account_id}."
        ),
        thoughts=(
            "I almost created the item without a confirmation interrupt. "
            "This customer expects an explicit confirm."
        ),
        action="Called the dummy create tool after the user confirmed.",
        result=(
            "Failure was prevented. Always confirm catalog creates before calling "
            "dummy tools. Do not retry with guessed account IDs."
        ),
    )
    spaced_dup = _episode(
        observation=(
            f"{person.full_name} ({person.email},  {person.episode_phone})  asked to "
            f"create an item billed to {person.account_id}."
        ),
        thoughts=(
            "I almost created the item without a confirmation interrupt. "
            "This customer expects an explicit confirm."
        ),
        action="Called the dummy create tool after the user confirmed.",
        result=(
            "Failure was prevented. Always confirm catalog creates before calling "
            "dummy tools. Do not retry with guessed account IDs."
        ),
    )
    id_fail = _episode(
        observation=(
            f"User {person.full_name} tried to create using account {person.account_id}."
        ),
        thoughts="The account id looks sensitive. I must not store or repeat it.",
        action="Asked the user to confirm the last four characters instead.",
        result=(
            "Failure: HTTP 400 declined. Prevent by never sending a full account "
            "id and by calling tools instead of guessing values."
        ),
    )
    transfer_ok = _episode(
        observation="User asked to list their usual items.",
        thoughts="This is a standing catalog request, not a one-off stranger.",
        action="Used the dummy list tool.",
        result="Succeeded after the user confirmed the request.",
    )
    unique_pref = _episode(
        observation=f"{person.full_name} said: {person.standing_pref}",
        thoughts="This is a durable how-to for this customer, not a one-off task.",
        action="Noted the standing constraint for future dummy turns.",
        result=f"Success. Reuse this standing preference: {person.standing_pref}",
    )
    return [confirm_lesson, spaced_dup, id_fail, transfer_ok, unique_pref]


def _episode(observation: str, thoughts: str, action: str, result: str) -> dict[str, str]:
    return {
        "observation": observation,
        "thoughts": thoughts,
        "action": action,
        "result": result,
    }


def _wipe_layer(store, user_id: str, layer: str) -> None:
    for template in memory_namespaces(layer).values():
        _wipe_namespace(store, bind_namespace(template, user_id=str(user_id)))


def _wipe_namespace(store, namespace: tuple[str, ...]) -> None:
    try:
        items = list(store.search(namespace, limit=80) or [])
    except Exception:
        items = []
    for item in items:
        key = getattr(item, "key", None)
        if key:
            try:
                store.delete(namespace, key=key)
            except TypeError:
                store.delete(namespace, key)


def _len(store, namespace: tuple[str, ...]) -> int:
    try:
        return len(list(store.search(namespace, limit=80) or []))
    except Exception:
        return 0
