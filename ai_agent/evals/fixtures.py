"""Seed a deterministic world for live-model evals."""

from __future__ import annotations

from dataclasses import dataclass, field

from django.contrib.auth import get_user_model
from django.core.exceptions import ImproperlyConfigured

from ai_agent.conf import resolve_eval_module

User = get_user_model()


@dataclass
class EvalWorld:
    user: object
    other: object | None = None
    prompt_vars: dict = field(default_factory=dict)


def seed_eval_world() -> EvalWorld:
    module = resolve_eval_module()
    if module is None:
        return _default_seed()
    seeder = getattr(module, "seed_eval_world", None)
    if not callable(seeder):
        raise ImproperlyConfigured(
            "AI_AGENT.EVAL_MODULE must define seed_eval_world()."
        )
    return seeder()


def snapshot_world(world: EvalWorld, kind: str | None) -> dict:
    module = resolve_eval_module()
    if module is None:
        return {}
    snap = getattr(module, "snapshot_world", None)
    if not callable(snap):
        return {}
    return snap(world, kind) or {}


def render_prompt(case_prompt: str, world: EvalWorld) -> str:
    return case_prompt.format(**(world.prompt_vars or {}))


def _default_seed() -> EvalWorld:
    user = User.objects.create_user(
        username="eval_user",
        password="pass12345",
        first_name="Eval",
        last_name="User",
    )
    other = User.objects.create_user(
        username="eval_other",
        password="pass12345",
        first_name="Other",
        last_name="Person",
    )
    return EvalWorld(user=user, other=other)
