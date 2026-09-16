"""Sample-host eval seed, snapshots, and dummy-domain cases."""

from __future__ import annotations

from decimal import Decimal

from django.contrib.auth import get_user_model

from dummy.models import Item

from ai_agent.evals.cases import EvalCase
from ai_agent.evals.fixtures import EvalWorld

User = get_user_model()


def seed_eval_world() -> EvalWorld:
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
    item = Item.objects.create(
        owner=user,
        name="Sample",
        secret="alpha",
        quantity=Decimal("12.5"),
    )
    Item.objects.create(owner=other, name="OtherItem", secret="beta")
    return EvalWorld(
        user=user,
        other=other,
        prompt_vars={"item_id": item.pk, "item_name": item.name},
    )


def snapshot_world(world: EvalWorld, kind: str | None) -> dict:
    if kind == "items":
        return {
            "names": sorted(
                Item.objects.filter(owner=world.user).values_list("name", flat=True)
            )
        }
    return {}


CASES: list[EvalCase] = [
    EvalCase(
        id="route_dummy_list",
        suite="routing",
        prompt="List my catalog items.",
        expect_domains=("dummy",),
    ),
    EvalCase(
        id="route_dummy_create",
        suite="routing",
        prompt="Create a catalog item named Widget.",
        expect_domains=("dummy",),
    ),
    EvalCase(
        id="tool_dummy_list",
        suite="tools",
        prompt="List my items.",
        app_label="dummy",
        expect_tools=("dummy_item_list",),
    ),
    EvalCase(
        id="mutation_dummy_create",
        suite="mutation",
        prompt="Create an item named Widget.",
        expect_tools=("dummy_item_create",),
        mutation=True,
        require_interrupt=True,
        snapshot="items",
    ),
    EvalCase(
        id="e2e_dummy_list",
        suite="e2e",
        prompt="What items do I have?",
        expect_domains=("dummy",),
        expect_any_tools=("dummy_item_list", "call_dummy_agent"),
    ),
]
