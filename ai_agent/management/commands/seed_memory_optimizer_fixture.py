"""Seed duplicate memories + PII-laden episodes, then optionally reconcile/optimize.

Dev/test only. Refused in production unless ``AI_AGENT_ALLOW_OPTIMIZER_FIXTURE=1``.

Use this to check:
1. The reconciler merges duplicate facts/episodes.
2. Local overlays learn standing prefs for each customer.
3. Global overlays learn shared lessons without leaking personal data.

Example::

    python manage.py seed_memory_optimizer_fixture --reset --reconcile --optimize
"""

import logging
import os

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from ai_agent.conf import (
    memory_curator_enabled,
    memory_enabled,
    memory_prompt_optimizer_enabled,
)
from ai_agent.memory.namespaces import SUPERVISOR_LAYER
from ai_agent.memory.optimizer_fixture import (
    LAYER,
    PERSON_A,
    PERSON_B,
    count_layer,
    pii_tokens,
    seed_optimizer_fixture,
    wipe_optimizer_fixture,
)
from ai_agent.memory.prompt_optimize import load_overlay, looks_like_pii
from ai_agent.memory.store import build_memory_store

logger = logging.getLogger(__name__)


def _allow_optimizer_fixture() -> bool:
    if getattr(settings, "TESTING", False) or getattr(settings, "DEBUG", False):
        return True
    raw = os.environ.get("AI_AGENT_ALLOW_OPTIMIZER_FIXTURE", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class Command(BaseCommand):
    help = (
        "Seed two fixture users with duplicate dummy-layer memories and PII in episodes, "
        "then optionally run reconcile and local/global prompt optimization."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Wipe prior fixture memories and overlays for these users/layers.",
        )
        parser.add_argument(
            "--reconcile",
            action="store_true",
            help="Run deterministic reconcile (no curator LLM) and print counts.",
        )
        parser.add_argument(
            "--optimize",
            action="store_true",
            help="Run local then global prompt optimize (needs a live memory model).",
        )

    def handle(self, *args, **options):
        if not _allow_optimizer_fixture():
            raise CommandError(
                "Refusing to seed optimizer fixtures outside DEBUG/TESTING. "
                "Set AI_AGENT_ALLOW_OPTIMIZER_FIXTURE=1 to override."
            )
        if not memory_enabled():
            raise CommandError(
                "Enable AI_AGENT_MEMORY_ENABLED (and a usable MEMORY_STORE)."
            )
        store = build_memory_store()
        if store is None:
            raise CommandError(
                "No memory store. For local runs set AI_AGENT_MEMORY_STORE=postgres "
                "with DATABASE_URI, or use sqlite/testing in-memory store."
            )
        user_a, user_b = _ensure_users()
        self.stdout.write(
            f"Fixture users: A={user_a.pk} {PERSON_A.login} "
            f"({PERSON_A.full_name}); B={user_b.pk} {PERSON_B.login} "
            f"({PERSON_B.full_name})"
        )
        if options["reset"]:
            wipe_optimizer_fixture(
                store, user_a_id=str(user_a.pk), user_b_id=str(user_b.pk)
            )
            self.stdout.write("Wiped prior fixture memories and global overlays.")
        seed_optimizer_fixture(
            store, user_a_id=str(user_a.pk), user_b_id=str(user_b.pk)
        )
        self.stdout.write(self.style.NOTICE("=== After seed ==="))
        self._print_counts(store, user_a.pk, user_b.pk)

        if options["reconcile"]:
            if not memory_curator_enabled():
                raise CommandError(
                    "Reconcile needs the curator enabled "
                    "(AI_AGENT_MEMORY_CURATOR_ENABLED follows MEMORY_ENABLED)."
                )
            from ai_agent.memory.reconcile import reconcile_layer

            for user in (user_a, user_b):
                for layer in (LAYER, SUPERVISOR_LAYER):
                    if user.pk == user_b.pk and layer == SUPERVISOR_LAYER:
                        continue
                    result = reconcile_layer(store, str(user.pk), layer)
                    self.stdout.write(
                        f"Reconcile {user.pk}/{layer}: dirty={result.dirty} "
                        f"facts={result.fact_count} episodes={result.episode_count} "
                        f"failures={len(result.signals)}"
                    )
            self.stdout.write(self.style.NOTICE("=== After reconcile ==="))
            self._print_counts(store, user_a.pk, user_b.pk)
            self.stdout.write(
                "Expect fewer semantic rows (duplicate prefers/language/Mellat card "
                "merged) and fewer episodes (whitespace duplicate collapsed)."
            )

        if not options["optimize"]:
            self.stdout.write(
                "Seeded. Re-run with --reconcile and/or --optimize "
                "(needs OPENAI_API_KEY and MEMORY_PROMPT_OPTIMIZER_ENABLED)."
            )
            return
        if not memory_prompt_optimizer_enabled():
            raise CommandError(
                "Prompt optimizer is disabled. Enable AI_AGENT_MEMORY_ENABLED "
                "and AI_AGENT_MEMORY_PROMPT_OPTIMIZER_ENABLED."
            )
        from ai_agent.memory.prompt_optimize import (
            run_global_optimize,
            run_local_optimize,
        )

        for user in (user_a, user_b):
            result = run_local_optimize(str(user.pk), LAYER, store=store)
            self._write_opt(f"local {user.pk}/{LAYER}", result)
            for name in (LAYER, SUPERVISOR_LAYER):
                overlay = load_overlay(
                    store, layer=name, scope="local", user_id=str(user.pk)
                )
                label = f"local {name} addendum"
                self._print_overlay(label, overlay.text)
                self._print_leaks(label, overlay.text, live_only=True)
        global_result = run_global_optimize(LAYER, store=store)
        self._write_opt(f"global {LAYER}", global_result)
        for name in (LAYER, SUPERVISOR_LAYER):
            global_doc = load_overlay(store, layer=name, scope="global")
            label = f"global {name} overlay"
            self._print_overlay(label, global_doc.text)
            leaks = self._print_leaks(label, global_doc.text, live_only=False)
            if name != LAYER:
                continue
            if global_result.published and not leaks:
                self.stdout.write(
                    self.style.SUCCESS(
                        "Global dummy overlay published and fixture PII tokens "
                        "were not found."
                    )
                )
            elif global_result.published and leaks:
                self.stdout.write(
                    self.style.ERROR(
                        "Global dummy overlay published but fixture PII leaked: "
                        + ", ".join(leaks)
                    )
                )
            elif looks_like_pii(global_doc.text):
                self.stdout.write(
                    self.style.ERROR("Global overlay still matches the PII regex.")
                )
            elif LAYER not in global_result.layers:
                self.stdout.write(
                    self.style.WARNING(
                        "Global dummy overlay was not published. "
                        f"Applied layers: {global_result.layers or []}"
                    )
                )

    def _print_counts(self, store, user_a_id, user_b_id):
        for user_id, layer in (
            (user_a_id, LAYER),
            (user_b_id, LAYER),
            (user_a_id, SUPERVISOR_LAYER),
        ):
            counts = count_layer(store, str(user_id), layer)
            self.stdout.write(
                f"  {user_id}/{layer}: semantic={counts['semantic']} "
                f"episodes={counts['episodes']} prompt={counts['prompt']}"
            )

    def _write_opt(self, label, result):
        if result.applied or result.published:
            self.stdout.write(
                self.style.SUCCESS(f"{label}: applied {result.layers}")
            )
        else:
            detail = result.skipped or "no change"
            self.stdout.write(self.style.WARNING(f"{label}: skipped ({detail})"))
        if result.errors:
            self.stdout.write(self.style.WARNING("  " + "; ".join(result.errors)))

    def _print_overlay(self, label, text):
        body = (text or "").strip() or "(empty)"
        self.stdout.write(f"--- {label} ---")
        self.stdout.write(body)
        self.stdout.write("------")

    def _print_leaks(self, label, text, *, live_only: bool) -> list[str]:
        haystack = (text or "").casefold()
        found = []
        for token in pii_tokens(PERSON_A, PERSON_B):
            if live_only and token in {PERSON_A.full_name, PERSON_B.full_name}:
                continue
            if token and token.casefold() in haystack:
                found.append(token)
        if found:
            self.stdout.write(
                self.style.ERROR(f"{label} leaked: " + ", ".join(found))
            )
        else:
            self.stdout.write(self.style.SUCCESS(f"{label}: no fixture PII tokens"))
        return found


_FIXTURE_PASSWORD = "optimizer-fixture"


def _ensure_users():
    return _ensure_user(PERSON_A), _ensure_user(PERSON_B)


def _ensure_user(person):
    User = get_user_model()
    username_field = User.USERNAME_FIELD
    existing = User.objects.filter(**{username_field: person.login}).first()
    if existing:
        return existing
    create_kwargs = {
        username_field: person.login,
        "first_name": person.first_name,
        "last_name": person.last_name,
    }
    if username_field != "username" and "username" in {
        field.attname for field in User._meta.concrete_fields
    }:
        create_kwargs.setdefault("username", person.login)
    return User.objects.create_user(password=_FIXTURE_PASSWORD, **create_kwargs)
