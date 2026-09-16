"""Backfill per-user memory reconcile + local optimize."""

from django.core.management.base import BaseCommand, CommandError

from ai_agent.conf import memory_curator_enabled
from ai_agent.memory.namespaces import SUPERVISOR_LAYER


class Command(BaseCommand):
    help = (
        "Run the local memory curator for one user and layer (or all layers). "
        "Reconciles duplicates/caps, then may patch the local playbook."
    )

    def add_arguments(self, parser):
        parser.add_argument("--user-id", required=True, help="Django user id")
        parser.add_argument(
            "--layer",
            default="",
            help="Memory layer (supervisor, wallet, trading, ...). "
            "Ignored when --all-layers is set.",
        )
        parser.add_argument(
            "--all-layers",
            action="store_true",
            help="Run curator for the supervisor and every exposed app.",
        )

    def handle(self, *args, **options):
        if not memory_curator_enabled():
            raise CommandError(
                "Memory curator is disabled. Enable AI_AGENT_MEMORY_ENABLED "
                "(and AI_AGENT_MEMORY_CURATOR_ENABLED unless it should follow)."
            )
        user_id = str(options["user_id"]).strip()
        if not user_id:
            raise CommandError("--user-id is required")
        layers = _layers(options)
        if not layers:
            raise CommandError("Specify --layer or --all-layers")
        from ai_agent.memory.curator import run_curator
        from ai_agent.memory.store import build_memory_store

        store = build_memory_store()
        for layer in layers:
            run_curator(user_id, layer, store=store)
            self.stdout.write(self.style.SUCCESS(f"Curated {user_id}/{layer}"))


def _layers(options) -> list[str]:
    if options["all_layers"]:
        from django.apps import apps

        labels = [SUPERVISOR_LAYER]
        for config in apps.get_app_configs():
            if config.label == "ai_agent":
                continue
            if getattr(config, "agent_expose", False) or getattr(
                config, "agent_memory_profile", None
            ):
                labels.append(config.label)
        return labels
    layer = str(options.get("layer") or "").strip()
    return [layer] if layer else []
