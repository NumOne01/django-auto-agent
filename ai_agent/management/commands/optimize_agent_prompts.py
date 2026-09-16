"""Backfill or cron entrypoint for local/global prompt overlays."""

from django.core.management.base import BaseCommand, CommandError

from ai_agent.conf import memory_prompt_optimizer_enabled
from ai_agent.memory.namespaces import agent_memory_layers


class Command(BaseCommand):
    help = (
        "Optimize supervisor/domain prompt overlays from stored episodes. "
        "Local: --user-id [--layer|--all-layers]. "
        "Global (cron): --scope global [--layer|--all-layers]."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--scope",
            choices=["local", "global"],
            default="local",
            help="Per-user addendum or shared global overlay.",
        )
        parser.add_argument("--user-id", default="", help="Django user id (local scope)")
        parser.add_argument(
            "--layer",
            default="",
            help="Memory layer (supervisor, wallet, ...). "
            "Ignored when --all-layers is set.",
        )
        parser.add_argument(
            "--all-layers",
            action="store_true",
            help="Run for the supervisor and every exposed app.",
        )

    def handle(self, *args, **options):
        if not memory_prompt_optimizer_enabled():
            raise CommandError(
                "Prompt optimizer is disabled. Enable AI_AGENT_MEMORY_ENABLED "
                "(and AI_AGENT_MEMORY_PROMPT_OPTIMIZER_ENABLED unless it should follow)."
            )
        layers = _layers(options)
        if not layers:
            raise CommandError("Specify --layer or --all-layers")
        from ai_agent.memory.prompt_optimize import (
            run_global_optimize,
            run_local_optimize,
        )
        from ai_agent.memory.store import build_memory_store

        store = build_memory_store()
        scope = options["scope"]
        if scope == "global":
            for layer in layers:
                result = run_global_optimize(layer, store=store)
                self._write_result("global", layer, result)
            return
        user_id = str(options.get("user_id") or "").strip()
        if not user_id:
            raise CommandError("--user-id is required for local scope")
        for layer in layers:
            result = run_local_optimize(user_id, layer, store=store)
            self._write_result(f"local {user_id}", layer, result)

    def _write_result(self, label, layer, result):
        if result.applied or result.published:
            self.stdout.write(
                self.style.SUCCESS(f"{label}/{layer}: applied {result.layers}")
            )
        else:
            detail = result.skipped or "no change"
            self.stdout.write(f"{label}/{layer}: skipped ({detail})")
        if result.errors:
            self.stdout.write(self.style.WARNING("  " + "; ".join(result.errors)))


def _layers(options) -> list[str]:
    if options["all_layers"] or (
        options["scope"] == "global" and not str(options.get("layer") or "").strip()
    ):
        return agent_memory_layers()
    layer = str(options.get("layer") or "").strip()
    return [layer] if layer else []
