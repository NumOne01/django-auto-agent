"""Run live-model agent evals against an isolated test database."""

from django.core.management.base import BaseCommand, CommandError
from django.test.runner import DiscoverRunner

from ai_agent.evals.harness import live_eval_enabled, run_suites
from ai_agent.evals.report import format_table, write_json


class Command(BaseCommand):
    help = (
        "Score supervisor routing, domain tool-calling, mutation HITL, and E2E "
        "golden paths. Requires AI_AGENT_LIVE_EVAL=1 and OPENAI_API_KEY."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--suite",
            default="all",
            choices=["routing", "tools", "mutation", "e2e", "all"],
        )
        parser.add_argument("--supervisor", default="", help="Override supervisor model")
        parser.add_argument("--subagent", default="", help="Override subagent model")
        parser.add_argument("--json", dest="json_path", default="")
        parser.add_argument(
            "--max-seconds",
            type=float,
            default=0,
            help="Fail a case if it exceeds this many seconds (0 disables)",
        )

    def handle(self, *args, **options):
        if not live_eval_enabled():
            raise CommandError(
                "Set AI_AGENT_LIVE_EVAL=1 and OPENAI_API_KEY before running evals"
            )

        suites = (
            ["routing", "tools", "mutation", "e2e"]
            if options["suite"] == "all"
            else [options["suite"]]
        )
        supervisor = options["supervisor"] or None
        subagent = options["subagent"] or None
        kwargs = {
            "supervisor_model": supervisor,
            "subagent_model": subagent,
            "max_seconds": options["max_seconds"],
        }

        runner = DiscoverRunner(
            verbosity=int(options.get("verbosity") or 1), interactive=False
        )
        runner.setup_test_environment()
        old_config = runner.setup_databases()
        reports = []
        try:
            reports = run_suites(suites, **kwargs)
            self._write_reports(reports, options)
        finally:
            runner.teardown_databases(old_config)
            runner.teardown_test_environment()
        self._fail_if_needed(reports)

    def _write_reports(self, reports, options):
        for report in reports:
            self.stdout.write(format_table(report))
        path = options.get("json_path") or ""
        if path:
            write_json(reports, path)
            self.stdout.write(f"Wrote {path}")

    def _fail_if_needed(self, reports):
        if any(not report.passed_thresholds() for report in reports):
            raise CommandError("Agent eval thresholds failed")
