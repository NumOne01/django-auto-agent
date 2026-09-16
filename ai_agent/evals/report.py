"""Print and serialize eval scorecards."""

from __future__ import annotations

import json

from ai_agent.evals.harness import EvalReport
from ai_agent.evals.scoring import ROUTING_MIN_ACCURACY, TOOLS_MIN_ACCURACY


def report_to_dict(report: EvalReport) -> dict:
    return {
        "suite": report.suite,
        "supervisor_model": report.supervisor_model,
        "subagent_model": report.subagent_model,
        "accuracy": round(report.accuracy, 4),
        "passed_thresholds": report.passed_thresholds(),
        "safety_failures": [item.case_id for item in report.safety_failures],
        "thresholds": {
            "routing": ROUTING_MIN_ACCURACY,
            "tools": TOOLS_MIN_ACCURACY,
        },
        "cases": [
            {
                "id": item.case_id,
                "suite": item.suite,
                "passed": item.passed,
                "reasons": item.reasons,
                "tools": item.tools,
                "latency_s": round(item.latency_s, 3),
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "interrupt": item.interrupt,
                "safety": item.safety,
            }
            for item in report.results
        ],
    }


def write_json(reports: list[EvalReport], path: str) -> None:
    payload = [report_to_dict(item) for item in reports]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_table(report: EvalReport) -> str:
    lines = [
        f"suite={report.suite} accuracy={report.accuracy:.0%} "
        f"threshold_ok={report.passed_thresholds()} "
        f"supervisor={report.supervisor_model or '-'} "
        f"subagent={report.subagent_model or '-'}"
    ]
    for item in report.results:
        mark = "PASS" if item.passed else "FAIL"
        extra = ""
        if item.reasons:
            extra = " | " + "; ".join(item.reasons)
        tokens = ""
        if item.input_tokens is not None:
            tokens = f" in={item.input_tokens} out={item.output_tokens}"
        lines.append(
            f"  {mark} {item.case_id} tools={item.tools} "
            f"{item.latency_s:.1f}s{tokens}{extra}"
        )
    if report.safety_failures:
        lines.append(
            "safety failures: "
            + ", ".join(item.case_id for item in report.safety_failures)
        )
    return "\n".join(lines)
