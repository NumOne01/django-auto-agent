"""Invoke supervisor/domain agents for live evals and collect a scorecard."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from ai_agent.conf import get_agent_settings
from ai_agent.context import AgentUserContext
from ai_agent.evals.cases import EvalCase, cases_for_suite
from ai_agent.evals.fixtures import EvalWorld, render_prompt, seed_eval_world, snapshot_world
from ai_agent.evals.scoring import (
    ROUTING_MIN_ACCURACY,
    TOOLS_MIN_ACCURACY,
    extract_interrupt_values,
    extract_tool_calls,
    extract_usage,
    score_case,
)
from ai_agent.evals.runtime import _immediate_executor, make_test_runtime


def live_eval_enabled() -> bool:
    flag = os.environ.get("AI_AGENT_LIVE_EVAL", "").strip().lower()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return flag in {"1", "true", "yes"} and bool(key)


@dataclass
class CaseResult:
    case_id: str
    suite: str
    passed: bool
    reasons: list[str]
    tools: list[str]
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    interrupt: bool
    safety: bool = False


@dataclass
class EvalReport:
    suite: str
    results: list[CaseResult] = field(default_factory=list)
    supervisor_model: str = ""
    subagent_model: str = ""

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for item in self.results if item.passed) / len(self.results)

    @property
    def safety_failures(self) -> list[CaseResult]:
        return [item for item in self.results if item.safety and not item.passed]

    def passed_thresholds(self) -> bool:
        if self.safety_failures:
            return False
        if self.suite == "routing":
            return self.accuracy >= ROUTING_MIN_ACCURACY
        if self.suite in {"tools", "e2e", "all"}:
            return self.accuracy >= TOOLS_MIN_ACCURACY
        return True


def run_suites(
    suites: list[str],
    *,
    supervisor_model: str | None = None,
    subagent_model: str | None = None,
    max_seconds: float = 0,
) -> list[EvalReport]:
    world = seed_eval_world()
    reports: list[EvalReport] = []
    for suite in suites:
        reports.append(
            run_suite(
                suite,
                world,
                supervisor_model=supervisor_model,
                subagent_model=subagent_model,
                max_seconds=max_seconds,
            )
        )
    return reports


def run_suite(
    suite: str,
    world: EvalWorld | None = None,
    *,
    supervisor_model: str | None = None,
    subagent_model: str | None = None,
    max_seconds: float = 0,
) -> EvalReport:
    world = world or seed_eval_world()
    cases = cases_for_suite(suite)
    checkpointer = InMemorySaver()
    graphs = _graphs_for_suite(
        suite,
        cases,
        checkpointer,
        supervisor_model=supervisor_model,
        subagent_model=subagent_model,
    )
    settings = get_agent_settings()
    report = EvalReport(
        suite=suite,
        supervisor_model=supervisor_model or settings.supervisor_model,
        subagent_model=subagent_model or settings.subagent_model,
    )
    for index, case in enumerate(cases):
        report.results.append(
            run_case(
                case,
                world,
                graphs,
                thread_id=f"eval-{suite}-{index}-{world.user.pk}",
                max_seconds=max_seconds,
            )
        )
    return report


def run_case(
    case: EvalCase,
    world: EvalWorld,
    graphs: dict,
    *,
    thread_id: str,
    max_seconds: float = 0,
) -> CaseResult:
    graph = graphs[case.app_label] if case.app_label else graphs["supervisor"]
    prompt = render_prompt(case.prompt, world)
    scored_case = replace(case, prompt=prompt)
    before = snapshot_world(world, case.snapshot)
    started = time.perf_counter()
    result, invoke_error = _invoke(
        graph,
        world,
        prompt,
        thread_id,
        decline=case.decline_mutation,
    )
    latency = time.perf_counter() - started
    if invoke_error:
        reasons = [f"invoke failed: {invoke_error}"]
        tools: list[str] = []
        interrupt = False
        usage = (None, None)
    else:
        reasons = score_case(
            scored_case,
            result,
            snapshot_before=before,
            snapshot_after=snapshot_world(world, case.snapshot),
        )
        tools = [item["name"] for item in extract_tool_calls(result)]
        interrupt = bool(extract_interrupt_values(result))
        usage = extract_usage(result)
    if max_seconds and latency > max_seconds:
        reasons.append(f"latency {latency:.1f}s exceeded {max_seconds}s")
    safety = bool(
        case.safety
        or case.require_interrupt
        or case.mutation
        or case.ground_numbers
    )
    return CaseResult(
        case_id=case.id,
        suite=case.suite,
        passed=not reasons,
        reasons=reasons,
        tools=tools,
        latency_s=latency,
        input_tokens=usage[0],
        output_tokens=usage[1],
        interrupt=interrupt,
        safety=safety,
    )


def _graphs_for_suite(
    suite: str,
    cases,
    checkpointer,
    *,
    supervisor_model: str | None,
    subagent_model: str | None,
) -> dict:
    common = {
        "supervisor_model": supervisor_model,
        "subagent_model": subagent_model,
    }
    agent_runtime = make_test_runtime(checkpointer=checkpointer)
    graphs: dict = {}
    if suite != "tools":
        graphs["supervisor"] = agent_runtime.supervisor(
            stub_subagents=(suite == "routing"),
            **common,
        )
    for label in {case.app_label for case in cases if case.app_label}:
        graphs[label] = agent_runtime.domain_agent(
            label,
            model=subagent_model,
        )
    return graphs


def _invoke(graph, world: EvalWorld, prompt: str, thread_id: str, *, decline: bool):
    config = {"configurable": {"thread_id": thread_id, "user_id": world.user.pk}}
    result = None
    try:
        with (
            AgentUserContext(world.user),
            patch(
                "langgraph.prebuilt.tool_node.get_executor_for_config",
                _immediate_executor,
            ),
        ):
            result = _invoke_once(
                graph, {"messages": [HumanMessage(content=prompt)]}, config
            )
            first_interrupt = extract_interrupt_values(result)
            if decline and first_interrupt:
                resumed = _invoke_once(graph, Command(resume=False), config)
                if isinstance(resumed, dict):
                    result = dict(resumed)
                    if not extract_interrupt_values(result):
                        result["__interrupt__"] = first_interrupt
                else:
                    result = resumed
        return result, None
    except Exception as exc:
        return result, exc


def _invoke_once(graph, payload, config):
    try:
        return graph.invoke(payload, config)
    except GraphInterrupt as exc:
        return {"__interrupt__": exc.args[0] if exc.args else ()}
