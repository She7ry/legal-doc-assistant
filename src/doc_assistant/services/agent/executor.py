"""Agent 步骤执行器：计划步骤的顺序/并行执行、重试、ReAct 微循环。

所有函数接收 ``qa_service`` 作为显式参数，不持有状态。
``LegalAgentService`` 通过薄方法委托到此处。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import Any

from doc_assistant.config.settings import settings
from doc_assistant.schemas.citation import QAAnswer
from doc_assistant.services.agent._constants import _AGENT_REACT_EXECUTABLE_TOOLS
from doc_assistant.services.agent._helpers import (
    ProgressCallback,
    _CitationRegistry,
    _append_agent_step_history,
    _call_accepts_keyword,
    _clean_text,
    _dedupe_texts,
    _emit_progress,
    _metadata_missing_information,
    _plan_step_payload,
    _remap_metadata,
    _remap_source_refs,
    _step_result_payload,
)
from doc_assistant.services.agent._planning import (
    _agent_max_parallel_steps,
    _agent_retry_backoff_seconds,
    _is_parallel_agent_step,
)
from doc_assistant.services.agent._react import (
    _agent_react_allowed_for_step,
    _agent_react_enabled,
    _agent_react_max_iterations,
    _mark_react_needs_input,
    _merge_react_action_step,
    _react_action_observation,
    _react_action_plan_step,
    _react_step_observation,
    _react_trace,
    _react_trace_item,
    _select_react_action,
    _with_react_trace,
)
from doc_assistant.services.agent.schemas import AgentPlanStep, AgentStepResult
from doc_assistant.services.evidence import build_evidence_profile


# ── 计划步骤执行（顺序 + 并行） ─────────────────────────────────────────


def execute_plan_steps(
    qa_service: Any,
    plan: list[AgentPlanStep],
    *,
    objective: str,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    citation_registry: _CitationRegistry,
    progress_callback: ProgressCallback | None,
) -> list[AgentStepResult]:
    """按计划逐步执行工具，汇总 AgentStepResult 列表。"""
    executable_steps = [step for step in plan if step.tool != "synthesize_report"]
    step_count = max(len(executable_steps), 1)
    executable_index = {
        id(step): index
        for index, step in enumerate(executable_steps, start=1)
    }

    def run_sequential(plan_step: AgentPlanStep) -> AgentStepResult:
        nonlocal step_history
        step_index = executable_index.get(id(plan_step), 1)
        _emit_step_started(
            plan_step, progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
        step = _execute_step(
            qa_service, plan_step, objective=objective, user_id=user_id,
            conversation_id=conversation_id, task_id=task_id,
            citation_registry=citation_registry, chat_history=step_history,
            progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
        step_history = _append_agent_step_history(step_history, step)
        _emit_step_completed(
            step, progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
        return step

    if not plan:
        return []

    ordered_steps: list[AgentStepResult] = []
    step_history: list[dict[str, object]] = [
        {"role": "user", "content": f"Agent objective: {objective}"}
    ]
    remaining_steps = list(plan)
    if remaining_steps and remaining_steps[0].tool != "synthesize_report":
        ordered_steps.append(run_sequential(remaining_steps.pop(0)))

    report_steps = [s for s in remaining_steps if s.tool == "synthesize_report"]
    middle_steps = [s for s in remaining_steps if s.tool != "synthesize_report"]
    parallel_steps = [s for s in middle_steps if _is_parallel_agent_step(s)]
    dependent_steps = [s for s in middle_steps if not _is_parallel_agent_step(s)]

    if len(parallel_steps) > 1 and _agent_max_parallel_steps() > 1:
        parallel_results = _execute_parallel_steps(
            qa_service, parallel_steps, objective=objective, user_id=user_id,
            conversation_id=conversation_id, task_id=task_id,
            citation_registry=citation_registry,
            progress_callback=progress_callback,
            executable_index=executable_index,
            step_count=step_count, chat_history=step_history,
        )
        ordered_steps.extend(parallel_results)
        for step in parallel_results:
            step_history = _append_agent_step_history(step_history, step)
    else:
        for plan_step in parallel_steps:
            ordered_steps.append(run_sequential(plan_step))

    for plan_step in dependent_steps:
        ordered_steps.append(run_sequential(plan_step))
    for plan_step in report_steps:
        ordered_steps.append(run_sequential(plan_step))

    return ordered_steps


def _execute_parallel_steps(
    qa_service: Any,
    plan_steps: list[AgentPlanStep],
    *,
    objective: str,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    citation_registry: _CitationRegistry,
    progress_callback: ProgressCallback | None,
    executable_index: dict[int, int],
    step_count: int,
    chat_history: list[dict[str, object]],
) -> list[AgentStepResult]:
    for plan_step in plan_steps:
        _emit_step_started(
            plan_step, progress_callback=progress_callback,
            step_index=executable_index.get(id(plan_step), 1),
            step_count=step_count,
        )

    raw_results: dict[str, QAAnswer | AgentStepResult] = {}
    max_workers = min(_agent_max_parallel_steps(), len(plan_steps))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _execute_step_raw_with_retry, qa_service, plan_step,
                objective=objective, user_id=user_id,
                conversation_id=conversation_id, task_id=task_id,
                chat_history=list(chat_history),
            ): plan_step
            for plan_step in plan_steps
        }
        for future in as_completed(futures):
            plan_step = futures[future]
            raw_results[plan_step.step_id] = future.result()

    ordered_results: list[AgentStepResult] = []
    for plan_step in plan_steps:
        step_index = executable_index.get(id(plan_step), 1)
        step = _finalize_step_execution(
            plan_step, raw_results[plan_step.step_id], citation_registry,
        )
        step = _run_react_micro_loop(
            qa_service, plan_step, step, objective=objective, user_id=user_id,
            conversation_id=conversation_id, task_id=task_id,
            citation_registry=citation_registry, chat_history=chat_history,
            progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
        ordered_results.append(step)
        _emit_step_completed(
            step, progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
    return ordered_results


# ── 进度事件 ─────────────────────────────────────────────────────────────


def _emit_step_started(
    plan_step: AgentPlanStep, *,
    progress_callback: ProgressCallback | None,
    step_index: int, step_count: int,
) -> None:
    if plan_step.tool == "synthesize_report":
        return
    _emit_progress(
        progress_callback, event_type="step_started",
        stage=plan_step.step_id,
        progress=10 + int((step_index - 1) / step_count * 70),
        message=f"Started step: {plan_step.title}",
        step_id=plan_step.step_id,
        payload={"step": _plan_step_payload(plan_step)},
    )


def _emit_step_completed(
    step: AgentStepResult, *,
    progress_callback: ProgressCallback | None,
    step_index: int, step_count: int,
) -> None:
    if step.tool == "synthesize_report":
        return
    _emit_progress(
        progress_callback, event_type="step_completed",
        stage=step.step_id,
        progress=15 + int(step_index / step_count * 70),
        message=f"Completed step: {step.title}",
        step_id=step.step_id,
        payload={"step": _step_result_payload(step)},
    )


def _emit_react_action_started(
    plan_step: AgentPlanStep, action: dict[str, Any], *,
    progress_callback: ProgressCallback | None,
    step_index: int, step_count: int,
) -> None:
    _emit_progress(
        progress_callback, event_type="react_action_started",
        stage=plan_step.step_id,
        progress=15 + int(step_index / max(step_count, 1) * 65),
        message=f"ReAct action selected for {plan_step.title}: {action['tool']}",
        step_id=plan_step.step_id, payload={"action": action},
    )


def _emit_react_action_completed(
    plan_step: AgentPlanStep, action: dict[str, Any],
    action_step: AgentStepResult, *,
    progress_callback: ProgressCallback | None,
    step_index: int, step_count: int,
) -> None:
    _emit_progress(
        progress_callback, event_type="react_action_completed",
        stage=plan_step.step_id,
        progress=18 + int(step_index / max(step_count, 1) * 65),
        message=f"ReAct action completed for {plan_step.title}: {action['tool']}",
        step_id=plan_step.step_id,
        payload={"action": action, "observation": _react_action_observation(action_step)},
    )


# ── ReAct 微循环 ─────────────────────────────────────────────────────────


def _run_react_micro_loop(
    qa_service: Any,
    plan_step: AgentPlanStep, step: AgentStepResult, *,
    objective: str, user_id: str | None, conversation_id: str | None,
    task_id: str, citation_registry: _CitationRegistry,
    chat_history: list[dict[str, object]],
    progress_callback: ProgressCallback | None,
    step_index: int, step_count: int,
) -> AgentStepResult:
    """单步执行后的受控 ReAct 微循环：证据不足时追加 document_qa 或 ask_user。"""
    if (
        not _agent_react_enabled()
        or not _agent_react_allowed_for_step(plan_step)
        or plan_step.tool == "synthesize_report"
        or step.status == "failed"
    ):
        return step

    max_iterations = _agent_react_max_iterations()
    if max_iterations <= 0:
        return step

    current_step = step
    trace = _react_trace(current_step)
    for iteration in range(max_iterations):
        observation = _react_step_observation(current_step)
        action = _select_react_action(
            plan_step, current_step, observation,
            iteration=iteration, max_iterations=max_iterations,
        )
        if action["tool"] == "finalize_report":
            break
        if action["tool"] == "ask_user":
            trace.append(_react_trace_item(
                iteration=iteration, observation=observation,
                action=action, action_step=None,
            ))
            current_step = _mark_react_needs_input(current_step, action, trace)
            break
        if action["tool"] not in _AGENT_REACT_EXECUTABLE_TOOLS:
            break

        before_citation_count = len(current_step.citations)
        _emit_react_action_started(
            plan_step, action, progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
        action_step = _execute_react_action(
            qa_service, plan_step, action, iteration=iteration,
            objective=objective, user_id=user_id,
            conversation_id=conversation_id, task_id=task_id,
            citation_registry=citation_registry, chat_history=chat_history,
        )
        _emit_react_action_completed(
            plan_step, action, action_step,
            progress_callback=progress_callback,
            step_index=step_index, step_count=step_count,
        )
        trace.append(_react_trace_item(
            iteration=iteration, observation=observation,
            action=action, action_step=action_step,
        ))
        current_step = _merge_react_action_step(current_step, action_step, trace)
        if len(current_step.citations) <= before_citation_count:
            break

    return _with_react_trace(current_step, trace) if trace else current_step


def _execute_react_action(
    qa_service: Any,
    plan_step: AgentPlanStep, action: dict[str, Any], *,
    iteration: int, objective: str, user_id: str | None,
    conversation_id: str | None, task_id: str,
    citation_registry: _CitationRegistry,
    chat_history: list[dict[str, object]],
) -> AgentStepResult:
    action_plan_step = _react_action_plan_step(plan_step, action, iteration=iteration)
    raw_result = _execute_step_raw_with_retry(
        qa_service, action_plan_step, objective=objective, user_id=user_id,
        conversation_id=conversation_id, task_id=task_id,
        chat_history=chat_history,
    )
    return _finalize_step_execution(action_plan_step, raw_result, citation_registry)


# ── 单步执行与重试 ───────────────────────────────────────────────────────


def _execute_step(
    qa_service: Any,
    plan_step: AgentPlanStep, *,
    objective: str, user_id: str | None, conversation_id: str | None,
    task_id: str, citation_registry: _CitationRegistry,
    chat_history: list[dict[str, object]],
    progress_callback: ProgressCallback | None,
    step_index: int, step_count: int,
) -> AgentStepResult:
    raw_result = _execute_step_raw_with_retry(
        qa_service, plan_step, objective=objective, user_id=user_id,
        conversation_id=conversation_id, task_id=task_id,
        chat_history=chat_history,
    )
    step = _finalize_step_execution(plan_step, raw_result, citation_registry)
    return _run_react_micro_loop(
        qa_service, plan_step, step, objective=objective, user_id=user_id,
        conversation_id=conversation_id, task_id=task_id,
        citation_registry=citation_registry, chat_history=chat_history,
        progress_callback=progress_callback,
        step_index=step_index, step_count=step_count,
    )


def _execute_step_raw_with_retry(
    qa_service: Any,
    plan_step: AgentPlanStep, *,
    objective: str, user_id: str | None, conversation_id: str | None,
    task_id: str, chat_history: list[dict[str, object]],
) -> QAAnswer | AgentStepResult:
    max_retries = max(0, int(getattr(settings, "agent_step_max_retries", 2)))
    backoff_seconds = _agent_retry_backoff_seconds()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _execute_step_raw(
                qa_service, plan_step, objective=objective, user_id=user_id,
                conversation_id=conversation_id, task_id=task_id,
                chat_history=chat_history,
            )
        except (RuntimeError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < max_retries:
                sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])

    return AgentStepResult(
        step_id=plan_step.step_id, title=plan_step.title,
        tool=plan_step.tool, status="failed",
        summary=f"Step failed after {max_retries + 1} attempt(s): {last_error or 'unknown error'}",
        output={"error": str(last_error or "unknown error")},
    )


def _execute_step_raw(
    qa_service: Any,
    plan_step: AgentPlanStep, *,
    objective: str, user_id: str | None, conversation_id: str | None,
    task_id: str, chat_history: list[dict[str, object]],
) -> QAAnswer | AgentStepResult:
    if plan_step.tool in ("document_qa", "extract_parties_dates_jurisdiction"):
        return _ask_agent_question(
            qa_service,
            str(plan_step.arguments["question"]),
            chat_history=chat_history, user_id=user_id,
            conversation_id=conversation_id, task_id=task_id,
        )

    if plan_step.tool == "review_clause":
        return qa_service.review_clause(
            clause_type=str(plan_step.arguments["clause_type"]),
            top_k=int(plan_step.arguments.get("top_k") or 5),
        )

    if plan_step.tool == "check_conflict":
        return qa_service.check_conflict(
            contract_query=str(plan_step.arguments["contract_query"]),
            policy_query=str(plan_step.arguments["policy_query"]),
            top_k=int(plan_step.arguments.get("top_k") or 5),
        )

    tool_prompts: dict[str, str] = {
        "compare_document_versions": (
            "Compare the available document versions or drafts relevant to this task. "
            "Identify changed obligations, risk allocation, dates, parties, governing law, "
            "and negotiation impact. Cite every changed position: "
            "{query}"
        ),
        "create_obligation_calendar": (
            "Extract a structured obligation calendar from the cited documents. "
            "For each item include obligation, trigger, deadline, owner if stated, "
            "status, and source citation. If a field is not stated, say it is missing. "
            "Task: {query}"
        ),
        "suggest_clause_revision": (
            "Suggest a revised clause position for the requested legal issue. "
            "Do not invent facts. Tie each drafting suggestion to the current cited clause "
            "and flag points requiring lawyer approval. "
            "Clause type: {clause_type}. Task: {objective}"
        ),
        "build_evidence_profile": (
            "Build an evidence profile for the task. List material claims, source "
            "citations, exact quoted support, support level, and unsupported reasons. "
            "Task: {objective}"
        ),
        "generate_negotiation_checklist": (
            "Generate a negotiation checklist from the cited contract excerpts. "
            "For each issue include the ask, fallback position, priority, owner, and "
            "source citation. Flag any item requiring lawyer approval. "
            "Task: {objective}"
        ),
    }
    if plan_step.tool in tool_prompts:
        query = str(plan_step.arguments.get("query") or objective)
        clause_type = str(plan_step.arguments.get("clause_type") or "requested clause")
        prompt_text = tool_prompts[plan_step.tool].format(
            query=query, objective=objective, clause_type=clause_type,
        )
        return _ask_agent_question(
            qa_service, prompt_text,
            chat_history=chat_history, user_id=user_id,
            conversation_id=conversation_id, task_id=task_id,
        )

    if plan_step.tool == "synthesize_report":
        return AgentStepResult(
            step_id=plan_step.step_id, title=plan_step.title,
            tool=plan_step.tool, status="completed",
            summary=f"Prepared the final report for: {objective}", output={},
        )

    return AgentStepResult(
        step_id=plan_step.step_id, title=plan_step.title,
        tool=plan_step.tool, status="failed",
        summary=f"Unknown agent tool: {plan_step.tool}",
        output={"error": f"Unknown agent tool: {plan_step.tool}"},
    )


def _ask_agent_question(
    qa_service: Any,
    question: str, *,
    chat_history: list[dict[str, object]], user_id: str | None,
    conversation_id: str | None, task_id: str,
) -> QAAnswer:
    kwargs: dict[str, object] = {
        "chat_history": chat_history, "user_id": user_id,
        "conversation_id": conversation_id, "task_id": task_id,
    }
    if _call_accepts_keyword(qa_service.ask, "merge_persisted_history"):
        kwargs["merge_persisted_history"] = False
    return qa_service.ask(question, **kwargs)


# ── 步骤结果转换 ─────────────────────────────────────────────────────────


def _finalize_step_execution(
    plan_step: AgentPlanStep,
    raw_result: QAAnswer | AgentStepResult,
    citation_registry: _CitationRegistry,
) -> AgentStepResult:
    if isinstance(raw_result, AgentStepResult):
        return raw_result
    return _answer_step(plan_step, raw_result, citation_registry)


def _answer_step(
    plan_step: AgentPlanStep, answer: QAAnswer,
    citation_registry: _CitationRegistry,
) -> AgentStepResult:
    citation_map, citations = citation_registry.add_step_citations(
        plan_step.step_id, answer.citations,
    )
    content = _remap_source_refs(answer.content, citation_map)
    metadata = _remap_metadata(answer.metadata, citation_map)
    evidence = metadata.get("evidence")
    if isinstance(evidence, dict):
        evidence = _remap_metadata(evidence, citation_map)
    elif answer.citations:
        evidence = build_evidence_profile(content, citations, answer.guard_warnings)

    missing_information = _metadata_missing_information(metadata)
    if not answer.citations and plan_step.tool != "synthesize_report":
        missing_information.append(
            f"No cited document evidence was found for step: {plan_step.title}."
        )

    status = "completed"
    if answer.guard_warnings or missing_information:
        status = "needs_review"

    return AgentStepResult(
        step_id=plan_step.step_id, title=plan_step.title,
        tool=plan_step.tool, status=status, summary=content,
        citations=citations,
        evidence=evidence if isinstance(evidence, dict) else None,
        guard_warnings=answer.guard_warnings,
        output={
            "metadata": metadata,
            "missing_information": _dedupe_texts(missing_information),
        },
    )
