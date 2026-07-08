"""Agent 步骤执行器：Agent 计划步骤的单步执行、重试和 ReAct 动作执行。

所有函数接收 ``qa_service`` 作为显式参数，不持有状态。
完整流程由 LangGraph workflow 编排。
"""

from __future__ import annotations

from time import sleep
from typing import Any

from doc_assistant.agent._constants import _AGENT_REACT_EXECUTABLE_TOOLS
from doc_assistant.agent._helpers import (
    _call_accepts_keyword,
    _CitationRegistry,
    _dedupe_texts,
    _metadata_missing_information,
    _remap_metadata,
    _remap_source_refs,
)
from doc_assistant.agent._planning import (
    _agent_retry_backoff_seconds,
)
from doc_assistant.agent._react import (
    _mark_react_needs_input,
    _merge_react_action_step,
    _react_action_plan_step,
    _react_step_observation,
    _react_trace_item,
    _select_react_action,
)
from doc_assistant.agent.schemas import AgentPlanStep, AgentStepResult
from doc_assistant.config.settings import settings
from doc_assistant.grounding.evidence import build_evidence_profile
from doc_assistant.schemas.citation import QAAnswer
from doc_assistant.services.qa_service import DocumentQAService

# ── ReAct 动作执行 ──────────────────────────────────────────────────────


def _execute_react_action(
    qa_service: DocumentQAService,
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


# ── P0-1：单步执行（不含 ReAct）与单次 ReAct 迭代 ──────────────────────────
# 这些函数供 LangGraph 图节点调用，将原有的内联 for 循环提升为图迭代。


def execute_one_step(
    qa_service: DocumentQAService,
    plan_step: AgentPlanStep, *,
    objective: str,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    citation_registry: _CitationRegistry,
    chat_history: list[dict[str, object]],
) -> AgentStepResult:
    """执行一个计划步骤，不含 ReAct 补证尾调用。

    ReAct 迭代由图的 do_react / advance_step 循环在外部处理。
    """
    raw_result = _execute_step_raw_with_retry(
        qa_service, plan_step,
        objective=objective, user_id=user_id,
        conversation_id=conversation_id, task_id=task_id,
        chat_history=chat_history,
    )
    return _finalize_step_execution(plan_step, raw_result, citation_registry)


def execute_one_react_iteration(
    qa_service: DocumentQAService,
    plan_step: AgentPlanStep,
    current_step: AgentStepResult, *,
    iteration: int,
    objective: str,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    citation_registry: _CitationRegistry,
    chat_history: list[dict[str, object]],
) -> tuple[AgentStepResult, dict[str, Any]]:
    """执行一轮 ReAct 补证迭代。

    Returns:
        (updated_step, trace_item): 合并后的步骤结果和本轮唯一 trace 条目。
        若动作为 finalize_report 或 ask_user，返回的 step 可能未修改。
    """
    from doc_assistant.agent._react import (
        _agent_react_max_iterations,
    )

    max_iterations = _agent_react_max_iterations()
    observation = _react_step_observation(current_step)
    action = _select_react_action(
        plan_step, current_step, observation,
        iteration=iteration, max_iterations=max_iterations,
    )

    if action["tool"] not in _AGENT_REACT_EXECUTABLE_TOOLS:
        trace_item = _react_trace_item(
            iteration=iteration, observation=observation,
            action=action, action_step=None,
        )
        if action["tool"] == "ask_user":
            updated = _mark_react_needs_input(current_step, action, [trace_item])
            return updated, trace_item
        # finalize_report
        return current_step, trace_item

    action_step = _execute_react_action(
        qa_service, plan_step, action, iteration=iteration,
        objective=objective, user_id=user_id,
        conversation_id=conversation_id, task_id=task_id,
        citation_registry=citation_registry, chat_history=chat_history,
    )
    trace_item = _react_trace_item(
        iteration=iteration, observation=observation,
        action=action, action_step=action_step,
    )
    merged = _merge_react_action_step(current_step, action_step, [trace_item])
    return merged, trace_item


def _execute_step_raw_with_retry(
    qa_service: DocumentQAService,
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
    qa_service: DocumentQAService,
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
    qa_service: DocumentQAService,
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
