"""Public ReAct task entry point and single-step runtime."""

from __future__ import annotations

from uuid import uuid4

from ai.agent.graph import (
    answer_step as _answer_step,
)
from ai.agent.graph import (
    run_planned_react_task as _run_planned_react_task,
)
from ai.agent.planning import (
    is_l2_candidate as _is_l2_candidate,
)
from ai.agent.planning import (
    plan_task as _plan_task,
)
from ai.agent.schemas import AgentTaskResult
from ai.agent.tool_calling import ToolCallingChatService
from ai.rag.qa_service import DocumentQAService


def run_react_agent_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    focus_areas: list[str] | None = None,
    user_role: str = "ordinary",
    max_steps: int = 6,
    user_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    progress_callback=None,
) -> AgentTaskResult:
    resolved_task_id = task_id or uuid4().hex
    resolved_focus_areas = focus_areas or []
    if _is_l2_candidate(objective, resolved_focus_areas):
        planned_steps, planning_mode = _plan_task(
            qa_service,
            objective=objective,
            focus_areas=resolved_focus_areas,
            user_role=user_role,
        )
        if len(planned_steps) >= 2:
            return _run_planned_react_task(
                qa_service,
                objective=objective,
                planned_steps=planned_steps,
                planning_mode=planning_mode,
                user_role=user_role,
                max_steps=max_steps,
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=resolved_task_id,
                progress_callback=progress_callback,
            )

    return _run_single_react_task(
        qa_service,
        objective=objective,
        focus_areas=resolved_focus_areas,
        user_role=user_role,
        max_steps=max_steps,
        user_id=user_id,
        conversation_id=conversation_id,
        task_id=resolved_task_id,
        progress_callback=progress_callback,
    )


def _run_single_react_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    focus_areas: list[str],
    user_role: str,
    max_steps: int,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    progress_callback,
) -> AgentTaskResult:
    if progress_callback:
        progress_callback(
            event_type="react_started",
            stage="answering",
            progress=10,
            message="正在执行 ReAct 工具调用流程。",
        )

    answer = ToolCallingChatService(qa_service).ask(
        _react_question(objective, focus_areas, user_role),
        user_id=user_id,
        conversation_id=conversation_id,
        task_id=task_id,
        enable_web_search=False,
        max_tool_iterations=max_steps,
    )

    if progress_callback:
        progress_callback(
            event_type="react_completed",
            stage="reporting",
            progress=90,
            message="ReAct 流程已完成。",
            payload={"tool_calls": [trace.name for trace in answer.tool_calls]},
        )

    human_review_required = bool(answer.guard_warnings)
    status = "needs_human_review" if human_review_required else "completed"
    memory_service = getattr(qa_service, "memory_service", None)
    if status == "completed" and user_id and memory_service:
        memory_service.mark_task_memories_stale(user_id, task_id)

    step = _answer_step(answer)
    return AgentTaskResult(
        task_id=task_id,
        status=status,
        objective=objective,
        steps=[step],
        human_review_required=human_review_required,
        report=answer.content,
        citations=answer.citations,
        confidence=answer.confidence,
        guard_warnings=answer.guard_warnings,
        evidence=answer.metadata.get("evidence") if isinstance(answer.metadata, dict) else None,
        metadata={
            "user_role": user_role,
            "runtime": "react_tool_calling_v1",
            "user_id": user_id,
            "available_tools": ["check_conflict", "review_clause", "search_documents"],
            "tool_calls": [trace.name for trace in answer.tool_calls],
            "max_tool_iterations": max_steps,
        },
    )


def _react_question(objective: str, focus_areas: list[str], user_role: str) -> str:
    parts = [objective.strip()]
    if focus_areas:
        parts.append("关注点：" + ", ".join(focus_areas))
    if user_role:
        parts.append(f"目标读者：{user_role}")
    return "\n\n".join(part for part in parts if part)
