"""ReAct-only Agent task adapter.

Keeps the existing Agent task API shape while routing execution through the
tool-calling ReAct loop.
"""

from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from doc_assistant.agent.schemas import AgentStepResult, AgentTaskResult
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.services.tool_calling_service import ToolCallingAnswer, ToolCallingChatService


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
    matter_id: str | None = None,
    progress_callback=None,
) -> AgentTaskResult:
    resolved_task_id = task_id or uuid4().hex
    if progress_callback:
        progress_callback(
            event_type="react_started",
            stage="answering",
            progress=10,
            message="Running ReAct tool-calling workflow.",
        )

    answer = ToolCallingChatService(qa_service).ask(
        _react_question(objective, focus_areas or [], user_role),
        user_id=user_id,
        conversation_id=conversation_id,
        task_id=resolved_task_id,
        enable_web_search=False,
        max_tool_iterations=max_steps,
    )

    if progress_callback:
        progress_callback(
            event_type="react_completed",
            stage="reporting",
            progress=90,
            message="ReAct workflow completed.",
            payload={"tool_calls": [trace.name for trace in answer.tool_calls]},
        )

    human_review_required = bool(answer.guard_warnings)
    status = "needs_human_review" if human_review_required else "completed"
    memory_service = getattr(qa_service, "memory_service", None)
    if status == "completed" and user_id and memory_service:
        memory_service.mark_task_memories_stale(
            qa_service.tenant_id,
            user_id,
            resolved_task_id,
        )

    step = _answer_step(answer)
    return AgentTaskResult(
        task_id=resolved_task_id,
        status=status,
        objective=objective,
        steps=[step],
        findings=[],
        human_review_required=human_review_required,
        report=answer.content,
        citations=answer.citations,
        confidence=answer.confidence,
        guard_warnings=answer.guard_warnings,
        evidence=answer.metadata.get("evidence") if isinstance(answer.metadata, dict) else None,
        matter_profile=None,
        artifacts=[],
        metadata={
            "user_role": user_role,
            "runtime": "react_tool_calling_v1",
            "tenant_id": qa_service.tenant_id,
            "matter_id": matter_id,
            "available_tools": ["check_conflict", "review_clause", "search_documents"],
            "tool_calls": [trace.name for trace in answer.tool_calls],
            "max_tool_iterations": max_steps,
        },
    )


def _react_question(objective: str, focus_areas: list[str], user_role: str) -> str:
    parts = [objective.strip()]
    if focus_areas:
        parts.append("Focus areas: " + ", ".join(focus_areas))
    if user_role:
        parts.append(f"Audience role: {user_role}")
    return "\n\n".join(part for part in parts if part)


def _answer_step(answer: ToolCallingAnswer) -> AgentStepResult:
    evidence = answer.metadata.get("evidence") if isinstance(answer.metadata, dict) else None
    return AgentStepResult(
        step_id="react",
        title="ReAct answer",
        tool="tool_calling_react",
        status="needs_review" if answer.guard_warnings else "completed",
        summary=answer.content,
        citations=answer.citations,
        evidence=evidence if isinstance(evidence, dict) else None,
        guard_warnings=answer.guard_warnings,
        output={
            "tool_calls": [asdict(trace) for trace in answer.tool_calls],
            "web_sources": [asdict(source) for source in answer.web_sources],
        },
    )
