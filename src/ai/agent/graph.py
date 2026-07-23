"""LangGraph runtime for bounded multi-step ReAct tasks."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ai.agent._helpers import _remap_metadata, _remap_source_refs
from ai.agent.planning import (
    renumber_citations,
    step_question,
    synthesize_steps,
)
from ai.agent.schemas import AgentStepResult, AgentTaskResult
from ai.agent.tool_calling import ToolCallingAnswer, ToolCallingChatService
from ai.rag.grounding.evidence import build_evidence_profile
from ai.rag.grounding.guard import AnswerGuardResult, validate_answer
from ai.rag.qa_service import DocumentQAService
from ai.rag.schemas import Citation

logger = logging.getLogger(__name__)


class _PlannedTaskState(TypedDict, total=False):
    planned_steps: list[tuple[str, str]]
    planning_mode: str
    next_step: int
    attempt: int
    retry_step: bool
    step_results: list[AgentStepResult]
    citations: list[Citation]
    citations_by_key: dict[tuple[Any, ...], Citation]
    report: str
    synthesis_warnings: list[str]
    guard_result: AnswerGuardResult
    result: AgentTaskResult


def run_planned_react_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    planned_steps: list[tuple[str, str]],
    planning_mode: str,
    user_role: str,
    max_steps: int,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    progress_callback,
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentTaskResult:
    if progress_callback:
        progress_callback(
            event_type="react_started",
            stage="answering",
            progress=10,
            message="正在执行有边界的多步骤 ReAct 流程。",
            payload={"step_count": len(planned_steps), "planning_mode": planning_mode},
        )

    def execute_step(state: _PlannedTaskState) -> _PlannedTaskState:
        index = state["next_step"]
        title, instruction = state["planned_steps"][index]
        step_id = f"step-{index + 1}"
        attempt = state["attempt"] + 1
        end_progress = 10 + int((index + 1) * 70 / len(state["planned_steps"]))
        if attempt == 1 and progress_callback:
            progress_callback(
                event_type="step_started",
                stage="answering",
                progress=10 + int(index * 70 / len(state["planned_steps"])),
                message=f"已开始：{title}。",
                step_id=step_id,
                payload={"title": title},
            )

        try:
            answer = ToolCallingChatService(qa_service).ask(
                step_question(objective, title, instruction, user_role),
                user_id=user_id,
                conversation_id=f"{conversation_id or task_id}:{step_id}",
                task_id=f"{task_id}:{step_id}",
                enable_web_search=False,
                max_tool_iterations=max_steps,
            )
        except Exception:
            logger.warning(
                "Agent step execution failed%s.",
                "; retrying once" if attempt == 1 else " after one retry",
                extra={"task_id": task_id, "step_id": step_id, "attempt": attempt},
                exc_info=True,
            )
            if attempt == 1:
                return {"attempt": attempt, "retry_step": True}

            failed_step = AgentStepResult(
                step_id=step_id,
                title=title,
                tool="tool_calling_react",
                status="failed",
                summary="该步骤重试一次后仍失败，需要人工审阅。",
                guard_warnings=["步骤重试一次后仍执行失败。"],
                output={"attempts": 2, "error": "step_execution_failed"},
            )
            if progress_callback:
                progress_callback(
                    event_type="step_failed",
                    stage="answering",
                    progress=end_progress,
                    message=f"{title} 重试一次后仍失败。",
                    step_id=step_id,
                    payload={"status": "failed", "attempts": 2},
                )
            return {
                "attempt": 0,
                "retry_step": False,
                "next_step": index + 1,
                "step_results": [*state["step_results"], failed_step],
            }

        citations = list(state["citations"])
        citations_by_key = dict(state["citations_by_key"])
        citation_mapping, step_citations = renumber_citations(
            answer.citations,
            citations_by_key,
            citations,
        )
        step = answer_step(
            answer,
            step_id=step_id,
            title=title,
            citation_mapping=citation_mapping,
            citations=step_citations,
        )
        if progress_callback:
            progress_callback(
                event_type="step_completed",
                stage="answering",
                progress=end_progress,
                message=f"已完成：{title}。",
                step_id=step_id,
                payload={"status": step.status, "citation_count": len(step.citations)},
            )
        return {
            "attempt": 0,
            "retry_step": False,
            "next_step": index + 1,
            "step_results": [*state["step_results"], step],
            "citations": citations,
            "citations_by_key": citations_by_key,
        }

    def route_after_step(state: _PlannedTaskState) -> str:
        if state["retry_step"] or state["next_step"] < len(state["planned_steps"]):
            return "execute_step"
        return "synthesize"

    def synthesize(state: _PlannedTaskState) -> _PlannedTaskState:
        report, warnings = synthesize_steps(
            qa_service,
            objective=objective,
            user_role=user_role,
            steps=state["step_results"],
            citations=state["citations"],
        )
        return {"report": report, "synthesis_warnings": warnings}

    def validate(state: _PlannedTaskState) -> _PlannedTaskState:
        return {
            "guard_result": validate_answer(
                state["report"],
                state["citations"],
                has_retrieved_documents=bool(state["citations"]),
            )
        }

    def route_after_validate(state: _PlannedTaskState) -> str:
        return "repair" if state["guard_result"].needs_repair else "finalize"

    def repair(state: _PlannedTaskState) -> _PlannedTaskState:
        try:
            report = qa_service.repair_content(
                state["report"], state["guard_result"], state["citations"]
            )
            guard_result = validate_answer(
                report,
                state["citations"],
                has_retrieved_documents=bool(state["citations"]),
            )
            return {"report": report, "guard_result": guard_result}
        except Exception:
            logger.warning("Final Agent synthesis repair failed.", exc_info=True)
            return {
                "synthesis_warnings": [
                    *state["synthesis_warnings"],
                    "最终汇总修复失败。",
                ]
            }

    def finalize(state: _PlannedTaskState) -> _PlannedTaskState:
        guard_warnings = list(
            dict.fromkeys(
                warning
                for warning in (
                    *state["synthesis_warnings"],
                    *(warning for step in state["step_results"] for warning in step.guard_warnings),
                    *state["guard_result"].issues,
                )
                if warning
            )
        )
        human_review_required = bool(guard_warnings) or any(
            step.status != "completed" for step in state["step_results"]
        )
        status = "needs_human_review" if human_review_required else "completed"
        if progress_callback:
            progress_callback(
                event_type="react_completed",
                stage="reporting",
                progress=90,
                message="多步骤 ReAct 流程已完成。",
                payload={"step_count": len(state["step_results"])},
            )
        memory_service = getattr(qa_service, "memory_service", None)
        if status == "completed" and user_id and memory_service:
            memory_service.mark_task_memories_stale(user_id, task_id)

        result = AgentTaskResult(
            task_id=task_id,
            status=status,
            objective=objective,
            steps=state["step_results"],
            human_review_required=human_review_required,
            report=state["report"],
            citations=state["citations"],
            confidence=state["guard_result"].confidence,
            guard_warnings=guard_warnings,
            evidence=build_evidence_profile(
                state["report"], state["citations"], state["guard_result"].issues
            ),
            metadata={
                "user_role": user_role,
                "runtime": "react_langgraph_bounded_v1",
                "checkpointing": checkpointer is not None,
                "planning_mode": state["planning_mode"],
                "planned_step_count": len(state["planned_steps"]),
                "user_id": user_id,
                "available_tools": ["check_conflict", "review_clause", "search_documents"],
                "tool_calls": [
                    trace["name"]
                    for step in state["step_results"]
                    for trace in step.output.get("tool_calls", [])
                    if isinstance(trace, dict) and trace.get("name")
                ],
                "max_tool_iterations": max_steps,
            },
        )
        return {"result": result}

    graph = StateGraph(_PlannedTaskState)
    graph.add_node("execute_step", execute_step)
    graph.add_node("synthesize", synthesize)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "execute_step")
    graph.add_conditional_edges(
        "execute_step",
        route_after_step,
        {"execute_step": "execute_step", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"repair": "repair", "finalize": "finalize"},
    )
    graph.add_edge("repair", "finalize")
    graph.add_edge("finalize", END)
    compiled = graph.compile(checkpointer=checkpointer)
    initial_state = {
        "planned_steps": planned_steps,
        "planning_mode": planning_mode,
        "next_step": 0,
        "attempt": 0,
        "retry_step": False,
        "step_results": [],
        "citations": [],
        "citations_by_key": {},
        "synthesis_warnings": [],
    }
    if checkpointer is None:
        final_state = compiled.invoke(initial_state)
    else:
        config = {"configurable": {"thread_id": f"{task_id}:planned"}}
        snapshot = compiled.get_state(config)
        if snapshot.values.get("result") is not None:
            return snapshot.values["result"]
        final_state = compiled.invoke(None if snapshot.next else initial_state, config=config)
    return final_state["result"]


def answer_step(
    answer: ToolCallingAnswer,
    *,
    step_id: str = "react",
    title: str = "ReAct 回答",
    citation_mapping: dict[str, str] | None = None,
    citations: list[Citation] | None = None,
) -> AgentStepResult:
    mapping = citation_mapping or {}
    evidence = answer.metadata.get("evidence") if isinstance(answer.metadata, dict) else None
    output = _remap_metadata(
        {
            "tool_calls": [asdict(trace) for trace in answer.tool_calls],
            "web_sources": [asdict(source) for source in answer.web_sources],
        },
        mapping,
    )
    return AgentStepResult(
        step_id=step_id,
        title=title,
        tool="tool_calling_react",
        status="needs_review" if answer.guard_warnings else "completed",
        summary=_remap_source_refs(answer.content, mapping),
        citations=answer.citations if citations is None else citations,
        evidence=_remap_metadata(evidence, mapping) if isinstance(evidence, dict) else None,
        guard_warnings=[_remap_source_refs(warning, mapping) for warning in answer.guard_warnings],
        output=output,
    )
