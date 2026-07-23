"""Public ReAct task entry point and single-step runtime."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import traceable

from ai.agent.clarification import clarification_questions_for_task
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
from ai.agent.schemas import AgentTaskPause, AgentTaskResult
from ai.agent.tool_calling import ToolCallingChatService
from ai.observability import agent_trace_inputs, agent_trace_outputs
from ai.rag.qa_service import DocumentQAService


class _AgentRuntimeState(TypedDict, total=False):
    objective: str
    focus_areas: list[str]
    user_role: str
    max_steps: int
    user_id: str | None
    conversation_id: str | None
    task_id: str
    planned_steps: list[tuple[str, str]]
    planning_mode: str
    result: AgentTaskResult


@traceable(
    name="legal_agent_task",
    run_type="chain",
    tags=["legal-agent"],
    process_inputs=agent_trace_inputs,
    process_outputs=agent_trace_outputs,
)
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
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentTaskResult | AgentTaskPause:
    resolved_task_id = task_id or uuid4().hex
    resolved_focus_areas = focus_areas or []
    resolved_checkpointer = checkpointer or InMemorySaver()

    def clarify(state: _AgentRuntimeState) -> _AgentRuntimeState:
        current = dict(state)
        questions = clarification_questions_for_task(current["objective"], current["focus_areas"])
        while questions:
            resumed = interrupt({"type": "clarification", "questions": questions})
            if not isinstance(resumed, dict):
                raise ValueError("Agent resume input must be an object.")
            current.update(_normalize_resume_input(current, resumed))
            questions = clarification_questions_for_task(
                current["objective"], current["focus_areas"]
            )
        return current

    def plan(state: _AgentRuntimeState) -> _AgentRuntimeState:
        if not _is_l2_candidate(state["objective"], state["focus_areas"]):
            return {"planned_steps": [], "planning_mode": "single"}
        planned_steps, planning_mode = _plan_task(
            qa_service,
            objective=state["objective"],
            focus_areas=state["focus_areas"],
            user_role=state["user_role"],
        )
        return {"planned_steps": planned_steps, "planning_mode": planning_mode}

    def route_after_plan(state: _AgentRuntimeState) -> str:
        return "planned" if len(state["planned_steps"]) >= 2 else "single"

    def run_planned(state: _AgentRuntimeState) -> _AgentRuntimeState:
        return {
            "result": _run_planned_react_task(
                qa_service,
                objective=state["objective"],
                planned_steps=state["planned_steps"],
                planning_mode=state["planning_mode"],
                user_role=state["user_role"],
                max_steps=state["max_steps"],
                user_id=state.get("user_id"),
                conversation_id=state.get("conversation_id"),
                task_id=state["task_id"],
                progress_callback=progress_callback,
                checkpointer=resolved_checkpointer,
            )
        }

    def run_single(state: _AgentRuntimeState) -> _AgentRuntimeState:
        return {
            "result": _run_single_react_task(
                qa_service,
                objective=state["objective"],
                focus_areas=state["focus_areas"],
                user_role=state["user_role"],
                max_steps=state["max_steps"],
                user_id=state.get("user_id"),
                conversation_id=state.get("conversation_id"),
                task_id=state["task_id"],
                progress_callback=progress_callback,
                checkpointing=checkpointer is not None,
            )
        }

    builder = StateGraph(_AgentRuntimeState)
    builder.add_node("clarify", clarify)
    builder.add_node("plan", plan)
    builder.add_node("single", run_single)
    builder.add_node("planned", run_planned)
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "plan")
    builder.add_conditional_edges(
        "plan", route_after_plan, {"single": "single", "planned": "planned"}
    )
    builder.add_edge("single", END)
    builder.add_edge("planned", END)
    graph = builder.compile(checkpointer=resolved_checkpointer)
    config = {"configurable": {"thread_id": resolved_task_id}}
    snapshot = graph.get_state(config)
    initial_state: _AgentRuntimeState = {
        "objective": objective,
        "focus_areas": resolved_focus_areas,
        "user_role": user_role,
        "max_steps": max_steps,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "task_id": resolved_task_id,
    }
    if snapshot.values.get("result") is not None:
        return snapshot.values["result"]
    graph_input: _AgentRuntimeState | Command | None
    if any(task.interrupts for task in snapshot.tasks):
        graph_input = Command(resume=initial_state)
    elif snapshot.next:
        graph_input = None
    else:
        graph_input = initial_state
    final_state = graph.invoke(graph_input, config=config)
    if interrupts := final_state.get("__interrupt__"):
        value = interrupts[0].value
        questions = value.get("questions", []) if isinstance(value, dict) else []
        return AgentTaskPause(task_id=resolved_task_id, questions=list(questions))
    return final_state["result"]


def _normalize_resume_input(state: dict[str, Any], resumed: dict[str, Any]) -> _AgentRuntimeState:
    objective = str(resumed.get("objective") or state["objective"]).strip()
    focus_areas = [
        str(value).strip()
        for value in resumed.get("focus_areas", state["focus_areas"])
        if str(value).strip()
    ][:8]
    user_role = str(resumed.get("user_role") or state["user_role"])
    if user_role not in {"ordinary", "lawyer"}:
        user_role = state["user_role"]
    max_steps = max(3, min(10, int(resumed.get("max_steps") or state["max_steps"])))
    conversation_id = resumed.get("conversation_id", state.get("conversation_id"))
    return {
        "objective": objective,
        "focus_areas": focus_areas,
        "user_role": user_role,
        "max_steps": max_steps,
        "conversation_id": str(conversation_id) if conversation_id else None,
    }


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
    checkpointing: bool,
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
            "runtime": "react_langgraph_v1",
            "checkpointing": checkpointing,
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
