"""LangGraph Agent 工作流的编排入口。

``run_agent_workflow`` 把 ``LegalAgentService`` 的方法包装成 6 个图节点，
具体业务逻辑仍在 ``agent_service.py``，此处只负责串联与进度回调。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any
from uuid import uuid4

from doc_assistant.graphs.agent import AgentWorkflowState, build_agent_graph
from doc_assistant.services.agent._artifacts import _build_agent_artifacts
from doc_assistant.services.agent._confirmation_gates import (
    _build_confirmation_gates,
    _confirmation_gate_payload,
)
from doc_assistant.services.agent._constants import (
    AGENT_REACT_ACTIONS,
    AGENT_TOOL_REGISTRY,
    _workflow_type,
)
from doc_assistant.services.agent._findings import _audit_findings
from doc_assistant.services.agent._helpers import (
    _CitationRegistry,
    _dedupe_texts,
    _emit_progress,
    _plan_step_payload,
    _renumber_findings,
)
from doc_assistant.services.agent._matter_profile import _build_matter_profile
from doc_assistant.services.agent._planning import _review_scope_from_plan
from doc_assistant.services.agent._react import (
    _agent_react_enabled,
    _agent_react_max_iterations,
)
from doc_assistant.services.agent.schemas import AgentFinding, AgentTaskResult
from doc_assistant.services.answer_guard import validate_answer
from doc_assistant.services.evidence import build_evidence_profile

ProgressCallback = Callable[..., None]


def run_agent_workflow(
    service: Any,
    *,
    objective: str,
    focus_areas: list[str] | None = None,
    user_role: str = "ordinary",
    max_steps: int = 6,
    user_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    matter_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentTaskResult:
    """通过 LangGraph 运行法律 Agent 工作流。

    本模块只负责编排；具体规划、执行、ReAct、引用与报告逻辑仍由
    LegalAgentService 提供，以保持公开 API 与现有测试钩子稳定。

    状态机节点：plan → execute_steps → collect_findings
    → build_deliverables → synthesize_report → finalize_result
    """
    resolved_task_id = task_id or uuid4().hex
    resolved_matter_id = matter_id or resolved_task_id
    resolved_focus_areas = focus_areas or []

    def plan_node(_state: AgentWorkflowState) -> dict[str, Any]:
        plan = service.plan_task(
            objective=objective,
            focus_areas=resolved_focus_areas,
            user_role=user_role,
            max_steps=max_steps,
        )
        _emit_progress(
            progress_callback,
            event_type="plan_created",
            stage="planning",
            progress=10,
            message=f"Created a {len(plan)} step agent plan.",
            payload={"plan": [_plan_step_payload(step) for step in plan]},
        )
        return {"plan": plan, "citation_registry": _CitationRegistry()}

    def execute_steps_node(state: AgentWorkflowState) -> dict[str, Any]:
        steps = service._execute_plan_steps(
            state["plan"],
            objective=objective,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=resolved_task_id,
            citation_registry=state["citation_registry"],
            progress_callback=progress_callback,
        )
        return {"steps": steps}

    def collect_findings_node(state: AgentWorkflowState) -> dict[str, Any]:
        citation_registry = state["citation_registry"]
        findings: list[AgentFinding] = []
        missing_information: list[str] = []
        for step in state["steps"]:
            findings.extend(service._findings_from_step(step))
            missing_information.extend(
                _dedupe_texts(step.output.get("missing_information", []))
            )
        return {
            "findings": _audit_findings(
                _renumber_findings(findings),
                citation_registry.citations,
            ),
            "missing_information": _dedupe_texts(missing_information),
        }

    def build_deliverables_node(state: AgentWorkflowState) -> dict[str, Any]:
        missing_information = state["missing_information"]
        matter_profile = _build_matter_profile(
            matter_id=resolved_matter_id,
            objective=objective,
            review_scope=_review_scope_from_plan(state["plan"]),
            steps=state["steps"],
            missing_information=missing_information,
        )
        missing_information = _dedupe_texts(
            [*missing_information, *matter_profile.open_questions]
        )
        artifacts = _build_agent_artifacts(
            matter_profile=matter_profile,
            findings=state["findings"],
            steps=state["steps"],
            missing_information=missing_information,
            user_role=user_role,
        )
        confirmation_gates = _build_confirmation_gates(
            objective=objective,
            matter_profile=matter_profile,
            findings=state["findings"],
            missing_information=missing_information,
            guard_warnings=[],
            artifacts=artifacts,
            user_role=user_role,
        )
        matter_profile = replace(
            matter_profile,
            confirmation_gates=[
                _confirmation_gate_payload(gate)
                for gate in confirmation_gates
            ],
        )
        return {
            "missing_information": missing_information,
            "matter_profile": matter_profile,
            "artifacts": artifacts,
            "confirmation_gates": confirmation_gates,
        }

    def synthesize_report_node(state: AgentWorkflowState) -> dict[str, Any]:
        citation_registry = state["citation_registry"]
        _emit_progress(
            progress_callback,
            event_type="report_started",
            stage="reporting",
            progress=90,
            message="Compiling the final agent report.",
        )
        report = service._render_report(
            objective=objective,
            user_role=user_role,
            steps=state["steps"],
            findings=state["findings"],
            missing_information=state["missing_information"],
            matter_profile=state["matter_profile"],
            artifacts=state["artifacts"],
            confirmation_gates=state["confirmation_gates"],
        )
        guard_result = validate_answer(
            report,
            citation_registry.citations,
            has_retrieved_documents=bool(citation_registry.citations),
        )
        evidence = build_evidence_profile(
            report,
            citation_registry.citations,
            guard_result.issues,
        )
        if not guard_result.issues:
            return {
                "report": report,
                "guard_result": guard_result,
                "evidence": evidence,
            }

        confirmation_gates = _build_confirmation_gates(
            objective=objective,
            matter_profile=state["matter_profile"],
            findings=state["findings"],
            missing_information=state["missing_information"],
            guard_warnings=guard_result.issues,
            artifacts=state["artifacts"],
            user_role=user_role,
        )
        matter_profile = replace(
            state["matter_profile"],
            confirmation_gates=[
                _confirmation_gate_payload(gate)
                for gate in confirmation_gates
            ],
        )
        return {
            "report": report,
            "guard_result": guard_result,
            "evidence": evidence,
            "matter_profile": matter_profile,
            "confirmation_gates": confirmation_gates,
        }

    def finalize_result_node(state: AgentWorkflowState) -> dict[str, Any]:
        citation_registry = state["citation_registry"]
        guard_result = state["guard_result"]
        human_review_required = (
            bool(state["missing_information"])
            or any(finding.needs_human_review for finding in state["findings"])
            or bool(guard_result.issues)
            or any(gate.required for gate in state["confirmation_gates"])
        )
        status = "needs_human_review" if human_review_required else "completed"
        memory_service = getattr(service.qa_service, "memory_service", None)
        if status == "completed" and user_id and memory_service:
            memory_service.mark_task_memories_stale(
                service.qa_service.tenant_id,
                user_id,
                resolved_task_id,
            )

        result = AgentTaskResult(
            task_id=resolved_task_id,
            status=status,
            objective=objective,
            plan=state["plan"],
            steps=state["steps"],
            findings=state["findings"],
            missing_information=state["missing_information"],
            human_review_required=human_review_required,
            report=state["report"],
            citations=citation_registry.citations,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            evidence=state["evidence"],
            matter_profile=state["matter_profile"],
            artifacts=state["artifacts"],
            confirmation_gates=state["confirmation_gates"],
            metadata={
                "user_role": user_role,
                "planner": "heuristic_v2",
                "executor": "plan_react_v1",
                "tenant_id": service.qa_service.tenant_id,
                "workflow_type": _workflow_type(objective),
                "available_tools": sorted(AGENT_TOOL_REGISTRY),
                "react": {
                    "enabled": _agent_react_enabled(),
                    "max_iterations": _agent_react_max_iterations(),
                    "allowed_actions": sorted(AGENT_REACT_ACTIONS),
                    "policy": "controlled_evidence_v1",
                },
            },
        )
        return {
            "human_review_required": human_review_required,
            "status": status,
            "result": result,
        }

    graph = build_agent_graph(
        plan=plan_node,
        execute_steps=execute_steps_node,
        collect_findings=collect_findings_node,
        build_deliverables=build_deliverables_node,
        synthesize_report=synthesize_report_node,
        finalize_result=finalize_result_node,
    )
    result_state = graph.invoke(
        {
            "objective": objective,
            "focus_areas": resolved_focus_areas,
            "user_role": user_role,
            "max_steps": max_steps,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "task_id": resolved_task_id,
            "matter_id": resolved_matter_id,
        },
        config={"recursion_limit": 20},
    )
    return result_state["result"]
