"""LangGraph Agent 工作流的编排入口。

``run_agent_workflow`` 把 ``LegalAgentService`` 的方法包装成图节点，
具体业务逻辑仍在 ``agent_service.py``，此处只负责串联与进度回调。

P0-1 重构：execute_steps 节点分解为 prepare_execution → do_step → do_react →
advance_step 四个节点，使 ReAct 迭代对 LangGraph 可见。
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
    _append_agent_step_history,
    _dedupe_texts,
    _emit_progress,
    _plan_step_payload,
    _renumber_findings,
    _step_result_payload,
)
from doc_assistant.services.agent._matter_profile import _build_matter_profile
from doc_assistant.services.agent._planning import (
    _is_parallel_agent_step,
    _review_scope_from_plan,
)
from doc_assistant.services.agent._react import (
    _agent_react_enabled,
    _agent_react_max_iterations,
    _react_trace,
    _with_react_trace,
)
from doc_assistant.services.agent.executor import (
    execute_one_step,
    execute_one_react_iteration,
    _execute_step_raw_with_retry,
    _finalize_step_execution,
)
from doc_assistant.services.agent.schemas import AgentFinding, AgentTaskResult
from doc_assistant.services.answer_guard import validate_answer
from doc_assistant.services.evidence import build_evidence_profile

ProgressCallback = Callable[..., None]


def _executable_plan_steps(plan: list[Any]) -> list[Any]:
    """返回计划中需要实际执行的步骤（synthesize_report 保留为 no-op 占位）。"""
    return list(plan)


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
    thread_id: str | None = None,  # P1-1: checkpointing thread id
    resume_value: Any = None,  # P1-1: GraphInterrupt 恢复值
) -> AgentTaskResult:
    """通过 LangGraph 运行法律 Agent 工作流。

    图结构（P0-1 重构后）：

        plan → prepare_execution
          → [has_next?] → do_step → [needs_react?]
              → do_react → [more_react?] → do_react / advance_step
              → advance_step → [has_next?] → do_step / collect_findings
          → collect_findings → build_deliverables → synthesize_report
          → finalize_result
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
        return {
            "plan": plan,
            "citation_registry": _CitationRegistry(),
            "findings": [],
            "missing_information": [],
            "matter_profile": None,
            "artifacts": [],
            "confirmation_gates": [],
        }

    # ── P0-1：分解后的执行子图节点 ──────────────────────────────────────

    def prepare_execution_node(state: AgentWorkflowState) -> dict[str, Any]:
        executable = _executable_plan_steps(state["plan"])
        step_count = max(len(executable), 1)
        initial_history: list[dict[str, object]] = [
            {"role": "user", "content": f"Agent objective: {objective}"}
        ]
        return {
            "_exec_plan_index": 0,
            "_exec_results": [],
            "_exec_chat_history": initial_history,
            "_exec_react_iteration": -1,
            "_exec_react_trace": [],
            "_exec_step_result": None,
            "_exec_step_count": step_count,
        }

    def do_step_node(state: AgentWorkflowState) -> dict[str, Any]:
        executable = _executable_plan_steps(state["plan"])
        plan_index = state.get("_exec_plan_index", 0)
        citation_registry = state["citation_registry"]
        chat_history = list(state.get("_exec_chat_history", []))
        results = list(state.get("_exec_results", []))

        if plan_index >= len(executable):
            return {"_exec_step_result": None}

        plan_step = executable[plan_index]

        # 检测连续 review_clause 步骤 → 并行批次执行
        parallel_batch: list[Any] = []
        idx = plan_index
        while idx < len(executable) and _is_parallel_agent_step(executable[idx]):
            parallel_batch.append(executable[idx])
            idx += 1

        if len(parallel_batch) > 1:
            # 并行批次：一次性用 ThreadPoolExecutor 执行，结果批量写入
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from doc_assistant.services.agent._planning import _agent_max_parallel_steps

            batch_count = min(len(parallel_batch), _agent_max_parallel_steps())
            batch_steps = parallel_batch[:batch_count]
            remaining = parallel_batch[batch_count:]

            raw_results: dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=batch_count, thread_name_prefix="agent-step") as ex:
                futures = {
                    ex.submit(
                        _execute_step_raw_with_retry, service.qa_service, ps,
                        objective=objective, user_id=user_id,
                        conversation_id=conversation_id, task_id=resolved_task_id,
                        chat_history=list(chat_history),
                    ): ps
                    for ps in batch_steps
                }
                for future in as_completed(futures):
                    ps = futures[future]
                    raw_results[ps.step_id] = future.result()

            for ps in batch_steps:
                step_result = _finalize_step_execution(ps, raw_results[ps.step_id], citation_registry)
                results.append(step_result)
                chat_history = _append_agent_step_history(chat_history, step_result)

            # 处理剩余的（如果超过 max_parallel）
            for ps in remaining:
                step_result = execute_one_step(
                    service.qa_service, ps,
                    objective=objective, user_id=user_id,
                    conversation_id=conversation_id, task_id=resolved_task_id,
                    citation_registry=citation_registry,
                    chat_history=chat_history,
                )
                results.append(step_result)
                chat_history = _append_agent_step_history(chat_history, step_result)

            advanced_index = plan_index + len(parallel_batch)
            return {
                "_exec_results": results,
                "_exec_plan_index": advanced_index,
                "_exec_chat_history": chat_history,
                "_exec_step_result": None,  # 并行批次跳过 ReAct
                "_exec_react_iteration": -1,
                "_exec_react_trace": [],
            }

        # 单步执行（不含 ReAct）
        step_result = execute_one_step(
            service.qa_service, plan_step,
            objective=objective, user_id=user_id,
            conversation_id=conversation_id, task_id=resolved_task_id,
            citation_registry=citation_registry,
            chat_history=chat_history,
        )
        return {
            "_exec_step_result": step_result,
            "_exec_react_iteration": 0,
            "_exec_react_trace": [],
        }

    def do_react_node(state: AgentWorkflowState) -> dict[str, Any]:
        executable = _executable_plan_steps(state["plan"])
        plan_index = state.get("_exec_plan_index", 0)
        plan_step = executable[plan_index] if plan_index < len(executable) else None
        current_step = state.get("_exec_step_result")
        iteration = state.get("_exec_react_iteration", 0)
        trace = list(state.get("_exec_react_trace", []))
        citation_registry = state["citation_registry"]
        chat_history = list(state.get("_exec_chat_history", []))

        if plan_step is None or current_step is None:
            return {"_exec_react_iteration": 999}  # 强制退出

        updated_step, action = execute_one_react_iteration(
            service.qa_service, plan_step, current_step,
            iteration=iteration,
            objective=objective, user_id=user_id,
            conversation_id=conversation_id, task_id=resolved_task_id,
            citation_registry=citation_registry,
            chat_history=chat_history,
        )

        # 构建 trace 条目并更新
        from doc_assistant.services.agent._react import (
            _react_action_observation,
            _react_step_observation,
            _react_trace_item,
        )
        observation = _react_step_observation(current_step)
        action_step_input = updated_step if action["tool"] in ("document_qa", "build_evidence_profile") else None
        trace_item = _react_trace_item(
            iteration=iteration, observation=observation,
            action=action, action_step=action_step_input,
        )
        trace.append(trace_item)

        return {
            "_exec_step_result": updated_step,
            "_exec_react_iteration": iteration + 1,
            "_exec_react_trace": trace,
        }

    def advance_step_node(state: AgentWorkflowState) -> dict[str, Any]:
        executable = _executable_plan_steps(state["plan"])
        plan_index = state.get("_exec_plan_index", 0)
        step_result = state.get("_exec_step_result")
        results = list(state.get("_exec_results", []))
        chat_history = list(state.get("_exec_chat_history", []))
        trace = state.get("_exec_react_trace", [])

        if step_result is not None:
            # ReAct 结束后，把 trace 写回 step_result
            if trace:
                step_result = _with_react_trace(step_result, trace)
            results.append(step_result)
            chat_history = _append_agent_step_history(chat_history, step_result)

        next_index = plan_index + 1 if step_result is not None else plan_index
        return {
            "steps": results,
            "_exec_results": results,
            "_exec_plan_index": next_index,
            "_exec_chat_history": chat_history,
            "_exec_step_result": None,
            "_exec_react_iteration": -1,
            "_exec_react_trace": [],
        }

    # ── 后执行阶段节点（不变） ──────────────────────────────────────────

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
        return {
            "missing_information": missing_information,
            "matter_profile": matter_profile,
            "artifacts": artifacts,
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
        preliminary_report = service._render_report(
            objective=objective,
            user_role=user_role,
            steps=state["steps"],
            findings=state["findings"],
            missing_information=state["missing_information"],
            matter_profile=state["matter_profile"],
            artifacts=state["artifacts"],
            confirmation_gates=[],
        )
        guard_result = validate_answer(
            preliminary_report,
            citation_registry.citations,
            has_retrieved_documents=bool(citation_registry.citations),
        )
        evidence = build_evidence_profile(
            preliminary_report,
            citation_registry.citations,
            guard_result.issues,
        )
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
        report = service._render_report(
            objective=objective,
            user_role=user_role,
            steps=state["steps"],
            findings=state["findings"],
            missing_information=state["missing_information"],
            matter_profile=matter_profile,
            artifacts=state["artifacts"],
            confirmation_gates=confirmation_gates,
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
        guard_result = state.get("guard_result")
        guard_issues = guard_result.issues if guard_result else []

        human_review_required = (
            bool(state.get("missing_information"))
            or any(finding.needs_human_review for finding in state.get("findings", []))
            or bool(guard_issues)
            or any(gate.required for gate in state.get("confirmation_gates", []))
        )

        # P1-1: HITL 中断 —— 仅当 checkpointer 可用时触发（需要 thread_id）。
        # 无 checkpointer 时（如测试或简单调用），interrupt() 会导致 LangGraph 返回
        # 部分状态（含 __interrupt__ 而非 result），引发 KeyError。
        if human_review_required and thread_id:
            from langgraph.types import interrupt

            required_gates = [
                gate for gate in state.get("confirmation_gates", [])
                if gate.required
            ]
            interrupt_payload = {
                "reason": "confirmation_gates_require_approval",
                "gates": [
                    {
                        "gate_id": gate.gate_id,
                        "gate_type": gate.gate_type,
                        "title": gate.title,
                        "question": gate.question,
                        "priority": gate.priority,
                        "reason": gate.reason,
                    }
                    for gate in required_gates
                ],
                "missing_information": state.get("missing_information", []),
                "guard_warnings": guard_issues,
            }
            # GraphInterrupt: checkpointer 保存当前状态，调用方通过 Command(resume=...) 恢复
            user_decision = interrupt(interrupt_payload)
            if isinstance(user_decision, dict) and not user_decision.get("approved", True):
                pass  # 用户拒绝 → human_review_required 保持 True

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
            confidence=guard_result.confidence if guard_result else 0.0,
            guard_warnings=guard_result.issues if guard_result else [],
            evidence=state["evidence"],
            matter_profile=state["matter_profile"],
            artifacts=state["artifacts"],
            confirmation_gates=state["confirmation_gates"],
            metadata={
                "user_role": user_role,
                "planner": "heuristic_v2",
                "executor": "plan_react_v2",  # P0-1: graph-native ReAct
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

    # P1-1: 仅在启用 HITL（有 thread_id）时使用 checkpointer
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_agent_graph(
        plan=plan_node,
        prepare_execution=prepare_execution_node,
        do_step=do_step_node,
        do_react=do_react_node,
        advance_step=advance_step_node,
        collect_findings=collect_findings_node,
        build_deliverables=build_deliverables_node,
        synthesize_report=synthesize_report_node,
        finalize_result=finalize_result_node,
        checkpointer=InMemorySaver() if thread_id else None,
    )
    config: dict[str, Any] = {
        "recursion_limit": 50,  # 提升以容纳 ReAct 图迭代
    }
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    if resume_value is not None:
        from langgraph.types import Command
        result_state = graph.invoke(
            Command(resume=resume_value),
            config=config,
        )
    else:
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
            config=config,
        )
    return result_state["result"]
