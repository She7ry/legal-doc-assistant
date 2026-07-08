"""LangGraph 法律 Agent「规划-执行」工作流状态图定义。

P0-1 优化：将 execute_steps 节点分解为 prepare_execution → do_step → do_react →
advance_step 的循环子图，使 ReAct 补证迭代对 LangGraph 可见（可 checkpoint、可 streaming）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from doc_assistant.agent._helpers import _CitationRegistry
from doc_assistant.agent.schemas import (
    AgentArtifact,
    AgentConfirmationGate,
    AgentFinding,
    AgentPlanStep,
    AgentStepResult,
    AgentTaskResult,
    MatterProfile,
)
from doc_assistant.grounding.guard import AnswerGuardResult
from doc_assistant.schemas.citation import Citation
from doc_assistant.skills.models import SkillRuntimeContext


class AgentWorkflowState(TypedDict, total=False):
    """LangGraph 在各节点间传递的 Agent 任务状态（total=False 表示字段均可选）。"""

    # ── 任务输入 ──
    objective: str  # 用户目标描述
    focus_areas: list[str]  # 希望重点审查的条款/领域
    user_role: str  # ordinary / lawyer 等，影响报告措辞
    max_steps: int  # 计划最大步数
    user_id: str | None
    conversation_id: str | None
    task_id: str
    matter_id: str
    progress_callback: Callable[..., None] | None  # SSE/回调进度

    # ── plan 节点产出 ──
    plan: list[AgentPlanStep]
    citation_registry: _CitationRegistry  # 全局 [S1][S2] 编号
    skill_context: SkillRuntimeContext  # 供审计元数据与 planner prompt 使用

    # ── execute_steps 节点产出（兼容旧单一节点） ──
    steps: list[AgentStepResult]

    # ── collect_findings 节点产出 ──
    findings: list[AgentFinding]
    missing_information: list[str]

    # ── build_deliverables 节点产出 ──
    matter_profile: MatterProfile | None
    artifacts: list[AgentArtifact]
    confirmation_gates: list[AgentConfirmationGate]

    # ── synthesize_report 节点产出 ──
    report: str
    guard_result: AnswerGuardResult
    evidence: dict[str, Any] | None

    # ── finalize_result 节点产出 ──
    human_review_required: bool
    status: Literal["completed", "needs_human_review"]
    result: AgentTaskResult

    # ── 内部执行传动字段（_exec_ 前缀，API 不可见） ──
    _exec_plan_index: int  # 当前 plan 步骤索引（0-based）
    _exec_step_result: AgentStepResult | None
    _exec_pending_results: list[AgentStepResult]  # 已并行执行、待逐步进入 ReAct 的结果
    _exec_react_iteration: int  # 当前 ReAct 迭代计数（-1 = 无活跃 ReAct）
    _exec_react_trace: list[dict[str, Any]]  # 当前步骤的 ReAct trace
    _exec_chat_history: list[dict[str, object]]  # 步骤间聊天历史
    _exec_results: list[AgentStepResult]  # 已完成步骤结果累加器
    _exec_step_count: int  # 可执行步骤总数（用于进度计算）


AgentNodeFn = Callable[[AgentWorkflowState], dict[str, Any]]

# ── 条件路由：按计划内容决定是否跳过 findings/deliverables ──────────────

_REVIEW_STEP_TOOLS = frozenset(
    {
        "review_clause",
        "check_conflict",
        "compare_document_versions",
        "create_obligation_calendar",
        "build_evidence_profile",
        "suggest_clause_revision",
        "generate_negotiation_checklist",
    }
)


def _route_after_execution(state: AgentWorkflowState) -> str:
    """若计划中包含审查类步骤则走完整流水线，否则跳过 findings/deliverables。"""
    plan = state.get("plan", [])
    needs_review = any(
        step.get("tool") in _REVIEW_STEP_TOOLS if isinstance(step, dict) else getattr(step, "tool", None) in _REVIEW_STEP_TOOLS
        for step in plan
    )
    if needs_review:
        return "collect_findings"
    return "synthesize_report"


# ── 步骤迭代路由（P0-1：ReAct 图节点化） ────────────────────────────────


def _route_after_prepare(state: AgentWorkflowState) -> str:
    """prepare_execution 之后：还有步骤 → do_step，无步骤 → 直接跳到后执行阶段。"""
    plan_index = state.get("_exec_plan_index", 0)
    step_count = state.get("_exec_step_count", 1)
    if plan_index < step_count:
        return "do_step"
    # 无可执行步骤 → 根据计划类型跳到正确的后执行节点
    return _route_after_execution(state)


def _route_after_step(state: AgentWorkflowState) -> str:
    """do_step 之后：步骤需要 ReAct 补证 → do_react，否则 → advance_step。"""
    from doc_assistant.agent._react import (
        _agent_react_enabled,
        _react_step_observation,
        _step_has_react_evidence_gap,
    )

    plan = state.get("plan", [])
    plan_index = state.get("_exec_plan_index", 0)
    if plan_index >= len(plan):
        return "advance_step"

    plan_step = plan[plan_index]
    step_result = state.get("_exec_step_result")

    # 兼容 dict（checkpoint 反序列化后）与原始 AgentPlanStep 对象
    tool = plan_step.get("tool") if isinstance(plan_step, dict) else getattr(plan_step, "tool", "")
    step_id = plan_step.get("step_id", "") if isinstance(plan_step, dict) else getattr(plan_step, "step_id", "")

    # ReAct 允许条件：全局启用 + 步骤非 profile + 非 synthesize_report + 步骤未失败
    react_allowed = (
        _agent_react_enabled()
        and (step_id != "profile")  # _agent_react_allowed_for_step 等价逻辑，兼容 dict
        and tool != "synthesize_report"
        and not (
            step_result
            and (step_result.get("status") if isinstance(step_result, dict) else getattr(step_result, "status", "")) == "failed"
        )
    )
    if not react_allowed:
        return "advance_step"

    if step_result is None:
        return "advance_step"

    # 兼容 dict（checkpoint 反序列化）
    obs_input = step_result if not isinstance(step_result, dict) else _rehydrate_step_result(step_result)
    observation = _react_step_observation(obs_input)
    if _step_has_react_evidence_gap(observation):
        return "do_react"
    return "advance_step"


def _rehydrate_step_result(raw: dict[str, Any]) -> AgentStepResult:
    """将 checkpoint 反序列化的 dict 恢复为现有步骤结果模型。"""
    citations = [
        Citation(**item) if isinstance(item, dict) else item
        for item in raw.get("citations", [])
        if isinstance(item, (Citation, dict))
    ]
    evidence = raw.get("evidence")
    output = raw.get("output")
    return AgentStepResult(
        step_id=str(raw.get("step_id", "")),
        title=str(raw.get("title", "")),
        tool=str(raw.get("tool", "")),
        status=raw.get("status", ""),
        summary=raw.get("summary", ""),
        citations=citations,
        evidence=evidence if isinstance(evidence, dict) else None,
        guard_warnings=list(raw.get("guard_warnings", [])),
        output=output if isinstance(output, dict) else {},
    )


def _route_after_react(state: AgentWorkflowState) -> str:
    """do_react 之后：还有迭代次数且证据缺口仍在 → 循环，否则 → advance_step。"""
    from doc_assistant.agent._react import _agent_react_max_iterations

    iteration = state.get("_exec_react_iteration", 0)
    max_iter = _agent_react_max_iterations()
    if iteration >= max_iter:
        return "advance_step"

    trace = state.get("_exec_react_trace", [])
    if trace:
        last = trace[-1]
        action = last.get("action", {})
        if action.get("tool") in ("finalize_report", "ask_user"):
            return "advance_step"

    return "do_react"


def _route_after_advance(state: AgentWorkflowState) -> str:
    """advance_step 之后：还有下一步 → do_step，全部完成 → 根据计划类型路由。"""
    plan_index = state.get("_exec_plan_index", 0)
    step_count = state.get("_exec_step_count", 1)
    if plan_index < step_count:
        return "do_step"
    # 所有步骤执行完成 → 跳到正确的后执行阶段
    return _route_after_execution(state)


def build_agent_graph(
    *,
    plan: AgentNodeFn,
    prepare_execution: AgentNodeFn,
    do_step: AgentNodeFn,
    do_react: AgentNodeFn,
    advance_step: AgentNodeFn,
    collect_findings: AgentNodeFn,
    build_deliverables: AgentNodeFn,
    synthesize_report: AgentNodeFn,
    finalize_result: AgentNodeFn,
    checkpointer: Any = None,
):
    """编译法律 Agent 工作流为 LangGraph 状态图。

    图结构（P0-1 重构后）：

        START → plan → prepare_execution
          → [has_next_step?]
              → do_step → [needs_react?]
                  → do_react → [more_react?]
                      → do_react（循环）
                      → advance_step
                  → advance_step → [has_next_step?]
                      → do_step（下一轮）
                      → collect_findings / synthesize_report
          → collect_findings → build_deliverables → synthesize_report
          → finalize_result → END

    ReAct 迭代对 LangGraph 完全可见：每轮迭代可被 checkpoint、可被 stream。

    Args:
        checkpointer: LangGraph checkpointer（默认 InMemorySaver）。
            用于支持 interrupt() 暂停与 Command(resume=...) 恢复。
    """

    graph = StateGraph(AgentWorkflowState)

    # ── 主要阶段节点 ──
    graph.add_node("plan", plan)
    graph.add_node("prepare_execution", prepare_execution)
    graph.add_node("do_step", do_step)
    graph.add_node("do_react", do_react)
    graph.add_node("advance_step", advance_step)
    graph.add_node("collect_findings", collect_findings)
    graph.add_node("build_deliverables", build_deliverables)
    graph.add_node("synthesize_report", synthesize_report)
    graph.add_node("finalize_result", finalize_result)

    # ── 执行子图：步骤迭代 + ReAct 微循环 ──
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "prepare_execution")

    graph.add_conditional_edges(
        "prepare_execution",
        _route_after_prepare,
        {
            "do_step": "do_step",
            "collect_findings": "collect_findings",
            "synthesize_report": "synthesize_report",
        },
    )

    graph.add_conditional_edges(
        "do_step",
        _route_after_step,
        {"do_react": "do_react", "advance_step": "advance_step"},
    )

    graph.add_conditional_edges(
        "do_react",
        _route_after_react,
        {"do_react": "do_react", "advance_step": "advance_step"},
    )

    graph.add_conditional_edges(
        "advance_step",
        _route_after_advance,
        {
            "do_step": "do_step",
            "collect_findings": "collect_findings",
            "synthesize_report": "synthesize_report",
        },
    )

    # ── 后执行流水线：findings → deliverables → report → result ──
    graph.add_edge("collect_findings", "build_deliverables")
    graph.add_edge("build_deliverables", "synthesize_report")
    graph.add_edge("synthesize_report", "finalize_result")
    graph.add_edge("finalize_result", END)

    return graph.compile(checkpointer=checkpointer)
