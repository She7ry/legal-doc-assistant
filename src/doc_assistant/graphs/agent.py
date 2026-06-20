"""LangGraph 法律 Agent「规划-执行」工作流状态图定义。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


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
    plan: list[Any]  # AgentPlanStep 列表
    citation_registry: Any  # _CitationRegistry，全局 [S1][S2] 编号

    # ── execute_steps 节点产出 ──
    steps: list[Any]  # AgentStepResult 列表

    # ── collect_findings 节点产出 ──
    findings: list[Any]  # AgentFinding 列表
    missing_information: list[str]

    # ── build_deliverables 节点产出 ──
    matter_profile: Any
    artifacts: list[Any]
    confirmation_gates: list[Any]

    # ── synthesize_report 节点产出 ──
    report: str
    guard_result: Any
    evidence: dict[str, Any] | None

    # ── finalize_result 节点产出 ──
    human_review_required: bool
    status: str  # completed | needs_human_review
    result: Any  # AgentTaskResult


AgentNodeFn = Callable[[AgentWorkflowState], dict[str, Any]]


def build_agent_graph(
    *,
    plan: AgentNodeFn,
    execute_steps: AgentNodeFn,
    collect_findings: AgentNodeFn,
    build_deliverables: AgentNodeFn,
    synthesize_report: AgentNodeFn,
    finalize_result: AgentNodeFn,
):
    """编译法律 Agent 工作流为 LangGraph 状态图。

    图本身只做编排；节点回调由 service 注入，以便保留现有规划、
    执行、ReAct、引用与合成逻辑，同时让流程成为显式状态机。
    """
    graph = StateGraph(AgentWorkflowState)
    # 六个线性节点，无分支跳转（ReAct 在 execute_steps 节点内部完成）
    graph.add_node("plan", plan)                       # 1. 规划
    graph.add_node("execute_steps", execute_steps)     # 2. 执行
    graph.add_node("collect_findings", collect_findings)  # 3. 汇总 finding
    graph.add_node("build_deliverables", build_deliverables)  # 4. 交付物
    graph.add_node("synthesize_report", synthesize_report)    # 5. 报告
    graph.add_node("finalize_result", finalize_result)        # 6. 终态

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_steps")
    graph.add_edge("execute_steps", "collect_findings")
    graph.add_edge("collect_findings", "build_deliverables")
    graph.add_edge("build_deliverables", "synthesize_report")
    graph.add_edge("synthesize_report", "finalize_result")
    graph.add_edge("finalize_result", END)

    return graph.compile()
