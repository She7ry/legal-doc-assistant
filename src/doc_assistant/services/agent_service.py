"""法律 Agent 核心实现：LegalAgentService 主类。

具体辅助逻辑已拆分到 ``services.agent`` 子包下的私有模块：

- ``_helpers``            文本清理、引用格式、进度回调、CitationRegistry
- ``_constants``          关键词表、工具注册表、任务分类
- ``_planning``           LLM 计划解析、并行/重试配置
- ``_react``              受控 ReAct 补证策略
- ``_matter_profile``     案件画像构建与信息抽取
- ``_findings``           Finding 提取与证据评估
- ``_artifacts``          风险矩阵、律师问题清单等交付物
- ``_confirmation_gates`` 人工确认闸门
- ``_report``             Markdown 报告渲染
- ``planner``             任务规划（启发式 + LLM）
- ``executor``            步骤执行、并行、重试、ReAct 微循环

对外入口：``LegalAgentService.run_task`` → ``workflow.run_agent_workflow``（LangGraph）。
"""

from __future__ import annotations

from doc_assistant.services.agent._findings import findings_from_step
from doc_assistant.services.agent._helpers import (
    ProgressCallback,
    _CitationRegistry,
)
from doc_assistant.services.agent._report import render_agent_report
from doc_assistant.services.agent.executor import execute_plan_steps
from doc_assistant.services.agent.planner import plan_task
from doc_assistant.services.agent.schemas import (
    AgentArtifact,
    AgentConfirmationGate,
    AgentFinding,
    AgentPlanStep,
    AgentStepResult,
    AgentTaskResult,
    MatterProfile,
)
from doc_assistant.services.agent.workflow import run_agent_workflow
from doc_assistant.services.qa_service import DocumentQAService


# ══════════════════════════════════════════════════════════════════════════════
# LegalAgentService — 任务规划、执行、报告（对外主类）
# ══════════════════════════════════════════════════════════════════════════════


class LegalAgentService:
    """面向复杂法律任务的 Agent 编排器。

    典型场景：用户给出 objective（如「审查这份 MSA 的付款与终止条款」），
    本类会：规划多步 → 逐步调用 document_qa / review_clause 等工具 →
    汇总 finding 与 artifact → 生成 Markdown 报告。

    设计要点：
    - 所有结论必须带 [Sx] 引用，由 _CitationRegistry 统一编号
    - 证据不足时可走受控 ReAct 补检索
    - 缺失信息或 guard 告警时标记 needs_human_review

    对外入口：``run_task()``；内部通过 LangGraph workflow 串联六个阶段。
    """

    def __init__(self, qa_service: DocumentQAService) -> None:
        self.qa_service = qa_service

    # ── 对外入口 ──────────────────────────────────────────────────────────

    def run_task(
        self,
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
        """执行完整 Agent 任务（对外主入口）。

        委托 ``run_agent_workflow`` 走 LangGraph 六阶段流水线；
        ``progress_callback`` 可接收 plan_created / step_started 等 SSE 事件。
        """
        return run_agent_workflow(
            self,
            objective=objective,
            focus_areas=focus_areas,
            user_role=user_role,
            max_steps=max_steps,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            matter_id=matter_id,
            progress_callback=progress_callback,
        )

    # ── workflow 节点委托方法 ─────────────────────────────────────────────
    # 这些方法由 workflow.py 中的图节点调用，内部委托到独立模块。

    def plan_task(
        self,
        *,
        objective: str,
        focus_areas: list[str],
        user_role: str,
        max_steps: int,
    ) -> list[AgentPlanStep]:
        return plan_task(
            self.qa_service,
            objective=objective,
            focus_areas=focus_areas,
            user_role=user_role,
            max_steps=max_steps,
        )

    def _execute_plan_steps(
        self,
        plan: list[AgentPlanStep],
        *,
        objective: str,
        user_id: str | None,
        conversation_id: str | None,
        task_id: str,
        citation_registry: _CitationRegistry,
        progress_callback: ProgressCallback | None,
    ) -> list[AgentStepResult]:
        return execute_plan_steps(
            self.qa_service, plan,
            objective=objective,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            citation_registry=citation_registry,
            progress_callback=progress_callback,
        )

    def _findings_from_step(self, step: AgentStepResult) -> list[AgentFinding]:
        return findings_from_step(step)

    def _render_report(
        self, *, objective: str, user_role: str,
        steps: list[AgentStepResult], findings: list[AgentFinding],
        missing_information: list[str], matter_profile: MatterProfile | None,
        artifacts: list[AgentArtifact],
        confirmation_gates: list[AgentConfirmationGate],
    ) -> str:
        return render_agent_report(
            objective=objective,
            user_role=user_role,
            steps=steps,
            findings=findings,
            missing_information=missing_information,
            matter_profile=matter_profile,
            artifacts=artifacts,
            confirmation_gates=confirmation_gates,
        )
