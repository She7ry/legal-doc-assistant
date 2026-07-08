"""法律 Agent 核心实现：LegalAgentService 主类。

具体辅助逻辑位于同一 ``agent`` 包的私有模块：

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
- ``executor``            单步执行、重试、ReAct 动作执行

对外入口：``LegalAgentService.run_task`` → ``workflow.run_agent_workflow``（LangGraph）。
"""

from __future__ import annotations

from typing import Any

from doc_assistant.agent._helpers import (
    ProgressCallback,
)
from doc_assistant.agent.planner import plan_task
from doc_assistant.agent.schemas import (
    AgentPlanStep,
    AgentTaskResult,
)
from doc_assistant.agent.workflow import run_agent_workflow
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.skills.models import SkillRuntimeContext

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
        thread_id: str | None = None,  # P1-1: checkpointing
        resume_value: Any = None,  # P1-1: GraphInterrupt resume
    ) -> AgentTaskResult:
        """执行完整 Agent 任务（对外主入口）。

        委托 ``run_agent_workflow`` 走 LangGraph 六阶段流水线；
        ``progress_callback`` 可接收 plan_created / step_started 等 SSE 事件。
        ``thread_id`` 与 ``resume_value`` 支持 HITL 中断后恢复。
        """
        return run_agent_workflow(
            self.qa_service,
            objective=objective,
            focus_areas=focus_areas,
            user_role=user_role,
            max_steps=max_steps,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            matter_id=matter_id,
            progress_callback=progress_callback,
            thread_id=thread_id,
            resume_value=resume_value,
        )

    # ── 计划预览 ──────────────────────────────────────────────────────────

    def plan_task(
        self,
        *,
        objective: str,
        focus_areas: list[str],
        user_role: str,
        max_steps: int,
        skill_context: SkillRuntimeContext | None = None,
    ) -> list[AgentPlanStep]:
        return plan_task(
            self.qa_service.chat_model,
            objective=objective,
            focus_areas=focus_areas,
            user_role=user_role,
            max_steps=max_steps,
            skill_context=skill_context,
        )
