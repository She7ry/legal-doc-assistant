"""法律 Agent 任务各阶段使用的数据结构。

一次 Agent 任务的生命周期：
  AgentPlanStep   → 规划器输出的单步计划（调用哪个 tool、参数是什么）
  AgentStepResult → 执行器跑完一步后的结果（摘要、引用、guard 告警）
  AgentFinding    → 从步骤中提炼的风险/问题条目（供报告与 matter 入库）
  MatterProfile   → 案件画像（当事方、法域、关键日期、待澄清问题）
  AgentArtifact   → 结构化交付物（风险矩阵、谈判清单、义务日历等）
  AgentConfirmationGate → 需用户确认的人工闸门
  AgentTaskResult → 整次任务的最终返回值（报告 + 元数据）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from doc_assistant.schemas.citation import Citation


@dataclass(frozen=True)
class AgentPlanStep:
    """Agent 计划中的单个执行步骤。

    用途：``plan_task`` 生成后交给 ``_execute_plan_steps`` 逐步执行。
    ``tool`` 必须是 AGENT_TOOL_REGISTRY 里注册的工具名（如 review_clause）；
    ``arguments`` 传给该工具；``requires_confirmation=True`` 表示需用户确认才能继续。
    """

    step_id: str
    title: str
    purpose: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass(frozen=True)
class AgentStepResult:
    """单个计划步骤执行完毕后的结果。

    用途：汇总进 ``AgentTaskResult.steps``；``findings_from_step`` 从中提取
    AgentFinding；``output`` 存放工具特有的结构化 JSON（如冲突比对详情）。
    ``status`` 常见值：completed / failed / needs_input。
    """

    step_id: str
    title: str
    tool: str
    status: str
    summary: str
    citations: list[Citation] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    guard_warnings: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentFinding:
    """从审查步骤提炼出的单条风险/问题发现。

    用途：写入最终报告、生成 artifact（风险矩阵/谈判清单）、同步到 MatterStore。
    ``evidence_coverage`` / ``support_level`` 由 ``_audit_findings`` 根据引用与
    原文摘录自动评估；``needs_human_review`` 为 True 时任务可能标记为需人工复核。
    """

    finding_id: str
    category: str
    severity: str
    summary: str
    citations: list[str] = field(default_factory=list)
    recommended_action: str = ""
    needs_human_review: bool = True
    source_step_id: str = ""
    clause_reference: str = ""
    evidence_coverage: str = "missing"
    support_level: str = "missing"
    unsupported_reason: str = ""
    source_quote: str = ""
    location_label: str = ""
    human_review_status: str = "pending"
    status: str = "open"
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MatterProfile:
    """Agent 任务对应的「案件画像」——从文档与步骤中归纳的背景信息。

    用途：报告开头的 Matter summary、open_questions 驱动 confirmation gate、
    artifact 生成（如义务日历需要 key_dates）。可持久化到 MatterStore。
    """

    matter_id: str
    document_type: str = "Unknown"
    parties: list[str] = field(default_factory=list)
    user_side: str = ""
    governing_law: str = ""
    jurisdiction: str = ""
    key_dates: list[dict[str, Any]] = field(default_factory=list)
    review_scope: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    confidence: str = "Low"
    citations: list[str] = field(default_factory=list)
    source_step_id: str = ""
    confirmation_gates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentArtifact:
    """Agent 产出的结构化交付物（可导出 Markdown/DOCX）。

    常见 artifact_type：risk_matrix、lawyer_questions、negotiation_checklist、
    obligation_calendar。``items`` 是类型相关的 JSON 行列表。
    """

    artifact_id: str
    artifact_type: str
    title: str
    summary: str
    items: list[dict[str, Any]] = field(default_factory=list)
    source_finding_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfirmationGate:
    """阻塞任务自动完成的人工确认项。

    用途：缺失信息、高风险 finding、answer_guard 告警、需确认的修订建议等
    会生成 gate；``required=True`` 且 status=pending 时任务 status 为 needs_human_review。
    """

    gate_id: str
    gate_type: str
    title: str
    question: str
    status: str = "pending"
    priority: str = "normal"
    required: bool = True
    reason: str = ""
    related_finding_ids: list[str] = field(default_factory=list)
    related_artifact_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTaskResult:
    """一次 Agent 任务的完整返回（``LegalAgentService.run_task``）。

    用途：API 返回 report + findings + artifacts；前端展示引用、闸门与进度。
    ``status``：completed 或 needs_human_review；``metadata`` 含 planner/executor 版本信息。
    """

    task_id: str
    status: str
    objective: str
    plan: list[AgentPlanStep]
    steps: list[AgentStepResult]
    findings: list[AgentFinding]
    missing_information: list[str]
    human_review_required: bool
    report: str
    citations: list[Citation]
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    matter_profile: MatterProfile | None = None
    artifacts: list[AgentArtifact] = field(default_factory=list)
    confirmation_gates: list[AgentConfirmationGate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentArtifact",
    "AgentConfirmationGate",
    "AgentFinding",
    "AgentPlanStep",
    "AgentStepResult",
    "AgentTaskResult",
    "MatterProfile",
]
