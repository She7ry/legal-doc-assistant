"""Agent 模块的数据结构定义。

当前 Agent 运行时使用 ReAct 生成报告，并从最终报告和已有引用中提取
可持久化的 Matter 画像、发现和交付物。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from doc_assistant.schemas.citation import Citation


class AgentPlanStepOutput(BaseModel):
    """Planner 输出的单个独立步骤；ID 由运行时生成。"""

    title: str = Field(min_length=1, max_length=120)
    instruction: str = Field(min_length=1, max_length=1200)


class AgentPlanOutput(BaseModel):
    """有边界的扁平计划，不支持依赖或递归。"""

    steps: list[AgentPlanStepOutput] = Field(min_length=2, max_length=5)


class MatterProfileOutput(BaseModel):
    """从 ReAct 报告提取的最小案件画像。"""

    document_type: str = ""
    parties: list[str] = Field(default_factory=list)
    user_side: str = ""
    governing_law: str = ""
    jurisdiction: str = ""
    key_dates: list[dict[str, Any]] = Field(default_factory=list)
    review_scope: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confirmation_gates: list[dict[str, Any]] = Field(default_factory=list)


class MatterFindingOutput(BaseModel):
    """从报告提取、随后由真实 Citation 补齐证据位置的发现。"""

    finding_id: str
    summary: str
    category: str = "Finding"
    severity: str = "Needs human review"
    recommended_action: str = ""
    citations: list[str] = Field(
        default_factory=list,
        description="Citation IDs from the supplied existing citations only.",
    )
    clause_reference: str = ""
    needs_human_review: bool = True


class MatterArtifactOutput(BaseModel):
    """MatterStore 可直接消费的最小结构化交付物。"""

    artifact_id: str
    artifact_type: str
    title: str
    summary: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    source_finding_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(
        default_factory=list,
        description="Citation IDs from the supplied existing citations only.",
    )


class MatterExtractionOutput(BaseModel):
    """一次结构化调用返回的完整 Matter 数据。"""

    matter_profile: MatterProfileOutput | None = None
    findings: list[MatterFindingOutput] = Field(default_factory=list)
    artifacts: list[MatterArtifactOutput] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentStepResult:
    """ReAct 单步执行结果，包含回答内容和工具调用轨迹。"""

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
class AgentTaskResult:
    """Agent 任务完整结果，由 ``LegalAgentService.run_task`` 返回。

    ReAct 模式填充 report、steps、citations、guard_warnings、evidence 和 metadata；
    Matter 字段来自报告完成后的一次结构化提取。
    """

    task_id: str
    status: str
    objective: str
    steps: list[AgentStepResult]
    findings: list[dict[str, Any]]
    human_review_required: bool
    report: str
    citations: list[Citation]
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    matter_profile: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentPlanOutput",
    "AgentPlanStepOutput",
    "AgentStepResult",
    "AgentTaskResult",
    "MatterArtifactOutput",
    "MatterExtractionOutput",
    "MatterFindingOutput",
    "MatterProfileOutput",
]
