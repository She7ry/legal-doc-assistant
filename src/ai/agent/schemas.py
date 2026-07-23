"""Agent 模块的数据结构定义。

当前 Agent 运行时使用 ReAct 生成报告、引用和工具调用轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ai.rag.schemas import Citation


class AgentPlanStepOutput(BaseModel):
    """Planner 输出的单个独立步骤；ID 由运行时生成。"""

    title: str = Field(min_length=1, max_length=120)
    instruction: str = Field(min_length=1, max_length=1200)


class AgentPlanOutput(BaseModel):
    """有边界的扁平计划，不支持依赖或递归。"""

    steps: list[AgentPlanStepOutput] = Field(min_length=2, max_length=5)


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
    """Agent 任务完整结果，由 ``run_react_agent_task`` 返回。

    ReAct 模式填充 report、steps、citations、guard_warnings、evidence 和 metadata；
    """

    task_id: str
    status: str
    objective: str
    steps: list[AgentStepResult]
    human_review_required: bool
    report: str
    citations: list[Citation]
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTaskPause:
    """A durable LangGraph interrupt waiting for user input."""

    task_id: str
    questions: list[str]


__all__ = [
    "AgentPlanOutput",
    "AgentPlanStepOutput",
    "AgentStepResult",
    "AgentTaskPause",
    "AgentTaskResult",
]
