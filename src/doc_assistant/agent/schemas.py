"""Legal Agent API data structures.

The current Agent runtime is ReAct-only. Matter storage compatibility fields
remain, but the adapter only returns one ReAct result step, citations, guard
warnings, evidence, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from doc_assistant.schemas.citation import Citation


@dataclass(frozen=True)
class AgentStepResult:
    """ReAct answer step plus tool trace output."""

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
    """Full Agent task result returned by ``LegalAgentService.run_task``.

    ReAct-only execution fills report, steps, citations, guard warnings,
    evidence, and metadata. Structured findings/artifacts/gates are retained as
    empty compatibility fields.
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
    "AgentStepResult",
    "AgentTaskResult",
]
