"""Agent 子包：数据结构、LangGraph 工作流编排；``LegalAgentService`` 延迟导入。"""

from __future__ import annotations

from doc_assistant.agent._constants import clarification_questions_for_task
from doc_assistant.agent.schemas import (
    AgentArtifact,
    AgentConfirmationGate,
    AgentFinding,
    AgentPlanStep,
    AgentStepResult,
    AgentTaskResult,
    MatterProfile,
)


def __getattr__(name: str):
    if name == "LegalAgentService":
        from doc_assistant.agent.service import LegalAgentService

        return LegalAgentService
    raise AttributeError(name)


__all__ = [
    "AgentArtifact",
    "AgentConfirmationGate",
    "AgentFinding",
    "AgentPlanStep",
    "AgentStepResult",
    "AgentTaskResult",
    "LegalAgentService",
    "MatterProfile",
    "clarification_questions_for_task",
]
