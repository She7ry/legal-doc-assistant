"""Agent 子包：数据结构、LangGraph 工作流编排；``LegalAgentService`` 延迟导入。"""

from __future__ import annotations

from doc_assistant.services.agent.schemas import (
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
        from doc_assistant.services.agent_service import LegalAgentService

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
]
