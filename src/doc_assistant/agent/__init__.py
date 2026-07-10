"""Agent package: ReAct adapter and lazy service import."""

from __future__ import annotations

from doc_assistant.agent._constants import clarification_questions_for_task


def __getattr__(name: str):
    if name == "LegalAgentService":
        from doc_assistant.agent.service import LegalAgentService

        return LegalAgentService
    raise AttributeError(name)


__all__ = [
    "LegalAgentService",
    "clarification_questions_for_task",
]
