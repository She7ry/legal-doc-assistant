"""Legal Agent service entry point."""

from __future__ import annotations

from collections.abc import Callable

from doc_assistant.agent.react_task import run_react_agent_task
from doc_assistant.agent.schemas import AgentTaskResult
from doc_assistant.services.qa_service import DocumentQAService


class LegalAgentService:
    """ReAct tool-calling entry point for legal tasks."""

    def __init__(self, qa_service: DocumentQAService) -> None:
        self.qa_service = qa_service

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
        progress_callback: Callable[..., None] | None = None,
    ) -> AgentTaskResult:
        """Run an Agent task through ReAct tool-calling."""
        return run_react_agent_task(
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
        )
