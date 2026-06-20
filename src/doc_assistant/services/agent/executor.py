"""Agent 单步执行结果类型的 re-export（主逻辑在 ``agent_service._execute_plan_steps``）。"""

from __future__ import annotations

from doc_assistant.services.agent.schemas import AgentStepResult

__all__ = ["AgentStepResult"]
