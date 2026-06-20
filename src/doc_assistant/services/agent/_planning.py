"""Agent 计划解析、并行/重试配置、步骤裁剪。"""

from __future__ import annotations

import json
import re
from typing import Any

from doc_assistant.config.settings import settings
from doc_assistant.services.agent._constants import AGENT_TOOL_REGISTRY
from doc_assistant.services.agent._helpers import _clean_text, _dedupe_texts
from doc_assistant.services.agent.schemas import AgentPlanStep


def _review_scope_from_plan(plan: list[AgentPlanStep]) -> list[str]:
    scope: list[str] = []
    for step in plan:
        if step.tool != "review_clause":
            continue
        clause_type = _clean_text(step.arguments.get("clause_type"))
        if clause_type:
            scope.append(clause_type)
    return _dedupe_texts(scope)


def _parse_llm_plan(response: str, max_steps: int) -> list[AgentPlanStep]:
    data = _extract_json_array(response)
    if not isinstance(data, list):
        return []

    steps: list[AgentPlanStep] = []
    seen_step_ids: set[str] = set()
    for index, item in enumerate(data[:max_steps], start=1):
        if not isinstance(item, dict):
            continue
        tool = _clean_text(item.get("tool"))
        if tool not in AGENT_TOOL_REGISTRY:
            continue
        step_id = _clean_step_id(item.get("step_id")) or f"step_{index}"
        if step_id in seen_step_ids:
            step_id = f"{step_id}_{index}"
        seen_step_ids.add(step_id)
        arguments = item.get("arguments")
        steps.append(
            AgentPlanStep(
                step_id=step_id,
                title=_clean_text(item.get("title")) or AGENT_TOOL_REGISTRY[tool]["label"],
                purpose=_clean_text(item.get("purpose")) or AGENT_TOOL_REGISTRY[tool]["description"],
                tool=tool,
                arguments=arguments if isinstance(arguments, dict) else {},
                requires_confirmation=bool(item.get("requires_confirmation", False)),
            )
        )

    if not steps:
        return []
    if steps[0].tool == "synthesize_report":
        return []
    if steps[-1].tool != "synthesize_report":
        if len(steps) >= max_steps:
            steps = steps[: max_steps - 1]
        steps.append(
            AgentPlanStep(
                step_id="report",
                title="Compile report",
                purpose="Synthesize findings, evidence, missing information, and human-review gates.",
                tool="synthesize_report",
                arguments={},
            )
        )
    return steps[:max_steps]


def _extract_json_array(content: str) -> list[Any] | None:
    text = (content or "").strip()
    if not text:
        return None
    fenced_match = re.search(
        r"```(?:json)?\s*(\[.*?\])\s*```", text, re.IGNORECASE | re.DOTALL
    )
    candidates = [fenced_match.group(1)] if fenced_match else []
    candidates.append(text)
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if 0 <= first_bracket < last_bracket:
        candidates.append(text[first_bracket : last_bracket + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _clean_step_id(value: Any) -> str:
    text = _clean_text(value).casefold().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_-]+", "_", text).strip("_-")
    return text[:80]


def _is_parallel_agent_step(step: AgentPlanStep) -> bool:
    return step.tool == "review_clause"


def _agent_max_parallel_steps() -> int:
    return max(1, int(getattr(settings, "agent_max_parallel_steps", 3)))


def _agent_retry_backoff_seconds() -> list[float]:
    raw_value = getattr(settings, "agent_step_retry_backoff_seconds", (2.0, 5.0))
    if not isinstance(raw_value, str):
        return [max(0.0, float(value)) for value in raw_value] or [2.0, 5.0]

    values: list[float] = []
    for item in raw_value.split(","):
        try:
            values.append(max(0.0, float(item.strip())))
        except ValueError:
            continue
    return values or [2.0, 5.0]


def _trim_plan(plan: list[AgentPlanStep], max_steps: int) -> list[AgentPlanStep]:
    """裁剪超长计划；始终保留末尾的 synthesize_report 汇总步骤。"""
    if len(plan) <= max_steps:
        return plan
    if not plan or plan[-1].tool != "synthesize_report":
        return plan[:max_steps]
    return [*plan[: max_steps - 1], plan[-1]]
