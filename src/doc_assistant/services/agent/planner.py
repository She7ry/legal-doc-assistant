"""Agent 任务规划：启发式 + LLM 规划器。

提供 ``plan_task`` 作为对外入口，内部先走启发式规则生成计划，
当条件满足时尝试 LLM 规划器替代。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from doc_assistant.config.settings import settings
from doc_assistant.models.language_model import ChatModelProtocol
from doc_assistant.models.langchain_adapter import ChatOpenAICompatible
from doc_assistant.services.agent._constants import (
    AGENT_TOOL_REGISTRY,
    PLANNER_PROMPT,
    _looks_like_conflict_task,
    _resolve_focus_areas,
    _workflow_type,
)
from doc_assistant.services.agent._planning import _parse_llm_plan, _trim_plan
from doc_assistant.services.agent.schemas import AgentPlanStep

logger = logging.getLogger(__name__)


def plan_task(
    qa_service: Any,
    *,
    objective: str,
    focus_areas: list[str],
    user_role: str,
    max_steps: int,
) -> list[AgentPlanStep]:
    """启发式生成多步计划（默认规划器）。"""
    del user_role
    normalized_max_steps = max(3, min(max_steps, 10))
    workflow = _workflow_type(objective)
    wants_conflict = _looks_like_conflict_task(objective)

    special_tool_count = 0
    if workflow in {"version_comparison", "obligation_calendar", "evidence_audit"}:
        special_tool_count += 1
    if workflow in {"clause_revision", "negotiation_prep"}:
        special_tool_count += 1

    reserved_steps = 2 + (1 if wants_conflict else 0) + special_tool_count
    review_budget = max(1, normalized_max_steps - reserved_steps)
    resolved_focus_areas = _resolve_focus_areas(objective, focus_areas)[:review_budget]

    profile_tool = (
        "extract_parties_dates_jurisdiction"
        if workflow in {"version_comparison", "obligation_calendar", "evidence_audit"}
        else "document_qa"
    )

    plan = [
        AgentPlanStep(
            step_id="profile",
            title="Build matter profile",
            purpose=(
                "Identify document type, parties, governing law, dates, "
                "and immediate gaps."
            ),
            tool=profile_tool,
            arguments={
                "question": (
                    "Identify the document type, parties, governing law or jurisdiction, "
                    "important dates, and missing context relevant to this task: "
                    f"{objective}"
                )
            },
        )
    ]

    if workflow == "version_comparison":
        plan.append(
            AgentPlanStep(
                step_id="version_compare",
                title="Compare document versions",
                purpose="Identify changed legal positions across drafts or versions.",
                tool="compare_document_versions",
                arguments={"query": objective, "top_k": 8},
            )
        )
    elif workflow == "obligation_calendar":
        plan.append(
            AgentPlanStep(
                step_id="obligation_calendar",
                title="Create obligation calendar",
                purpose="Extract deadlines, triggers, owners, and follow-up obligations.",
                tool="create_obligation_calendar",
                arguments={"query": objective, "top_k": 8},
            )
        )
    else:
        for index, area in enumerate(resolved_focus_areas, start=1):
            plan.append(
                AgentPlanStep(
                    step_id=f"review_{index}",
                    title=f"Review {area}",
                    purpose=f"Assess the {area} clause or issue and produce evidence-backed risks.",
                    tool="review_clause",
                    arguments={"clause_type": area, "top_k": 5},
                )
            )

    if workflow == "clause_revision":
        target_clause = resolved_focus_areas[0] if resolved_focus_areas else "requested clause"
        plan.append(
            AgentPlanStep(
                step_id="clause_revision",
                title="Suggest clause revision",
                purpose="Draft a safer clause position from the cited review evidence.",
                tool="suggest_clause_revision",
                arguments={"clause_type": target_clause, "objective": objective},
                requires_confirmation=True,
            )
        )

    if workflow == "negotiation_prep":
        plan.append(
            AgentPlanStep(
                step_id="negotiation_checklist",
                title="Generate negotiation checklist",
                purpose="Turn reviewed risks into asks, fallbacks, and priorities.",
                tool="generate_negotiation_checklist",
                arguments={"objective": objective},
                requires_confirmation=True,
            )
        )

    if workflow == "evidence_audit":
        plan.append(
            AgentPlanStep(
                step_id="evidence_profile",
                title="Build evidence profile",
                purpose="Audit cited claims and identify unsupported statements.",
                tool="build_evidence_profile",
                arguments={"query": objective},
            )
        )

    if wants_conflict:
        plan.append(
            AgentPlanStep(
                step_id="conflict_check",
                title="Check document-policy conflicts",
                purpose="Compare contract obligations with policy or compliance excerpts.",
                tool="check_conflict",
                arguments={
                    "contract_query": f"contract obligations {objective}",
                    "policy_query": f"policy compliance requirements {objective}",
                    "top_k": 5,
                },
            )
        )

    plan.append(
        AgentPlanStep(
            step_id="report",
            title="Compile report",
            purpose="Synthesize findings, evidence, missing information, and human-review gates.",
            tool="synthesize_report",
            arguments={},
        )
    )
    heuristic_plan = _trim_plan(plan, normalized_max_steps)
    if _should_use_llm_planner(objective, focus_areas, heuristic_plan):
        llm_plan = plan_task_with_llm(
            qa_service,
            objective=objective,
            focus_areas=focus_areas,
            max_steps=normalized_max_steps,
        )
        if llm_plan:
            return llm_plan
    return heuristic_plan


def _should_use_llm_planner(
    objective: str,
    focus_areas: list[str],
    heuristic_plan: list[AgentPlanStep],
) -> bool:
    if not settings.agent_llm_planner_enabled:
        return False
    if focus_areas:
        return False
    if len(heuristic_plan) <= 2:
        return True
    lowered = objective.casefold()
    return any(
        keyword in lowered
        for keyword in (
            "gdpr", "ccpa", "hipaa", "compliance", "data processing",
            "privacy compliance", "regulatory", "合规", "监管", "个人信息",
        )
    )


def plan_task_with_llm(
    qa_service: Any,
    *,
    objective: str,
    focus_areas: list[str],
    max_steps: int,
) -> list[AgentPlanStep]:
    """用 LLM + agent_planner.txt 生成 JSON 计划。"""
    tool_descriptions = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in sorted(AGENT_TOOL_REGISTRY.items())
    )
    prompt = PLANNER_PROMPT.format(
        tool_descriptions=tool_descriptions,
        objective=objective,
        focus_areas=", ".join(focus_areas) or "None provided",
        max_steps=max_steps,
    )
    messages = [
        SystemMessage(content="You are a legal workflow planner."),
        HumanMessage(content=prompt),
    ]
    try:
        chat_model = qa_service.chat_model
        if isinstance(chat_model, BaseChatModel):
            llm = chat_model
        elif isinstance(chat_model, ChatModelProtocol):
            llm = ChatOpenAICompatible(client=chat_model)
        else:
            response = qa_service._invoke_chat_messages(
                [
                    {"role": "system", "content": "You are a legal workflow planner."},
                    {"role": "user", "content": prompt},
                ]
            )
            return _parse_llm_plan(response, max_steps)
        response_message = llm.invoke(messages)
        response = str(getattr(response_message, "content", response_message) or "")
    except Exception:
        logger.warning(
            "LLM planner failed; falling back to heuristic plan.",
            extra={"objective": objective},
            exc_info=True,
        )
        return []
    return _parse_llm_plan(response, max_steps)
