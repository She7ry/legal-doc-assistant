"""Agent 受控 ReAct 补证：证据不足时追加 document_qa / ask_user 等动作。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from doc_assistant.config.settings import settings
from doc_assistant.services.agent._constants import (
    AGENT_REACT_ACTIONS,
    _AGENT_REACT_EXECUTABLE_TOOLS,
)
from doc_assistant.services.agent._helpers import (
    SOURCE_REF_PATTERN,
    _as_text_list,
    _clean_text,
    _dedupe_citations,
    _dedupe_texts,
    _format_refs,
    _is_generated_no_evidence_missing,
    _step_missing_information,
)
from doc_assistant.services.agent.schemas import AgentPlanStep, AgentStepResult
from doc_assistant.services.answer_guard import validate_answer
from doc_assistant.services.evidence import build_evidence_profile


def _agent_react_enabled() -> bool:
    return settings.agent_react_enabled


def _agent_react_max_iterations() -> int:
    return max(0, min(settings.agent_react_max_iterations, 5))


def _agent_react_allowed_for_step(step: AgentPlanStep) -> bool:
    return step.step_id != "profile"


def _react_trace(step: AgentStepResult) -> list[dict[str, Any]]:
    raw_trace = step.output.get("react_trace")
    if not isinstance(raw_trace, list):
        return []
    return [item for item in raw_trace if isinstance(item, dict)]


def _react_step_observation(step: AgentStepResult) -> dict[str, Any]:
    missing_information = _step_missing_information(step)
    evidence = step.evidence if isinstance(step.evidence, dict) else {}
    missing_evidence = _as_text_list(evidence.get("missing_evidence"))
    weak_claims: list[dict[str, Any]] = []
    claims = evidence.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            support_level = _clean_text(claim.get("support_level"))
            if support_level and support_level != "direct":
                weak_claims.append(
                    {
                        "text": _clean_text(claim.get("text"))[:500],
                        "support_level": support_level,
                        "uncertainty": _clean_text(claim.get("uncertainty")),
                    }
                )
    return {
        "status": step.status,
        "citation_count": len(step.citations),
        "guard_warnings": step.guard_warnings,
        "missing_information": missing_information,
        "missing_evidence": missing_evidence,
        "weak_claims": weak_claims[:3],
    }


def _select_react_action(
    plan_step: AgentPlanStep,
    step: AgentStepResult,
    observation: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    """受控 ReAct 策略（controlled_evidence_v1）：仅在证据缺口时补检索。"""
    if not _step_has_react_evidence_gap(observation):
        missing_information = _as_text_list(observation.get("missing_information"))
        if missing_information and iteration >= max_iterations - 1:
            return {
                "tool": "ask_user",
                "reason": "The step still depends on user- or matter-specific missing information.",
                "arguments": {
                    "question": missing_information[0],
                    "missing_information": missing_information[:5],
                },
            }
        return {
            "tool": "finalize_report",
            "reason": "The step has no open evidence gap that a controlled action can resolve.",
            "arguments": {},
        }

    if iteration >= max_iterations:
        return {
            "tool": "ask_user",
            "reason": "The controlled ReAct action budget was exhausted before evidence was complete.",
            "arguments": {
                "question": "Confirm or provide the missing source evidence for this step.",
                "missing_information": _as_text_list(observation.get("missing_information"))[:5],
            },
        }

    tool = "document_qa"
    if observation.get("citation_count") and (
        observation.get("guard_warnings")
        or (observation.get("status") == "needs_review" and observation.get("weak_claims"))
    ):
        tool = "build_evidence_profile"

    if tool not in AGENT_REACT_ACTIONS:
        tool = "document_qa"
    question = _react_evidence_question(plan_step, step, observation, tool=tool)
    return {
        "tool": tool,
        "reason": "The step observation shows missing, weak, or uncited evidence.",
        "arguments": {"question": question, "allowed_actions": sorted(AGENT_REACT_ACTIONS)},
    }


def _step_has_react_evidence_gap(observation: dict[str, Any]) -> bool:
    """判断步骤结果是否存在需要 ReAct 补证的证据缺口。"""
    if int(observation.get("citation_count") or 0) <= 0:
        return True
    if observation.get("guard_warnings"):
        return True
    if observation.get("missing_evidence"):
        return True
    if observation.get("status") == "needs_review" and observation.get("weak_claims"):
        return True
    return any(
        _is_generated_no_evidence_missing(item)
        for item in _as_text_list(observation.get("missing_information"))
    )


def _react_evidence_question(
    plan_step: AgentPlanStep,
    step: AgentStepResult,
    observation: dict[str, Any],
    *,
    tool: str,
) -> str:
    gap_text = "; ".join(
        _dedupe_texts(
            [
                *_as_text_list(observation.get("guard_warnings")),
                *_as_text_list(observation.get("missing_evidence")),
                *_as_text_list(observation.get("missing_information")),
                *[
                    _clean_text(item.get("text"))
                    for item in observation.get("weak_claims", [])
                    if isinstance(item, dict)
                ],
            ]
        )[:5]
    )
    action_label = (
        "Audit the current cited evidence and retrieve direct support"
        if tool == "build_evidence_profile"
        else "Find direct cited document excerpts"
    )
    return (
        f"{action_label} for agent step '{plan_step.title}'. "
        f"Step purpose: {plan_step.purpose}. "
        f"Current summary: {_clean_text(step.summary)[:900]}. "
        f"Observed evidence gap: {gap_text or 'missing direct citation support'}. "
        "Use uploaded documents only. If support is unavailable, state the missing evidence."
    )


def _react_action_plan_step(
    plan_step: AgentPlanStep,
    action: dict[str, Any],
    *,
    iteration: int,
) -> AgentPlanStep:
    tool = _clean_text(action.get("tool"))
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    question = _clean_text(arguments.get("question"))
    step_id = f"{plan_step.step_id}_react_{iteration + 1}"
    title = f"ReAct evidence action for {plan_step.title}"
    purpose = "Resolve evidence gaps observed after the planned step."
    if tool == "build_evidence_profile":
        return AgentPlanStep(
            step_id=step_id,
            title=title,
            purpose=purpose,
            tool="build_evidence_profile",
            arguments={"query": question or plan_step.purpose},
        )
    return AgentPlanStep(
        step_id=step_id,
        title=title,
        purpose=purpose,
        tool="document_qa",
        arguments={"question": question or plan_step.purpose},
    )


def _react_trace_item(
    *,
    iteration: int,
    observation: dict[str, Any],
    action: dict[str, Any],
    action_step: AgentStepResult | None,
) -> dict[str, Any]:
    return {
        "iteration": iteration + 1,
        "observation": observation,
        "action": {
            "tool": action.get("tool"),
            "reason": action.get("reason"),
            "arguments": action.get("arguments", {}),
        },
        "result": _react_action_observation(action_step) if action_step else {},
    }


def _react_action_observation(action_step: AgentStepResult | None) -> dict[str, Any]:
    if action_step is None:
        return {}
    return {
        "step_id": action_step.step_id,
        "tool": action_step.tool,
        "status": action_step.status,
        "citation_count": len(action_step.citations),
        "guard_warnings": action_step.guard_warnings,
        "missing_information": _step_missing_information(action_step),
    }


def _merge_react_action_step(
    step: AgentStepResult,
    action_step: AgentStepResult,
    trace: list[dict[str, Any]],
) -> AgentStepResult:
    """将 ReAct 补证步骤的结果合并回原计划步骤，并重算 guard 与证据画像。"""
    output = dict(step.output)
    metadata = output.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    citations = _dedupe_citations([*step.citations, *action_step.citations])
    source_refs = _format_refs([citation.source_id for citation in action_step.citations[:2]])
    summary = step.summary
    if source_refs and not SOURCE_REF_PATTERN.search(summary):
        summary = f"{summary.rstrip()}{source_refs}"
    if action_step.summary:
        summary = (
            f"{summary.rstrip()}\n\n"
            f"ReAct evidence action ({action_step.title}): {action_step.summary}"
        ).strip()

    missing_information = _dedupe_texts(
        [*_step_missing_information(step), *_step_missing_information(action_step)]
    )
    if action_step.citations:
        missing_information = [
            item for item in missing_information if not _is_generated_no_evidence_missing(item)
        ]

    guard_result = validate_answer(
        summary,
        citations,
        has_retrieved_documents=bool(citations),
    )
    evidence = (
        build_evidence_profile(summary, citations, guard_result.issues)
        if citations
        else step.evidence
    )
    react_metadata = metadata.get("react")
    react_metadata = dict(react_metadata) if isinstance(react_metadata, dict) else {}
    react_metadata.update(
        {
            "enabled": True,
            "policy": "controlled_evidence_v1",
            "allowed_actions": sorted(AGENT_REACT_ACTIONS),
            "action_count": len(trace),
            "added_source_ids": [citation.source_id for citation in action_step.citations],
        }
    )
    metadata["react"] = react_metadata
    output["metadata"] = metadata
    output["missing_information"] = missing_information
    output["react_trace"] = trace
    status = "needs_review" if guard_result.issues or missing_information else "completed"
    return replace(
        step,
        status=status,
        summary=summary,
        citations=citations,
        evidence=evidence if isinstance(evidence, dict) else None,
        guard_warnings=guard_result.issues,
        output=output,
    )


def _mark_react_needs_input(
    step: AgentStepResult,
    action: dict[str, Any],
    trace: list[dict[str, Any]],
) -> AgentStepResult:
    output = dict(step.output)
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    missing_information = _dedupe_texts(
        [
            *_step_missing_information(step),
            *_as_text_list(arguments.get("missing_information")),
            _clean_text(arguments.get("question")),
        ]
    )
    output["missing_information"] = missing_information
    output["react_trace"] = trace
    return replace(step, status="needs_review", output=output)


def _with_react_trace(
    step: AgentStepResult,
    trace: list[dict[str, Any]],
) -> AgentStepResult:
    output = dict(step.output)
    output["react_trace"] = trace
    return replace(step, output=output)
