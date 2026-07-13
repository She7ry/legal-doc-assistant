"""ReAct Agent task adapter with bounded decomposition for complex work."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from typing import Any
from uuid import uuid4

from doc_assistant.agent._helpers import _clean_text, _remap_metadata, _remap_source_refs
from doc_assistant.agent.schemas import (
    AgentPlanOutput,
    AgentStepResult,
    AgentTaskResult,
    MatterExtractionOutput,
    MatterProfileOutput,
)
from doc_assistant.grounding.evidence import build_evidence_profile
from doc_assistant.grounding.guard import validate_answer
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.services.tool_calling_service import ToolCallingAnswer, ToolCallingChatService
from doc_assistant.skills.runtime import is_complex_retrieval_query

logger = logging.getLogger(__name__)


def run_react_agent_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    focus_areas: list[str] | None = None,
    user_role: str = "ordinary",
    max_steps: int = 6,
    user_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    matter_id: str | None = None,
    progress_callback=None,
) -> AgentTaskResult:
    resolved_task_id = task_id or uuid4().hex
    planned_steps, planning_mode = _plan_task(
        qa_service,
        objective=objective,
        focus_areas=focus_areas or [],
        user_role=user_role,
    )
    if len(planned_steps) >= 2:
        return _run_planned_react_task(
            qa_service,
            objective=objective,
            planned_steps=planned_steps,
            planning_mode=planning_mode,
            user_role=user_role,
            max_steps=max_steps,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=resolved_task_id,
            matter_id=matter_id,
            progress_callback=progress_callback,
        )

    return _run_single_react_task(
        qa_service,
        objective=objective,
        focus_areas=focus_areas or [],
        user_role=user_role,
        max_steps=max_steps,
        user_id=user_id,
        conversation_id=conversation_id,
        task_id=resolved_task_id,
        matter_id=matter_id,
        progress_callback=progress_callback,
    )


def _run_single_react_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    focus_areas: list[str],
    user_role: str,
    max_steps: int,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    matter_id: str | None,
    progress_callback,
) -> AgentTaskResult:
    if progress_callback:
        progress_callback(
            event_type="react_started",
            stage="answering",
            progress=10,
            message="Running ReAct tool-calling workflow.",
        )

    answer = ToolCallingChatService(qa_service).ask(
        _react_question(objective, focus_areas, user_role),
        user_id=user_id,
        conversation_id=conversation_id,
        task_id=task_id,
        enable_web_search=False,
        max_tool_iterations=max_steps,
    )

    if progress_callback:
        progress_callback(
            event_type="react_completed",
            stage="reporting",
            progress=90,
            message="ReAct workflow completed.",
            payload={"tool_calls": [trace.name for trace in answer.tool_calls]},
        )

    matter_profile, findings, artifacts = _extract_matter(
        qa_service,
        objective=objective,
        report=answer.content,
        citations=answer.citations,
        matter_id=matter_id,
    )
    human_review_required = bool(answer.guard_warnings)
    status = "needs_human_review" if human_review_required else "completed"
    memory_service = getattr(qa_service, "memory_service", None)
    if status == "completed" and user_id and memory_service:
        memory_service.mark_task_memories_stale(
            qa_service.tenant_id,
            user_id,
            task_id,
        )

    step = _answer_step(answer)
    return AgentTaskResult(
        task_id=task_id,
        status=status,
        objective=objective,
        steps=[step],
        findings=findings,
        human_review_required=human_review_required,
        report=answer.content,
        citations=answer.citations,
        confidence=answer.confidence,
        guard_warnings=answer.guard_warnings,
        evidence=answer.metadata.get("evidence") if isinstance(answer.metadata, dict) else None,
        matter_profile=matter_profile,
        artifacts=artifacts,
        metadata={
            "user_role": user_role,
            "runtime": "react_tool_calling_v1",
            "tenant_id": qa_service.tenant_id,
            "matter_id": matter_id,
            "available_tools": ["check_conflict", "review_clause", "search_documents"],
            "tool_calls": [trace.name for trace in answer.tool_calls],
            "max_tool_iterations": max_steps,
        },
    )


def _plan_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    focus_areas: list[str],
    user_role: str,
) -> tuple[list[tuple[str, str]], str]:
    normalized_focus = list(
        dict.fromkeys(area for area in (_clean_text(value) for value in focus_areas) if area)
    )
    if len(normalized_focus) >= 2:
        return [
            (
                area[:120],
                f"Analyze only the {area} focus area and produce a self-contained legal finding.",
            )
            for area in normalized_focus
        ], "focus_areas"

    if not is_complex_retrieval_query(objective):
        return [], "single"

    try:
        output = AgentPlanOutput.model_validate(
            qa_service.chat_model.with_structured_output(AgentPlanOutput).invoke(
                qa_service._build_messages(_planner_prompt(objective, normalized_focus, user_role))
            )
        )
    except Exception:
        logger.warning("Agent planning failed; falling back to single-step ReAct.", exc_info=True)
        return [], "single"

    steps: list[tuple[str, str]] = []
    seen: set[str] = set()
    for step in output.steps:
        title = _clean_text(step.title)
        instruction = _clean_text(step.instruction)
        key = instruction.casefold()
        if not title or not instruction or key in seen:
            continue
        seen.add(key)
        steps.append((title, instruction))
    if len(steps) < 2:
        logger.warning("Agent plan was invalid; falling back to single-step ReAct.")
        return [], "single"
    return steps[:5], "planner"


def _run_planned_react_task(
    qa_service: DocumentQAService,
    *,
    objective: str,
    planned_steps: list[tuple[str, str]],
    planning_mode: str,
    user_role: str,
    max_steps: int,
    user_id: str | None,
    conversation_id: str | None,
    task_id: str,
    matter_id: str | None,
    progress_callback,
) -> AgentTaskResult:
    step_results: list[AgentStepResult] = []
    citations: list[Citation] = []
    citations_by_key: dict[tuple[Any, ...], Citation] = {}

    if progress_callback:
        progress_callback(
            event_type="react_started",
            stage="answering",
            progress=10,
            message="Running bounded multi-step ReAct workflow.",
            payload={"step_count": len(planned_steps), "planning_mode": planning_mode},
        )

    for index, (title, instruction) in enumerate(planned_steps, start=1):
        step_id = f"step-{index}"
        start_progress = 10 + int((index - 1) * 70 / len(planned_steps))
        end_progress = 10 + int(index * 70 / len(planned_steps))
        if progress_callback:
            progress_callback(
                event_type="step_started",
                stage="answering",
                progress=start_progress,
                message=f"Started {title}.",
                step_id=step_id,
                payload={"title": title},
            )

        answer = None
        for attempt in range(1, 3):
            try:
                answer = ToolCallingChatService(qa_service).ask(
                    _step_question(objective, title, instruction, user_role),
                    user_id=user_id,
                    conversation_id=f"{conversation_id or task_id}:{step_id}",
                    task_id=f"{task_id}:{step_id}",
                    enable_web_search=False,
                    max_tool_iterations=max_steps,
                )
                break
            except Exception:
                logger.warning(
                    "Agent step execution failed%s.",
                    "; retrying once" if attempt == 1 else " after one retry",
                    extra={"task_id": task_id, "step_id": step_id, "attempt": attempt},
                    exc_info=True,
                )

        if answer is None:
            step = AgentStepResult(
                step_id=step_id,
                title=title,
                tool="tool_calling_react",
                status="failed",
                summary="Step failed after one retry; human review is required.",
                guard_warnings=["Step execution failed after one retry."],
                output={"attempts": 2, "error": "step_execution_failed"},
            )
            step_results.append(step)
            if progress_callback:
                progress_callback(
                    event_type="step_failed",
                    stage="answering",
                    progress=end_progress,
                    message=f"{title} failed after one retry.",
                    step_id=step_id,
                    payload={"status": "failed", "attempts": 2},
                )
            continue

        citation_mapping, step_citations = _renumber_citations(
            answer.citations,
            citations_by_key,
            citations,
        )
        step = _answer_step(
            answer,
            step_id=step_id,
            title=title,
            citation_mapping=citation_mapping,
            citations=step_citations,
        )
        step_results.append(step)
        if progress_callback:
            progress_callback(
                event_type="step_completed",
                stage="answering",
                progress=end_progress,
                message=f"Completed {title}.",
                step_id=step_id,
                payload={"status": step.status, "citation_count": len(step.citations)},
            )

    report, synthesis_warnings = _synthesize_steps(
        qa_service,
        objective=objective,
        user_role=user_role,
        steps=step_results,
        citations=citations,
    )
    guard_result = validate_answer(
        report,
        citations,
        has_retrieved_documents=bool(citations),
    )
    if guard_result.needs_repair:
        try:
            report = qa_service.repair_content(report, guard_result, citations)
            guard_result = validate_answer(
                report,
                citations,
                has_retrieved_documents=bool(citations),
            )
        except Exception:
            logger.warning("Final Agent synthesis repair failed.", exc_info=True)
            synthesis_warnings.append("Final synthesis repair failed.")

    if progress_callback:
        progress_callback(
            event_type="react_completed",
            stage="reporting",
            progress=90,
            message="Multi-step ReAct workflow completed.",
            payload={"step_count": len(step_results)},
        )

    guard_warnings = list(
        dict.fromkeys(
            warning
            for warning in (
                *synthesis_warnings,
                *(warning for step in step_results for warning in step.guard_warnings),
                *guard_result.issues,
            )
            if warning
        )
    )
    human_review_required = bool(guard_warnings) or any(
        step.status != "completed" for step in step_results
    )
    status = "needs_human_review" if human_review_required else "completed"
    matter_profile, findings, artifacts = _extract_matter(
        qa_service,
        objective=objective,
        report=report,
        citations=citations,
        matter_id=matter_id,
    )
    memory_service = getattr(qa_service, "memory_service", None)
    if status == "completed" and user_id and memory_service:
        memory_service.mark_task_memories_stale(qa_service.tenant_id, user_id, task_id)

    return AgentTaskResult(
        task_id=task_id,
        status=status,
        objective=objective,
        steps=step_results,
        findings=findings,
        human_review_required=human_review_required,
        report=report,
        citations=citations,
        confidence=guard_result.confidence,
        guard_warnings=guard_warnings,
        evidence=build_evidence_profile(report, citations, guard_result.issues),
        matter_profile=matter_profile,
        artifacts=artifacts,
        metadata={
            "user_role": user_role,
            "runtime": "react_tool_calling_bounded_v1",
            "planning_mode": planning_mode,
            "planned_step_count": len(planned_steps),
            "tenant_id": qa_service.tenant_id,
            "matter_id": matter_id,
            "available_tools": ["check_conflict", "review_clause", "search_documents"],
            "tool_calls": [
                trace["name"]
                for step in step_results
                for trace in step.output.get("tool_calls", [])
                if isinstance(trace, dict) and trace.get("name")
            ],
            "max_tool_iterations": max_steps,
        },
    )


def _renumber_citations(
    source_citations: list[Citation],
    citations_by_key: dict[tuple[Any, ...], Citation],
    citations: list[Citation],
) -> tuple[dict[str, str], list[Citation]]:
    mapping: dict[str, str] = {}
    step_citations: list[Citation] = []
    for citation in source_citations:
        key = _citation_key(citation)
        global_citation = citations_by_key.get(key)
        if global_citation is None:
            global_citation = replace(citation, source_id=f"D{len(citations) + 1}")
            citations_by_key[key] = global_citation
            citations.append(global_citation)
        mapping[citation.source_id.upper()] = global_citation.source_id
        if global_citation not in step_citations:
            step_citations.append(global_citation)
    return mapping, step_citations


def _citation_key(citation: Citation) -> tuple[Any, ...]:
    return (
        citation.source_type,
        citation.document_key or citation.file_id or citation.file_name,
        citation.document_version,
        citation.page,
        citation.chunk_id,
        citation.char_start,
        citation.char_end,
        citation.exact_quote or citation.preview,
    )


def _synthesize_steps(
    qa_service: DocumentQAService,
    *,
    objective: str,
    user_role: str,
    steps: list[AgentStepResult],
    citations: list[Citation],
) -> tuple[str, list[str]]:
    prompt = _synthesis_prompt(objective, user_role, steps, citations)
    for attempt in range(1, 3):
        try:
            return qa_service._invoke_chat_messages(qa_service._build_messages(prompt)), []
        except Exception:
            logger.warning(
                "Final Agent synthesis failed%s.",
                "; retrying once" if attempt == 1 else " after one retry",
                extra={"attempt": attempt},
                exc_info=True,
            )
    return _fallback_report(steps), ["Final synthesis failed after one retry."]


def _planner_prompt(objective: str, focus_areas: list[str], user_role: str) -> str:
    return (
        "Split this compound legal task into 2-5 flat, independent steps. "
        "Each step must be executable on its own with document-search and legal-review tools. "
        "Do not create dependencies, recursive substeps, or a final synthesis step. "
        "Return only title and instruction for each step."
        f"\n\nObjective:\n{objective.strip()}"
        f"\n\nExisting focus context:\n{', '.join(focus_areas) or 'None.'}"
        f"\n\nAudience role:\n{user_role}"
    )


def _step_question(objective: str, title: str, instruction: str, user_role: str) -> str:
    return (
        f"Task objective (background only):\n{objective.strip()}"
        f"\n\nIndependent step: {title}\n{instruction}"
        "\n\nWork only on this step. Do not attempt other planned steps or synthesize them. "
        "Use the available tools and cite retrieved evidence."
        f"\n\nAudience role: {user_role}"
    )


def _synthesis_prompt(
    objective: str,
    user_role: str,
    steps: list[AgentStepResult],
    citations: list[Citation],
) -> str:
    step_blocks = []
    for step in steps:
        guard_status = "passed" if not step.guard_warnings else "; ".join(step.guard_warnings[:5])
        source_ids = ", ".join(citation.source_id for citation in step.citations) or "None"
        step_blocks.append(
            f"[{step.step_id}] {step.title}\n"
            f"Status: {step.status}\nGuard: {guard_status}\nCitations: {source_ids}\n"
            f"Summary:\n{step.summary[:2000]}"
        )
    citation_lines = [
        (
            f"- {citation.source_id} | {citation.file_name}{citation.location_label()} | "
            f"{(citation.exact_quote or citation.preview).strip()[:800]}"
        )
        for citation in citations
    ]
    step_context = "\n\n".join(step_blocks)
    citation_context = "\n".join(citation_lines) or "None."
    return (
        "Produce one concise final legal work-product from the independent step summaries below. "
        "Use no facts beyond those summaries and the supplied citations. Preserve citation IDs exactly, "
        "cite every material paragraph, expose failed or needs-review steps, and do not mention tool traces."
        f"\n\nObjective:\n{objective.strip()}"
        f"\n\nAudience role:\n{user_role}"
        f"\n\nStep summaries:\n{step_context}"
        f"\n\nCitations:\n{citation_context}"
    )


def _fallback_report(steps: list[AgentStepResult]) -> str:
    lines = ["Final synthesis was unavailable; the independent step results follow."]
    for step in steps:
        lines.extend(("", f"## {step.title} ({step.status})", step.summary))
    return "\n".join(lines)


def _extract_matter(
    qa_service: DocumentQAService,
    *,
    objective: str,
    report: str,
    citations: list[Citation],
    matter_id: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        output = MatterExtractionOutput.model_validate(
            qa_service.chat_model.with_structured_output(MatterExtractionOutput).invoke(
                qa_service._build_messages(_matter_extraction_prompt(objective, report, citations))
            )
        )
    except Exception:
        logger.warning(
            "Matter structured output failed; returning no Matter",
            exc_info=True,
        )
        return None, [], []

    profile = _profile_payload(output.matter_profile, matter_id)
    citations_by_id = {
        citation.source_id.strip().casefold(): citation
        for citation in citations
        if citation.source_id.strip()
    }
    findings = _finding_payloads(output, citations_by_id)
    artifacts = _artifact_payloads(output, citations_by_id)
    if not profile and not findings and not artifacts:
        logger.warning("Matter structured output was empty; returning no Matter.")
        return None, [], []
    return profile, findings, artifacts


def _profile_payload(
    profile: MatterProfileOutput | None,
    matter_id: str | None,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    payload = {key: value for key, value in profile.model_dump().items() if _has_content(value)}
    if not payload:
        return None
    if matter_id:
        payload["matter_id"] = matter_id
    return payload


def _finding_payloads(
    output: MatterExtractionOutput,
    citations_by_id: dict[str, Citation],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for finding in output.findings:
        finding_id = finding.finding_id.strip()
        summary = finding.summary.strip()
        if not finding_id or not summary:
            continue
        known_citations = _known_citations(finding.citations, citations_by_id)
        citation = known_citations[0] if known_citations else None
        source_quote = ((citation.exact_quote or citation.preview) if citation else "").strip()
        location_label = citation.location_label() if citation else ""
        evidence_coverage = (
            "direct" if source_quote and location_label else "partial" if citation else "missing"
        )
        needs_human_review = finding.needs_human_review or citation is None
        payload = finding.model_dump()
        payload.update(
            finding_id=finding_id,
            summary=summary,
            citations=[item.source_id for item in known_citations],
            source_step_id="react",
            evidence_coverage=evidence_coverage,
            support_level="direct" if source_quote else "missing",
            unsupported_reason="" if source_quote else "Missing source citation.",
            source_quote=source_quote[:1200],
            location_label=location_label,
            needs_human_review=needs_human_review,
            human_review_status="pending" if needs_human_review else "not_required",
        )
        payloads.append(payload)
    return payloads


def _artifact_payloads(
    output: MatterExtractionOutput,
    citations_by_id: dict[str, Citation],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    finding_ids = {finding.finding_id.strip() for finding in output.findings}
    for artifact in output.artifacts:
        artifact_id = artifact.artifact_id.strip()
        artifact_type = artifact.artifact_type.strip()
        if not artifact_id or not artifact_type:
            continue
        payload = artifact.model_dump()
        payload.update(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=artifact.title.strip() or artifact_id,
            source_finding_ids=[
                finding_id
                for finding_id in dict.fromkeys(
                    source_id.strip() for source_id in artifact.source_finding_ids
                )
                if finding_id in finding_ids
            ],
            citations=[
                citation.source_id
                for citation in _known_citations(artifact.citations, citations_by_id)
            ],
        )
        payloads.append(payload)
    return payloads


def _known_citations(
    source_ids: list[str],
    citations_by_id: dict[str, Citation],
) -> list[Citation]:
    found: list[Citation] = []
    for source_id in source_ids:
        citation = citations_by_id.get(source_id.strip().casefold())
        if citation and citation not in found:
            found.append(citation)
    return found


def _has_content(value: Any) -> bool:
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def _matter_extraction_prompt(
    objective: str,
    report: str,
    citations: list[Citation],
) -> str:
    citation_lines = [
        (
            f"- {citation.source_id} | {citation.file_name}{citation.location_label()} | "
            f"{(citation.exact_quote or citation.preview).strip()}"
        )
        for citation in citations
    ]
    citation_context = "\n".join(citation_lines) or "None."
    return (
        "Extract a legal matter profile, findings, and useful artifacts from the final report. "
        "Use only the objective, report, and existing citations below. Do not invent facts or "
        "citation IDs. Return an empty extraction when the report contains no useful Matter data."
        f"\n\nObjective:\n{objective.strip()}"
        f"\n\nFinal report:\n{report.strip()}"
        f"\n\nExisting citations:\n{citation_context}"
    )


def _react_question(objective: str, focus_areas: list[str], user_role: str) -> str:
    parts = [objective.strip()]
    if focus_areas:
        parts.append("Focus areas: " + ", ".join(focus_areas))
    if user_role:
        parts.append(f"Audience role: {user_role}")
    return "\n\n".join(part for part in parts if part)


def _answer_step(
    answer: ToolCallingAnswer,
    *,
    step_id: str = "react",
    title: str = "ReAct answer",
    citation_mapping: dict[str, str] | None = None,
    citations: list[Citation] | None = None,
) -> AgentStepResult:
    mapping = citation_mapping or {}
    evidence = answer.metadata.get("evidence") if isinstance(answer.metadata, dict) else None
    output = _remap_metadata(
        {
            "tool_calls": [asdict(trace) for trace in answer.tool_calls],
            "web_sources": [asdict(source) for source in answer.web_sources],
        },
        mapping,
    )
    return AgentStepResult(
        step_id=step_id,
        title=title,
        tool="tool_calling_react",
        status="needs_review" if answer.guard_warnings else "completed",
        summary=_remap_source_refs(answer.content, mapping),
        citations=answer.citations if citations is None else citations,
        evidence=_remap_metadata(evidence, mapping) if isinstance(evidence, dict) else None,
        guard_warnings=[_remap_source_refs(warning, mapping) for warning in answer.guard_warnings],
        output=output,
    )
