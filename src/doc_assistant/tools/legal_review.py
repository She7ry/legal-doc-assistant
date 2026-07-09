"""Legal review tools for clause review and conflict checking."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from doc_assistant.grounding.document_context import format_document_context
from doc_assistant.grounding.guard import (
    AnswerGuardResult,
    validate_answer,
)
from doc_assistant.review.clause import (
    ClauseReviewOutput,
    clause_review_metadata,
    empty_clause_review_metadata,
    render_clause_review,
)
from doc_assistant.review.conflict import (
    ConflictCheckOutput,
    conflict_metadata,
    empty_conflict_metadata,
    render_conflict_check,
)
from doc_assistant.review.taxonomy import (
    clause_taxonomy_prompt,
    conflict_types_prompt,
    resolve_clause_profile,
)
from doc_assistant.schemas.citation import QAAnswer

logger = logging.getLogger(__name__)


def review_clause(
    qa_service: Any,
    clause_type: str,
    top_k: int | None = None,
) -> QAAnswer:
    """Review a clause type against retrieved documents."""
    profile = resolve_clause_profile(clause_type)
    documents = qa_service.vector_store.search(profile.expanded_query(clause_type), k=top_k)
    skill_context, evidence_assessment, skill_guidance = qa_service.prepare_skill_guidance(
        f"Review the {clause_type} clause using retrieved document evidence.",
        documents,
    )
    if not documents:
        metadata = empty_clause_review_metadata(clause_type, profile)
        metadata.update(_skill_metadata(skill_context, evidence_assessment))
        return QAAnswer(
            content=render_clause_review(metadata, []),
            citations=[],
            confidence="Low",
            metadata=metadata,
        )

    context, citations = format_document_context(documents)
    task_prompt = qa_service.clause_review_prompt.format(
        clause_type=clause_type,
        normalized_clause_type=profile.label,
        clause_taxonomy=clause_taxonomy_prompt(),
        risk_rules=profile.risk_rules_prompt(),
        context=context,
    )
    if skill_guidance:
        task_prompt = f"{task_prompt}\n\n{skill_guidance}"
    try:
        output = qa_service.chat_model.with_structured_output(ClauseReviewOutput).invoke(
            qa_service._build_messages(task_prompt)
        )
        metadata = clause_review_metadata(clause_type, profile, output, citations)
    except (OutputParserException, ValidationError) as exc:
        logger.warning("Clause review structured output was invalid: %s", exc)
        metadata = empty_clause_review_metadata(clause_type, profile)
        metadata["structured_output_error"] = str(exc)
    content = render_clause_review(metadata, citations)
    verify_support = "verify-citation-support" in skill_context.selected_skills
    guard_result = validate_answer(
        content,
        citations,
        has_retrieved_documents=True,
        verify_citation_semantics=verify_support,
    )
    if guard_result.needs_repair:
        content = qa_service.repair_content(content, guard_result, citations)
        guard_result = validate_answer(
            content,
            citations,
            has_retrieved_documents=True,
            verify_citation_semantics=verify_support,
        )
    result_metadata = {k: v for k, v in metadata.items() if k != "structured"}
    result_metadata.update(_skill_metadata(skill_context, evidence_assessment, guard_result))
    return QAAnswer(
        content=content,
        citations=citations,
        confidence=guard_result.confidence,
        guard_warnings=guard_result.issues,
        metadata=result_metadata,
    )


def check_conflict(
    qa_service: Any,
    contract_query: str,
    policy_query: str,
    top_k: int | None = None,
) -> QAAnswer:
    """Compare contract and policy excerpts for conflicts."""
    contract_docs = qa_service.vector_store.search(contract_query, k=top_k)
    policy_docs = qa_service.vector_store.search(policy_query, k=top_k)
    skill_context, evidence_assessment, skill_guidance = qa_service.prepare_skill_guidance(
        f"Compare contract evidence for {contract_query} with policy evidence for {policy_query}.",
        [*contract_docs, *policy_docs],
    )

    if not contract_docs and not policy_docs:
        metadata = empty_conflict_metadata()
        metadata.update(_skill_metadata(skill_context, evidence_assessment))
        return QAAnswer(
            content=render_conflict_check(metadata),
            citations=[],
            confidence="Low",
            metadata=metadata,
        )

    contract_context, contract_citations = format_document_context(contract_docs, prefix="C")
    policy_context, policy_citations = format_document_context(policy_docs, prefix="P")
    citations = contract_citations + policy_citations

    task_prompt = qa_service.conflict_check_prompt.format(
        contract_context=contract_context or "No contract excerpts found.",
        policy_context=policy_context or "No policy excerpts found.",
        conflict_types=conflict_types_prompt(),
    )
    if skill_guidance:
        task_prompt = f"{task_prompt}\n\n{skill_guidance}"
    try:
        output = qa_service.chat_model.with_structured_output(ConflictCheckOutput).invoke(
            qa_service._build_messages(task_prompt)
        )
        metadata = conflict_metadata(output, citations)
    except (OutputParserException, ValidationError) as exc:
        logger.warning("Conflict check structured output was invalid: %s", exc)
        metadata = empty_conflict_metadata()
        metadata["structured_output_error"] = str(exc)
    content = render_conflict_check(metadata)
    verify_support = "verify-citation-support" in skill_context.selected_skills
    guard_result = validate_answer(
        content,
        citations,
        has_retrieved_documents=True,
        verify_citation_semantics=verify_support,
    )
    if guard_result.needs_repair:
        content = qa_service.repair_content(content, guard_result, citations)
        guard_result = validate_answer(
            content,
            citations,
            has_retrieved_documents=True,
            verify_citation_semantics=verify_support,
        )
    result_metadata = {k: v for k, v in metadata.items() if k != "structured"}
    result_metadata.update(_skill_metadata(skill_context, evidence_assessment, guard_result))
    return QAAnswer(
        content=content,
        citations=citations,
        confidence=guard_result.confidence,
        guard_warnings=guard_result.issues,
        metadata=result_metadata,
    )


def _skill_metadata(
    skill_context: Any,
    evidence_assessment: Any,
    guard_result: AnswerGuardResult | None = None,
) -> dict[str, Any]:
    metadata = skill_context.metadata_payload()
    if evidence_assessment is not None:
        metadata["evidence_sufficiency"] = evidence_assessment.as_dict()
    if guard_result and guard_result.citation_support is not None:
        metadata["citation_support"] = [
            {
                "claim": check.claim,
                "citation_ids": list(check.citation_ids),
                "status": check.status,
                "reason": check.reason,
            }
            for check in guard_result.citation_support.checks
        ]
    return metadata


__all__ = ["check_conflict", "review_clause"]
