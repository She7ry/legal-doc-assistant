"""条款审阅与冲突检查工具。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from ai.rag.grounding.document_context import format_document_context
from ai.rag.grounding.guard import validate_answer
from ai.rag.schemas import QAAnswer
from ai.review.clause import (
    ClauseReviewOutput,
    clause_review_metadata,
    empty_clause_review_metadata,
    render_clause_review,
)
from ai.review.conflict import (
    ConflictCheckOutput,
    conflict_metadata,
    empty_conflict_metadata,
    render_conflict_check,
)
from ai.review.taxonomy import (
    clause_taxonomy_prompt,
    conflict_types_prompt,
    resolve_clause_profile,
)

logger = logging.getLogger(__name__)


def review_clause(
    qa_service: Any,
    clause_type: str,
    top_k: int | None = None,
) -> QAAnswer:
    """根据检索到的文档审阅指定条款类型。"""
    profile = resolve_clause_profile(clause_type)
    documents = qa_service.vector_store.search(profile.expanded_query(clause_type), k=top_k)
    if not documents:
        metadata = empty_clause_review_metadata(clause_type, profile)
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
    try:
        output = qa_service.chat_model.with_structured_output(ClauseReviewOutput).invoke(
            qa_service._build_messages(task_prompt)
        )
        metadata = clause_review_metadata(clause_type, profile, output, citations)
    except (OutputParserException, ValidationError):
        logger.warning("Clause review structured output was invalid", exc_info=True)
        metadata = empty_clause_review_metadata(clause_type, profile)
        metadata["structured_output_error"] = "invalid_structured_output"
    content = render_clause_review(metadata, citations)
    guard_result = validate_answer(
        content,
        citations,
        has_retrieved_documents=True,
        verify_citation_semantics=True,
    )
    if guard_result.needs_repair:
        content = qa_service.repair_content(content, guard_result, citations)
        guard_result = validate_answer(
            content,
            citations,
            has_retrieved_documents=True,
            verify_citation_semantics=True,
        )
    result_metadata = {k: v for k, v in metadata.items() if k != "structured"}
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
    """对比合同与政策摘录中的潜在冲突。"""
    contract_docs = qa_service.vector_store.search(contract_query, k=top_k)
    policy_docs = qa_service.vector_store.search(policy_query, k=top_k)

    if not contract_docs and not policy_docs:
        metadata = empty_conflict_metadata()
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
        contract_context=contract_context or "未找到合同摘录。",
        policy_context=policy_context or "未找到政策摘录。",
        conflict_types=conflict_types_prompt(),
    )
    try:
        output = qa_service.chat_model.with_structured_output(ConflictCheckOutput).invoke(
            qa_service._build_messages(task_prompt)
        )
        metadata = conflict_metadata(output, citations)
    except (OutputParserException, ValidationError):
        logger.warning("Conflict check structured output was invalid", exc_info=True)
        metadata = empty_conflict_metadata()
        metadata["structured_output_error"] = "invalid_structured_output"
    content = render_conflict_check(metadata)
    guard_result = validate_answer(
        content,
        citations,
        has_retrieved_documents=True,
        verify_citation_semantics=True,
    )
    if guard_result.needs_repair:
        content = qa_service.repair_content(content, guard_result, citations)
        guard_result = validate_answer(
            content,
            citations,
            has_retrieved_documents=True,
            verify_citation_semantics=True,
        )
    result_metadata = {k: v for k, v in metadata.items() if k != "structured"}
    return QAAnswer(
        content=content,
        citations=citations,
        confidence=guard_result.confidence,
        guard_warnings=guard_result.issues,
        metadata=result_metadata,
    )

__all__ = ["check_conflict", "review_clause"]
