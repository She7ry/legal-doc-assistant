"""条款审查与冲突检查服务。

将 ``review_clause`` / ``check_conflict`` 从 ``DocumentQAService`` 中独立出来，
便于单独测试与复用。两个入口函数均接收 ``qa_service`` 作为显式参数。
"""

from __future__ import annotations

import logging
from typing import Any

from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.answer_guard import (
    AnswerGuardResult,
    validate_answer,
)
from doc_assistant.services.clause_review import (
    clause_review_metadata,
    empty_clause_review_metadata,
    render_clause_review,
)
from doc_assistant.services.conflict_check import (
    conflict_metadata,
    empty_conflict_metadata,
    render_conflict_check,
)
from doc_assistant.services.review_taxonomy import (
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
    """按条款类型检索文档并输出结构化风险审查（高/中/低 + 理由 + 引用）。"""
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

    context, citations = qa_service._format_context(documents)
    task_prompt = qa_service.clause_review_prompt.format(
        clause_type=clause_type,
        normalized_clause_type=profile.label,
        clause_taxonomy=clause_taxonomy_prompt(),
        risk_rules=profile.risk_rules_prompt(),
        context=context,
    )
    raw_content = qa_service._invoke_chat_messages(qa_service._build_messages(task_prompt))
    metadata = clause_review_metadata(clause_type, profile, raw_content, citations)
    content = (
        render_clause_review(metadata, citations)
        if metadata.get("structured")
        else raw_content
    )
    guard_result = validate_answer(content, citations, has_retrieved_documents=True)
    if guard_result.needs_repair:
        content = qa_service._repair_content(content, guard_result, citations)
        guard_result = validate_answer(content, citations, has_retrieved_documents=True)
    return QAAnswer(
        content=content,
        citations=citations,
        confidence=guard_result.confidence,
        guard_warnings=guard_result.issues,
        metadata={k: v for k, v in metadata.items() if k != "structured"},
    )


def check_conflict(
    qa_service: Any,
    contract_query: str,
    policy_query: str,
    top_k: int | None = None,
) -> QAAnswer:
    """分别检索合同与政策片段，比对义务/定义是否冲突并输出结构化结论。"""
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

    contract_context, contract_citations = qa_service._format_context_prefixed(
        contract_docs, prefix="C"
    )
    policy_context, policy_citations = qa_service._format_context_prefixed(
        policy_docs, prefix="P"
    )
    citations = contract_citations + policy_citations

    task_prompt = qa_service.conflict_check_prompt.format(
        contract_context=contract_context or "No contract excerpts found.",
        policy_context=policy_context or "No policy excerpts found.",
        conflict_types=conflict_types_prompt(),
    )
    raw_content = qa_service._invoke_chat_messages(qa_service._build_messages(task_prompt))
    metadata = conflict_metadata(raw_content, citations)
    content = (
        render_conflict_check(metadata)
        if metadata.get("structured")
        else raw_content
    )
    guard_result = validate_answer(content, citations, has_retrieved_documents=True)
    if guard_result.needs_repair:
        content = qa_service._repair_content(content, guard_result, citations)
        guard_result = validate_answer(content, citations, has_retrieved_documents=True)
    return QAAnswer(
        content=content,
        citations=citations,
        confidence=guard_result.confidence,
        guard_warnings=guard_result.issues,
        metadata={k: v for k, v in metadata.items() if k != "structured"},
    )
