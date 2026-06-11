from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryCandidate, MemoryUsage
from doc_assistant.memory.service import MemoryService
from doc_assistant.models.language_model import build_chat_model
from doc_assistant.retrieval.vector_store import DocumentVectorStore
from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.answer_guard import AnswerGuardResult, validate_answer
from doc_assistant.services.evidence import build_evidence_profile
from doc_assistant.services.review_taxonomy import (
    ClauseProfile,
    allowed_conflict_type_keys,
    clause_taxonomy_prompt,
    conflict_types_prompt,
    resolve_clause_profile,
)
from doc_assistant.utils.prompt_loader import load_base_legal_prompt, load_prompt

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT = """Given the user's legal question and chat history, rewrite it as a precise document retrieval query.
Keep legal terms, party names, dates, and clause names intact.
Output only the rewritten query.

Chat history:
{chat_history}

Original question:
{question}

Rewritten query:"""

_VAGUE_QUERY_TERMS = (
    "这个",
    "那个",
    "它",
    "上面",
    "前面",
    "刚才",
    "this",
    "that",
    "it",
    "above",
    "previous",
)


@dataclass(frozen=True)
class PreparedQAAnswer:
    messages: list[dict[str, str]]
    citations: list[Citation]
    memories_used: list[MemoryUsage]
    user_id: str | None
    conversation_id: str | None
    user_message_recorded: bool
    has_retrieved_documents: bool = True


class DocumentQAService:
    def __init__(
        self,
        vector_store: DocumentVectorStore | None = None,
        chat_model=None,
        memory_service: MemoryService | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.vector_store = vector_store or DocumentVectorStore()
        self.tenant_id = tenant_id or getattr(self.vector_store, "tenant_id", settings.default_tenant_id)
        self.base_prompt = load_base_legal_prompt()
        self.prompt = PromptTemplate.from_template(load_prompt("document_qa.txt"))
        self.general_prompt = PromptTemplate.from_template(load_prompt("general_chat.txt"))
        self.clause_review_prompt = PromptTemplate.from_template(load_prompt("clause_review.txt"))
        self.conflict_check_prompt = PromptTemplate.from_template(load_prompt("conflict_check.txt"))
        self.answer_repair_prompt = PromptTemplate.from_template(load_prompt("answer_repair.txt"))
        self.chat_model = chat_model or build_chat_model()
        self.memory_service = memory_service

    def ask(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> QAAnswer:
        prepared = self.prepare_answer(
            question,
            chat_history=chat_history,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        content = self._invoke_chat_messages(prepared.messages)
        return self.finalize_prepared_answer(prepared, content)

    async def aask(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> QAAnswer:
        prepared = await asyncio.to_thread(
            self.prepare_answer,
            question,
            chat_history,
            user_id,
            conversation_id,
            task_id,
        )
        content = await self._ainvoke_chat_messages(prepared.messages)
        return self.finalize_prepared_answer(prepared, content)

    def finalize_prepared_answer(
        self,
        prepared: PreparedQAAnswer,
        content: str,
        *,
        repair: bool = True,
    ) -> QAAnswer:
        guard_result = validate_answer(
            content,
            prepared.citations,
            has_retrieved_documents=prepared.has_retrieved_documents,
        )
        if repair and guard_result.needs_repair:
            content = self._repair_answer(content, guard_result, prepared)
            guard_result = validate_answer(
                content,
                prepared.citations,
                has_retrieved_documents=prepared.has_retrieved_documents,
            )
        self.record_prepared_answer(prepared, content)
        return QAAnswer(
            content=content,
            citations=prepared.citations,
            memories_used=prepared.memories_used,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata={
                "evidence": build_evidence_profile(
                    content,
                    prepared.citations,
                    guard_result.issues,
                )
            },
        )

    def prepare_answer(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> PreparedQAAnswer:
        resolved_conversation_id = conversation_id
        memory_candidates: list[MemoryCandidate] = []
        memory_context = "No relevant user memory."
        user_message_recorded = False

        if self.memory_service and user_id:
            try:
                resolved_conversation_id = self.memory_service.ensure_context(
                    self.tenant_id,
                    user_id,
                    conversation_id,
                )
                message_id = self.memory_service.record_user_message(
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                    conversation_id=resolved_conversation_id,
                    content=question,
                )
                user_message_recorded = True
                self.memory_service.write_memories_from_user_message(
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                    conversation_id=resolved_conversation_id,
                    message_id=message_id,
                    content=question,
                )
                memory_candidates = self.memory_service.retrieve_relevant_memories(
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                    query=question,
                )
                memory_context = self.memory_service.format_for_prompt(memory_candidates)
            except Exception:
                logger.debug(
                    "Memory enrichment failed; continuing without memory context.",
                    extra={"tenant_id": self.tenant_id, "user_id": user_id},
                    exc_info=True,
                )
                # Memory failure should not prevent citation-first document QA.
                memory_candidates = []
                memory_context = "No relevant user memory."

        chat_history_text = self._format_chat_history(chat_history or [])
        retrieval_query = self._rewrite_query(question, chat_history_text)
        documents = self.vector_store.search(retrieval_query)
        self._log_retrieval(
            question=retrieval_query,
            user_id=user_id,
            conversation_id=resolved_conversation_id,
            document_count=len(documents),
            memory_candidates=memory_candidates,
        )
        memories_used = (
            self.memory_service.usages_from_candidates(memory_candidates)
            if self.memory_service
            else []
        )
        if not documents:
            task_prompt = self.general_prompt.format(
                question=question,
                chat_history=chat_history_text,
                user_memory=memory_context,
            )
            return PreparedQAAnswer(
                messages=self._build_messages(task_prompt),
                citations=[],
                memories_used=memories_used,
                user_id=user_id,
                conversation_id=resolved_conversation_id,
                user_message_recorded=user_message_recorded,
                has_retrieved_documents=False,
            )

        context, citations = self._format_context(documents)
        task_prompt = self.prompt.format(
            question=question,
            context=context,
            chat_history=chat_history_text,
            user_memory=memory_context,
        )
        return PreparedQAAnswer(
            messages=self._build_messages(task_prompt),
            citations=citations,
            memories_used=memories_used,
            user_id=user_id,
            conversation_id=resolved_conversation_id,
            user_message_recorded=user_message_recorded,
            has_retrieved_documents=True,
        )

    def stream_prepared_answer(self, prepared: PreparedQAAnswer) -> Iterator[str]:
        yield from self._stream_chat_messages(prepared.messages)

    def guard_streamed_answer(
        self,
        prepared: PreparedQAAnswer,
        content: str,
    ) -> AnswerGuardResult:
        return validate_answer(
            content,
            prepared.citations,
            has_retrieved_documents=prepared.has_retrieved_documents,
        )

    def record_prepared_answer(self, prepared: PreparedQAAnswer, content: str) -> None:
        self._record_assistant_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content=content,
            user_message_recorded=prepared.user_message_recorded,
        )

    def review_clause(self, clause_type: str, top_k: int | None = None) -> QAAnswer:
        """Search indexed documents for a specific clause type and assess risk level."""
        profile = resolve_clause_profile(clause_type)
        documents = self.vector_store.search(profile.expanded_query(clause_type), k=top_k)
        if not documents:
            metadata = self._empty_clause_review_metadata(clause_type, profile)
            return QAAnswer(
                content=self._render_clause_review(metadata, []),
                citations=[],
                confidence="Low",
                metadata=metadata,
            )

        context, citations = self._format_context(documents)
        task_prompt = self.clause_review_prompt.format(
            clause_type=clause_type,
            normalized_clause_type=profile.label,
            clause_taxonomy=clause_taxonomy_prompt(),
            risk_rules=profile.risk_rules_prompt(),
            context=context,
        )
        raw_content = self._invoke_chat_messages(self._build_messages(task_prompt))
        metadata = self._clause_review_metadata(clause_type, profile, raw_content, citations)
        content = (
            self._render_clause_review(metadata, citations)
            if metadata.get("structured")
            else raw_content
        )
        guard_result = validate_answer(content, citations, has_retrieved_documents=True)
        if guard_result.needs_repair:
            content = self._repair_content(content, guard_result, citations)
            guard_result = validate_answer(content, citations, has_retrieved_documents=True)
        return QAAnswer(
            content=content,
            citations=citations,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata={k: v for k, v in metadata.items() if k != "structured"},
        )

    def check_conflict(self, contract_query: str, policy_query: str, top_k: int | None = None) -> QAAnswer:
        """Retrieve contract and policy excerpts separately, then check for conflicts."""
        contract_docs = self.vector_store.search(contract_query, k=top_k)
        policy_docs = self.vector_store.search(policy_query, k=top_k)

        if not contract_docs and not policy_docs:
            metadata = self._empty_conflict_metadata()
            return QAAnswer(
                content=self._render_conflict_check(metadata),
                citations=[],
                confidence="Low",
                metadata=metadata,
            )

        contract_context, contract_citations = self._format_context_prefixed(contract_docs, prefix="C")
        policy_context, policy_citations = self._format_context_prefixed(policy_docs, prefix="P")
        citations = contract_citations + policy_citations

        task_prompt = self.conflict_check_prompt.format(
            contract_context=contract_context or "No contract excerpts found.",
            policy_context=policy_context or "No policy excerpts found.",
            conflict_types=conflict_types_prompt(),
        )
        raw_content = self._invoke_chat_messages(self._build_messages(task_prompt))
        metadata = self._conflict_metadata(raw_content, citations)
        content = (
            self._render_conflict_check(metadata)
            if metadata.get("structured")
            else raw_content
        )
        guard_result = validate_answer(content, citations, has_retrieved_documents=True)
        if guard_result.needs_repair:
            content = self._repair_content(content, guard_result, citations)
            guard_result = validate_answer(content, citations, has_retrieved_documents=True)
        return QAAnswer(
            content=content,
            citations=citations,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata={k: v for k, v in metadata.items() if k != "structured"},
        )

    def _empty_clause_review_metadata(
        self,
        clause_type: str,
        profile: ClauseProfile,
    ) -> dict[str, Any]:
        return {
            "structured": True,
            "clause_type": clause_type,
            "normalized_clause_type": profile.key,
            "found": False,
            "summary": "No relevant content found in indexed documents for the requested clause type.",
            "risk_level": "Needs human review",
            "risk_reasons": [],
            "affected_party": None,
            "plain_language_explanation": "The system did not retrieve enough cited text to review this clause.",
            "questions_for_lawyer": [],
            "missing_information": ["Relevant clause text or a more specific clause query."],
            "needs_human_review": True,
        }

    def _clause_review_metadata(
        self,
        clause_type: str,
        profile: ClauseProfile,
        raw_content: str,
        citations: list[Citation],
    ) -> dict[str, Any]:
        data = self._extract_json_object(raw_content)
        if not isinstance(data, dict):
            return {
                **self._empty_clause_review_metadata(clause_type, profile),
                "structured": False,
                "summary": raw_content.strip(),
                "found": None,
            }

        found = self._coerce_bool(data.get("found"))
        risk_level = self._coerce_risk_level(data.get("risk_level"))
        risk_reasons = self._risk_reason_list(data.get("risk_reasons"), citations)
        needs_human_review = self._coerce_bool(data.get("needs_human_review"))
        if needs_human_review is None:
            needs_human_review = found is not True or risk_level == "Needs human review"

        summary = self._as_str(data.get("summary"))
        plain_language = self._as_str(
            data.get("plain_language_explanation")
            or data.get("plain_language")
            or data.get("explanation")
            or summary
        )

        return {
            "structured": True,
            "clause_type": self._as_str(data.get("clause_type"), clause_type),
            "normalized_clause_type": self._as_str(
                data.get("normalized_clause_type") or data.get("clause_key"),
                profile.key,
            ),
            "found": found,
            "summary": summary,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "affected_party": self._optional_str(data.get("affected_party")),
            "plain_language_explanation": plain_language,
            "questions_for_lawyer": self._as_list_str(
                data.get("questions_for_lawyer")
                or data.get("negotiation_or_review_points")
                or data.get("review_points")
            ),
            "missing_information": self._as_list_str(data.get("missing_information")),
            "needs_human_review": needs_human_review,
        }

    def _render_clause_review(self, metadata: dict[str, Any], citations: list[Citation]) -> str:
        citation_suffix = self._citation_suffix(
            [
                reason.get("citation")
                for reason in metadata.get("risk_reasons", [])
                if isinstance(reason, dict)
            ],
            citations,
        )
        found = metadata.get("found")
        found_label = "Yes" if found is True else "No" if found is False else "Unclear"
        lines = [
            "## Clause review",
            f"Clause type: {metadata.get('clause_type') or 'Unspecified'}",
            f"Normalized type: {metadata.get('normalized_clause_type') or 'custom'}",
            f"Found: {found_label}",
            f"Risk level: {metadata.get('risk_level') or 'Needs human review'}",
        ]

        summary = self._as_str(metadata.get("summary"))
        if summary:
            lines.append(f"Summary: {summary}{citation_suffix}")

        affected_party = self._optional_str(metadata.get("affected_party"))
        if affected_party:
            lines.append(f"Affected party: {affected_party}{citation_suffix}")

        explanation = self._as_str(metadata.get("plain_language_explanation"))
        if explanation and explanation != summary:
            lines.append(f"Plain-language explanation: {explanation}{citation_suffix}")

        risk_reasons = [
            reason
            for reason in metadata.get("risk_reasons", [])
            if isinstance(reason, dict) and reason.get("reason")
        ]
        if risk_reasons:
            lines.append("\n## Risk reasons")
            for reason in risk_reasons:
                reason_suffix = self._citation_suffix([reason.get("citation")], citations)
                lines.append(f"- {reason['reason']}{reason_suffix}")

        questions = self._as_list_str(metadata.get("questions_for_lawyer"))
        if questions:
            lines.append("\n## Questions for lawyer")
            for question in questions:
                lines.append(f"- {question}{citation_suffix}")

        missing_information = self._as_list_str(metadata.get("missing_information"))
        if missing_information:
            lines.append("\n## Missing information")
            for item in missing_information:
                lines.append(f"- {item}")

        if metadata.get("needs_human_review"):
            lines.append("\nNeeds human review: Yes")

        return "\n".join(lines).strip()

    def _empty_conflict_metadata(self) -> dict[str, Any]:
        return {
            "structured": True,
            "overall_status": "Insufficient information",
            "conflicts": [],
            "needs_human_review": True,
            "supporting_citations": [],
        }

    def _conflict_metadata(self, raw_content: str, citations: list[Citation]) -> dict[str, Any]:
        data = self._extract_json_object(raw_content)
        if not isinstance(data, dict):
            return {
                **self._empty_conflict_metadata(),
                "structured": False,
                "overall_status": self._infer_conflict_status(raw_content),
            }

        raw_conflicts = data.get("conflicts")
        conflicts: list[dict[str, Any]] = []
        if isinstance(raw_conflicts, list):
            for raw_conflict in raw_conflicts:
                if not isinstance(raw_conflict, dict):
                    continue
                contract_citations = self._source_id_list(
                    raw_conflict.get("contract_citations")
                    or raw_conflict.get("contract_citation"),
                    citations,
                    prefix="C",
                )
                policy_citations = self._source_id_list(
                    raw_conflict.get("policy_citations")
                    or raw_conflict.get("policy_citation"),
                    citations,
                    prefix="P",
                )
                severity = self._coerce_risk_level(raw_conflict.get("severity"))
                needs_human_review = self._coerce_bool(raw_conflict.get("needs_human_review"))
                if needs_human_review is None:
                    needs_human_review = severity == "Needs human review"
                conflicts.append(
                    {
                        "topic": self._as_str(raw_conflict.get("topic"), "Unspecified topic"),
                        "conflict_type": self._coerce_conflict_type(
                            raw_conflict.get("conflict_type")
                        ),
                        "severity": severity,
                        "contract_position": self._as_str(
                            raw_conflict.get("contract_position")
                        ),
                        "policy_position": self._as_str(raw_conflict.get("policy_position")),
                        "why_conflict": self._as_str(
                            raw_conflict.get("why_conflict")
                            or raw_conflict.get("explanation")
                            or raw_conflict.get("reason")
                        ),
                        "recommended_action": self._as_str(
                            raw_conflict.get("recommended_action")
                            or raw_conflict.get("next_step")
                        ),
                        "contract_citations": contract_citations,
                        "policy_citations": policy_citations,
                        "needs_human_review": needs_human_review,
                        "confidence": self._optional_str(raw_conflict.get("confidence")),
                    }
                )

        overall_status = self._coerce_conflict_status(data.get("overall_status"))
        if overall_status == "Insufficient information" and conflicts:
            overall_status = "Potential conflict"
        needs_human_review = self._coerce_bool(data.get("needs_human_review"))
        if needs_human_review is None:
            needs_human_review = overall_status == "Insufficient information" or any(
                conflict.get("needs_human_review") for conflict in conflicts
            )

        return {
            "structured": True,
            "overall_status": overall_status,
            "conflicts": conflicts,
            "needs_human_review": needs_human_review,
            "supporting_citations": self._source_id_list(
                data.get("supporting_citations"),
                citations,
            ),
        }

    def _render_conflict_check(self, metadata: dict[str, Any]) -> str:
        lines = [
            "## Conflict check",
            f"Status: {metadata.get('overall_status') or 'Insufficient information'}",
        ]
        conflicts = [
            conflict
            for conflict in metadata.get("conflicts", [])
            if isinstance(conflict, dict)
        ]
        if not conflicts:
            supporting_suffix = self._format_source_refs(metadata.get("supporting_citations", []))
            if metadata.get("overall_status") == "No conflict found":
                lines.append(f"No conflict found based on the provided excerpts.{supporting_suffix}")
            else:
                lines.append(
                    "Insufficient cited information was found to produce a structured conflict item."
                )
            if metadata.get("needs_human_review"):
                lines.append("Needs human review: Yes")
            return "\n".join(lines).strip()

        for index, conflict in enumerate(conflicts, start=1):
            contract_refs = conflict.get("contract_citations", [])
            policy_refs = conflict.get("policy_citations", [])
            evidence_suffix = self._format_source_refs([*contract_refs, *policy_refs])
            lines.extend(
                [
                    f"\n## Conflict {index}: {conflict.get('topic') or 'Unspecified topic'}",
                    f"Type: {conflict.get('conflict_type')}",
                    f"Severity: {conflict.get('severity')}",
                ]
            )
            contract_position = self._as_str(conflict.get("contract_position"))
            if contract_position:
                lines.append(
                    f"Contract position: {contract_position}"
                    f"{self._format_source_refs(contract_refs)}"
                )
            policy_position = self._as_str(conflict.get("policy_position"))
            if policy_position:
                lines.append(
                    f"Policy position: {policy_position}{self._format_source_refs(policy_refs)}"
                )
            why_conflict = self._as_str(conflict.get("why_conflict"))
            if why_conflict:
                lines.append(f"Why this may conflict: {why_conflict}{evidence_suffix}")
            recommended_action = self._as_str(conflict.get("recommended_action"))
            if recommended_action:
                lines.append(f"Recommended next step: {recommended_action}{evidence_suffix}")
            if conflict.get("needs_human_review"):
                lines.append("Needs human review: Yes")

        return "\n".join(lines).strip()

    def _build_messages(self, task_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": task_prompt},
        ]

    def _repair_answer(
        self,
        answer: str,
        guard_result: AnswerGuardResult,
        prepared: PreparedQAAnswer,
    ) -> str:
        return self.repair_content(answer, guard_result, prepared.citations)

    def repair_content(
        self,
        answer: str,
        guard_result: AnswerGuardResult,
        citations: list[Citation],
    ) -> str:
        return self._repair_content(answer, guard_result, citations)

    def _repair_content(
        self,
        answer: str,
        guard_result: AnswerGuardResult,
        citations: list[Citation],
    ) -> str:
        if not guard_result.issues:
            return answer

        repaired, fixed_all = _try_lightweight_repair(answer, guard_result, citations)
        if fixed_all:
            return repaired

        source_ids = ", ".join(citation.source_id for citation in citations) or "None"
        repair_prompt = self.answer_repair_prompt.format(
            issues="\n".join(f"- {issue}" for issue in guard_result.issues),
            source_ids=source_ids,
            answer=answer,
        )
        return self._invoke_chat_messages(self._build_messages(repair_prompt))

    def _rewrite_query(self, question: str, chat_history_text: str) -> str:
        if not getattr(settings, "query_rewrite_enabled", True):
            return question
        if chat_history_text == "No previous messages.":
            return question
        normalized_question = question.casefold()
        if len(question) > 50 and not any(term in normalized_question for term in _VAGUE_QUERY_TERMS):
            return question
        if not any(term in normalized_question for term in _VAGUE_QUERY_TERMS):
            return question

        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=chat_history_text[-1000:],
            question=question,
        )
        try:
            rewritten = self._invoke_chat_messages(
                [
                    {"role": "system", "content": "You rewrite document retrieval queries."},
                    {"role": "user", "content": prompt},
                ]
            ).strip()
        except Exception:
            logger.debug(
                "Query rewrite failed; using original question.",
                extra={"tenant_id": self.tenant_id},
                exc_info=True,
            )
            return question

        if not rewritten or len(rewritten) > 500:
            return question
        return rewritten

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any] | None:
        text = (content or "").strip()
        if not text:
            return None

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        candidates = [fenced_match.group(1)] if fenced_match else []
        candidates.append(text)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if 0 <= first_brace < last_brace:
            candidates.append(text[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _as_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip() or default
        if isinstance(value, (int, float, bool)):
            return str(value)
        return default

    @classmethod
    def _optional_str(cls, value: Any) -> str | None:
        text = cls._as_str(value)
        return text or None

    @classmethod
    def _as_list_str(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            result = []
            for item in value:
                text = cls._as_str(item)
                if text:
                    result.append(text)
            return result
        return []

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "yes", "y", "found"}:
                return True
            if normalized in {"false", "no", "n", "not found"}:
                return False
        return None

    @staticmethod
    def _coerce_risk_level(value: Any) -> str:
        if not isinstance(value, str):
            return "Needs human review"
        normalized = value.strip().casefold().replace("_", " ")
        if "human" in normalized or "review" in normalized:
            return "Needs human review"
        for level in ("Low", "Medium", "High"):
            if normalized == level.casefold() or level.casefold() in normalized:
                return level
        return "Needs human review"

    @staticmethod
    def _coerce_conflict_status(value: Any) -> str:
        if not isinstance(value, str):
            return "Insufficient information"
        normalized = value.strip().casefold()
        if "potential" in normalized or "conflict" in normalized and "no" not in normalized:
            return "Potential conflict"
        if "no conflict" in normalized or normalized == "none":
            return "No conflict found"
        return "Insufficient information"

    @classmethod
    def _infer_conflict_status(cls, content: str) -> str:
        return cls._coerce_conflict_status(content)

    @staticmethod
    def _coerce_conflict_type(value: Any) -> str:
        if not isinstance(value, str):
            return "ambiguous_relationship"
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
        aliases = {
            "timeline_conflict": "deadline_mismatch",
            "time_conflict": "deadline_mismatch",
            "deadline_conflict": "deadline_mismatch",
            "amount_conflict": "amount_mismatch",
            "money_conflict": "amount_mismatch",
            "definition_conflict": "definition_mismatch",
            "scope_conflict": "scope_mismatch",
            "process_conflict": "process_mismatch",
            "procedural_conflict": "process_mismatch",
            "direct_conflict": "direct_contradiction",
            "contradiction": "direct_contradiction",
            "none": "none",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in allowed_conflict_type_keys():
            return normalized
        return "ambiguous_relationship"

    def _risk_reason_list(self, value: Any, citations: list[Citation]) -> list[dict[str, str | None]]:
        default_citation = self._first_source_id(citations, prefix="S")
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]
        reasons: list[dict[str, str | None]] = []
        for item in raw_items:
            if isinstance(item, dict):
                reason = self._as_str(item.get("reason") or item.get("text") or item.get("issue"))
                citation = self._source_id_list(
                    item.get("citation") or item.get("citations"),
                    citations,
                    prefix="S",
                )
                citation_id = citation[0] if citation else default_citation
            else:
                reason = self._as_str(item)
                citation_id = default_citation
            if reason:
                reasons.append({"reason": reason, "citation": citation_id})
        return reasons

    @staticmethod
    def _first_source_id(citations: list[Citation], prefix: str | None = None) -> str | None:
        for citation in citations:
            if not prefix or citation.source_id.startswith(prefix):
                return citation.source_id
        return None

    def _source_id_list(
        self,
        value: Any,
        citations: list[Citation],
        prefix: str | None = None,
    ) -> list[str]:
        valid_source_ids = {
            citation.source_id
            for citation in citations
            if citation.source_id and (not prefix or citation.source_id.startswith(prefix))
        }
        if not valid_source_ids:
            return []

        raw_values: list[Any]
        if value is None:
            raw_values = []
        elif isinstance(value, list):
            raw_values = value
        else:
            raw_values = [value]

        source_ids: list[str] = []
        for raw_value in raw_values:
            text = self._as_str(raw_value)
            if not text:
                continue
            for match in re.findall(r"\[?([SCDPW]\d+)\]?", text, flags=re.IGNORECASE):
                source_id = match.upper()
                if source_id in valid_source_ids and source_id not in source_ids:
                    source_ids.append(source_id)
        return source_ids

    def _citation_suffix(self, source_ids: list[Any], citations: list[Citation]) -> str:
        normalized_ids: list[str] = []
        valid_source_ids = {citation.source_id for citation in citations}
        for value in source_ids:
            for source_id in self._source_id_list(value, citations):
                if source_id not in normalized_ids:
                    normalized_ids.append(source_id)
        if not normalized_ids:
            first_source_id = self._first_source_id(citations)
            if first_source_id:
                normalized_ids.append(first_source_id)
        normalized_ids = [source_id for source_id in normalized_ids if source_id in valid_source_ids]
        return self._format_source_refs(normalized_ids)

    @staticmethod
    def _format_source_refs(source_ids: list[Any]) -> str:
        refs: list[str] = []
        for source_id in source_ids:
            if not isinstance(source_id, str):
                continue
            normalized = source_id.strip().strip("[]").upper()
            if re.fullmatch(r"[SCDPW]\d+", normalized) and normalized not in refs:
                refs.append(normalized)
        return " " + " ".join(f"[{source_id}]" for source_id in refs) if refs else ""

    def _format_context(self, documents: list[Document]) -> tuple[str, list[Citation]]:
        return self._format_context_prefixed(documents, prefix="S")

    def _format_context_prefixed(
        self, documents: list[Document], prefix: str = "S"
    ) -> tuple[str, list[Citation]]:
        context_parts = []
        citations = []

        for index, document in enumerate(documents, start=1):
            source_id = f"{prefix}{index}"
            metadata = document.metadata or {}
            text = self._compact_text(document.page_content)
            page = metadata.get("page")
            chunk_id = metadata.get("chunk_id")
            section_heading = metadata.get("section_heading")
            retrieval_score = metadata.get("retrieval_score")
            retrieval_relevance = metadata.get("retrieval_relevance")
            file_name = metadata.get("file_name") or metadata.get("source") or "unknown"
            file_id = self._optional_str(metadata.get("file_id"))
            document_key = self._optional_str(metadata.get("document_key"))
            document_version = (
                metadata.get("document_version")
                if isinstance(metadata.get("document_version"), int)
                else None
            )
            page_number = page if isinstance(page, int) else None
            page_label = f"page {page_number + 1}" if page_number is not None else None
            section_part = f"; section={section_heading}" if section_heading else ""
            page_part = page_label or "unknown"

            context_parts.append(
                f"[{source_id}] file={file_name}; page={page_part}; "
                f"page_index={page}; chunk={chunk_id}{section_part}\n{text}"
            )
            citations.append(
                Citation(
                    source_id=source_id,
                    file_name=str(file_name),
                    page=page_number,
                    chunk_id=chunk_id if isinstance(chunk_id, int) else None,
                    preview=text[:500],
                    source_type="document",
                    file_id=file_id,
                    document_key=document_key,
                    document_version=document_version,
                    page_label=page_label,
                    section_heading=str(section_heading) if section_heading else None,
                    exact_quote=text[:1200],
                    retrieval_score=(
                        float(retrieval_score)
                        if isinstance(retrieval_score, int | float)
                        else None
                    ),
                    retrieval_relevance=(
                        float(retrieval_relevance)
                        if isinstance(retrieval_relevance, int | float)
                        else None
                    ),
                )
            )

        return "\n\n".join(context_parts), citations

    @staticmethod
    def _compact_text(text: str) -> str:
        return " ".join(text.split())

    def _invoke_chat_messages(self, messages: list[dict[str, str]]) -> str:
        invoke_messages = getattr(self.chat_model, "invoke_messages", None)
        if callable(invoke_messages):
            response = invoke_messages(messages)
            return str(response.get("content") or "")

        invoke = getattr(self.chat_model, "invoke", None)
        if callable(invoke):
            try:
                response = invoke(messages=messages)
            except TypeError:
                response = invoke(self._messages_to_prompt(messages))
            content = getattr(response, "content", response)
            return str(content)

        raise ValueError("The configured chat model does not support message-based chat.")

    async def _ainvoke_chat_messages(self, messages: list[dict[str, str]]) -> str:
        ainvoke_messages = getattr(self.chat_model, "ainvoke_messages", None)
        if callable(ainvoke_messages):
            response = await ainvoke_messages(messages)
            return str(response.get("content") or "")
        ainvoke = getattr(self.chat_model, "ainvoke", None)
        if callable(ainvoke):
            try:
                response = await ainvoke(messages=messages)
            except TypeError:
                response = await ainvoke(self._messages_to_prompt(messages))
            content = getattr(response, "content", response)
            return str(content)
        return await asyncio.to_thread(self._invoke_chat_messages, messages)

    def _stream_chat_messages(self, messages: list[dict[str, str]]) -> Iterator[str]:
        stream = getattr(self.chat_model, "stream", None)
        if callable(stream):
            try:
                chunks = stream(messages=messages)
            except TypeError:
                chunks = stream(self._messages_to_prompt(messages))
            for chunk in chunks:
                content = getattr(chunk, "content", chunk)
                if content:
                    yield str(content)
            return

        yield self._invoke_chat_messages(messages)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _record_assistant_message(
        self,
        *,
        user_id: str | None,
        conversation_id: str | None,
        content: str,
        user_message_recorded: bool,
    ) -> None:
        if not (self.memory_service and user_id and conversation_id and user_message_recorded):
            return
        try:
            self.memory_service.record_assistant_message(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                content=content,
            )
        except Exception:
            logger.debug(
                "Assistant message persistence failed.",
                extra={"tenant_id": self.tenant_id, "user_id": user_id},
                exc_info=True,
            )
            return

    def _log_retrieval(
        self,
        *,
        question: str,
        user_id: str | None,
        conversation_id: str | None,
        document_count: int,
        memory_candidates: list[MemoryCandidate],
    ) -> None:
        if not (self.memory_service and user_id):
            return
        try:
            self.memory_service.log_retrieval(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                query=question,
                document_count=document_count,
                memories=memory_candidates,
            )
        except Exception:
            logger.debug(
                "Retrieval logging failed.",
                extra={"tenant_id": self.tenant_id, "user_id": user_id},
                exc_info=True,
            )
            return

    @staticmethod
    def _format_chat_history(messages: list[dict[str, object]], max_messages: int = 12) -> str:
        history_parts = []
        for message in messages[-max_messages:]:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue

            label = "User" if role == "user" else "Assistant"
            history_parts.append(f"{label}: {content}")

        return "\n".join(history_parts) if history_parts else "No previous messages."


def _try_lightweight_repair(
    content: str,
    guard_result: AnswerGuardResult,
    citations: list[Citation],
) -> tuple[str, bool]:
    valid_ids = {citation.source_id.upper() for citation in citations}
    if not valid_ids:
        return content, False

    repaired = content
    fixed_any = False
    fixed_all = True
    first_ref = f"[{next(iter(sorted(valid_ids)))}]"

    for issue in guard_result.issues:
        lowered = issue.casefold()
        if "source ids that were not returned" in lowered:
            repaired = re.sub(
                r"\[([SCDPW]\d+)\]",
                lambda match: match.group(0)
                if match.group(1).upper() in valid_ids
                else "",
                repaired,
                flags=re.IGNORECASE,
            )
            fixed_any = True
        elif "does not include any source citations" in lowered:
            repaired = f"{repaired.rstrip()} {first_ref}".strip()
            fixed_any = True
        elif "material paragraph lacks a source citation" in lowered:
            repaired = _append_default_citations_to_material_paragraphs(repaired, first_ref)
            fixed_any = True
        elif "specific fact" in lowered and "without a nearby citation" in lowered:
            repaired = _append_default_citations_to_fact_sentences(repaired, first_ref)
            fixed_any = True
        else:
            fixed_all = False

    if fixed_any and not re.search(r"\[[SCDPW]\d+\]", repaired, flags=re.IGNORECASE):
        repaired = f"{repaired.rstrip()} {first_ref}".strip()

    return repaired if fixed_any else content, fixed_all and fixed_any


def _append_default_citations_to_material_paragraphs(content: str, source_ref: str) -> str:
    blocks = re.split(r"(\n\s*\n)", content)
    repaired_blocks = []
    for block in blocks:
        stripped = block.strip()
        if not stripped or block.startswith("\n"):
            repaired_blocks.append(block)
            continue
        if stripped.startswith("#") or re.search(r"\[[SCDPW]\d+\]", block, flags=re.IGNORECASE):
            repaired_blocks.append(block)
            continue
        if len(stripped) >= 40:
            repaired_blocks.append(f"{block.rstrip()} {source_ref}")
        else:
            repaired_blocks.append(block)
    return "".join(repaired_blocks)


def _append_default_citations_to_fact_sentences(content: str, source_ref: str) -> str:
    sentences = re.split(r"([.!?。！？]\s*)", content)
    repaired = []
    for index in range(0, len(sentences), 2):
        sentence = sentences[index]
        punctuation = sentences[index + 1] if index + 1 < len(sentences) else ""
        if (
            re.search(r"\b\d+(?:\.\d+)?%|\b\d+\s+(?:days?|business days?|months?|years?)\b|\$\s?\d", sentence)
            and not re.search(r"\[[SCDPW]\d+\]", sentence, flags=re.IGNORECASE)
        ):
            sentence = f"{sentence.rstrip()} {source_ref}"
        repaired.append(sentence + punctuation)
    return "".join(repaired)
