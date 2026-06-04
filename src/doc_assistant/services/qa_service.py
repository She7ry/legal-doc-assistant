from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryCandidate, MemoryUsage
from doc_assistant.memory.service import MemoryService
from doc_assistant.models.language_model import build_chat_model
from doc_assistant.retrieval.vector_store import DocumentVectorStore
from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.utils.prompt_loader import load_prompt


@dataclass(frozen=True)
class PreparedQAAnswer:
    prompt: str
    citations: list[Citation]
    memories_used: list[MemoryUsage]
    user_id: str | None
    conversation_id: str | None
    user_message_recorded: bool


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
        self.prompt = PromptTemplate.from_template(load_prompt("document_qa.txt"))
        self.general_prompt = PromptTemplate.from_template(load_prompt("general_chat.txt"))
        self.clause_review_prompt = PromptTemplate.from_template(load_prompt("clause_review.txt"))
        self.conflict_check_prompt = PromptTemplate.from_template(load_prompt("conflict_check.txt"))
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
        content = self._invoke_chat(prepared.prompt)
        self.record_prepared_answer(prepared, content)
        return QAAnswer(
            content=content,
            citations=prepared.citations,
            memories_used=prepared.memories_used,
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
                # Memory failure should not prevent citation-first document QA.
                memory_candidates = []
                memory_context = "No relevant user memory."

        documents = self.vector_store.search(question)
        self._log_retrieval(
            question=question,
            user_id=user_id,
            conversation_id=resolved_conversation_id,
            document_count=len(documents),
            memory_candidates=memory_candidates,
        )
        chat_history_text = self._format_chat_history(chat_history or [])
        if not documents:
            prompt = self.general_prompt.format(
                question=question,
                chat_history=chat_history_text,
                user_memory=memory_context,
            )
            memories_used = (
                self.memory_service.usages_from_candidates(memory_candidates)
                if self.memory_service
                else []
            )
            return PreparedQAAnswer(
                prompt=prompt,
                citations=[],
                memories_used=memories_used,
                user_id=user_id,
                conversation_id=resolved_conversation_id,
                user_message_recorded=user_message_recorded,
            )

        context, citations = self._format_context(documents)
        prompt = self.prompt.format(
            question=question,
            context=context,
            chat_history=chat_history_text,
            user_memory=memory_context,
        )
        memories_used = (
            self.memory_service.usages_from_candidates(memory_candidates)
            if self.memory_service
            else []
        )
        return PreparedQAAnswer(
            prompt=prompt,
            citations=citations,
            memories_used=memories_used,
            user_id=user_id,
            conversation_id=resolved_conversation_id,
            user_message_recorded=user_message_recorded,
        )

    def stream_prepared_answer(self, prepared: PreparedQAAnswer) -> Iterator[str]:
        yield from self._stream_chat(prepared.prompt)

    def record_prepared_answer(self, prepared: PreparedQAAnswer, content: str) -> None:
        self._record_assistant_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content=content,
            user_message_recorded=prepared.user_message_recorded,
        )

    def review_clause(self, clause_type: str, top_k: int | None = None) -> QAAnswer:
        """Search indexed documents for a specific clause type and assess risk level."""
        documents = self.vector_store.search(clause_type, k=top_k)
        if not documents:
            return QAAnswer(
                content="No relevant content found in indexed documents for the requested clause type.",
                citations=[],
            )

        context, citations = self._format_context(documents)
        prompt = self.clause_review_prompt.format(clause_type=clause_type, context=context)
        content = self._invoke_chat(prompt)
        return QAAnswer(content=content, citations=citations)

    def check_conflict(self, contract_query: str, policy_query: str, top_k: int | None = None) -> QAAnswer:
        """Retrieve contract and policy excerpts separately, then check for conflicts."""
        contract_docs = self.vector_store.search(contract_query, k=top_k)
        policy_docs = self.vector_store.search(policy_query, k=top_k)

        if not contract_docs and not policy_docs:
            return QAAnswer(
                content="No relevant content found in indexed documents for conflict analysis.",
                citations=[],
            )

        contract_context, contract_citations = self._format_context_prefixed(contract_docs, prefix="C")
        policy_context, policy_citations = self._format_context_prefixed(policy_docs, prefix="P")

        prompt = self.conflict_check_prompt.format(
            contract_context=contract_context or "No contract excerpts found.",
            policy_context=policy_context or "No policy excerpts found.",
        )
        content = self._invoke_chat(prompt)
        return QAAnswer(content=content, citations=contract_citations + policy_citations)

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
            file_name = metadata.get("file_name") or metadata.get("source") or "unknown"

            context_parts.append(
                f"[{source_id}] file={file_name}; page={page}; chunk={chunk_id}\n{text}"
            )
            citations.append(
                Citation(
                    source_id=source_id,
                    file_name=str(file_name),
                    page=page if isinstance(page, int) else None,
                    chunk_id=chunk_id if isinstance(chunk_id, int) else None,
                    preview=text[:500],
                )
            )

        return "\n\n".join(context_parts), citations

    @staticmethod
    def _compact_text(text: str) -> str:
        return " ".join(text.split())

    def _invoke_chat(self, prompt: str) -> str:
        response = self.chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content)

    def _stream_chat(self, prompt: str) -> Iterator[str]:
        stream = getattr(self.chat_model, "stream", None)
        if callable(stream):
            for chunk in stream(prompt):
                content = getattr(chunk, "content", chunk)
                if content:
                    yield str(content)
            return

        yield self._invoke_chat(prompt)

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
