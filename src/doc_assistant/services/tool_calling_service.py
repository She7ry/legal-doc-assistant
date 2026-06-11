from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from hashlib import sha1
import json
from typing import Any

from langchain_core.documents import Document

from doc_assistant.config.settings import settings
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.answer_guard import validate_answer
from doc_assistant.services.evidence import build_evidence_profile
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.tools.web_search import (
    DisabledWebSearchClient,
    WebSearchClient,
    WebSearchResult,
    build_web_search_client,
)
from doc_assistant.utils.prompt_loader import load_base_legal_prompt, load_prompt


def build_tool_system_prompt() -> str:
    return f"{load_base_legal_prompt()}\n\n{load_prompt('tool_calling_system.txt')}"


@dataclass(frozen=True)
class WebSource:
    source_id: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class ToolCallTrace:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class ToolCallingAnswer:
    content: str
    citations: list[Citation] = field(default_factory=list)
    web_sources: list[WebSource] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ToolExecutionState:
    citations: list[Citation] = field(default_factory=list)
    web_sources: list[WebSource] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    document_source_ids: dict[str, str] = field(default_factory=dict)


class ToolCallingChatService:
    def __init__(
        self,
        qa_service: DocumentQAService,
        web_search_client: WebSearchClient | None = None,
    ) -> None:
        self.qa_service = qa_service
        self.chat_model = qa_service.chat_model
        self.invoke_messages = getattr(self.chat_model, "invoke_messages", None)
        if not callable(self.invoke_messages):
            raise ValueError("The configured chat model does not support tool calling.")
        self.vector_store = qa_service.vector_store
        self.web_search_client = web_search_client or build_web_search_client()
        self._tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-call")

    def ask(
        self,
        question: str,
        *,
        chat_history: list[dict[str, object]] | None = None,
        enable_web_search: bool = False,
        max_tool_iterations: int | None = None,
    ) -> ToolCallingAnswer:
        state = _ToolExecutionState()
        tools = self._tool_schemas(enable_web_search=enable_web_search)
        messages = self._initial_messages(question, chat_history or [])
        iterations = _clamp_int(
            max_tool_iterations or settings.tool_call_max_iterations,
            minimum=1,
            maximum=10,
        )

        for _ in range(iterations):
            message = self.invoke_messages(messages, tools=tools, tool_choice="auto")
            tool_calls = _normalise_tool_calls(message)
            if not tool_calls:
                return self._finalize_answer(str(message.get("content") or ""), state)

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for tool_call in tool_calls:
                tool_result = self._execute_tool_call(tool_call, state, enable_web_search)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        final_message = self.invoke_messages(messages, tools=None, tool_choice=None)
        return self._finalize_answer(str(final_message.get("content") or ""), state)

    def _finalize_answer(self, content: str, state: _ToolExecutionState) -> ToolCallingAnswer:
        guard_citations = state.citations + _web_source_citations(state.web_sources)
        guard_result = validate_answer(
            content,
            guard_citations,
            has_retrieved_documents=bool(guard_citations),
        )
        if guard_result.needs_repair:
            content = self.qa_service.repair_content(content, guard_result, guard_citations)
            guard_result = validate_answer(
                content,
                guard_citations,
                has_retrieved_documents=bool(guard_citations),
            )

        return ToolCallingAnswer(
            content=content,
            citations=state.citations,
            web_sources=state.web_sources,
            tool_calls=state.tool_calls,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata={
                "evidence": build_evidence_profile(content, guard_citations, guard_result.issues)
            },
        )

    def _initial_messages(
        self,
        question: str,
        chat_history: list[dict[str, object]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": build_tool_system_prompt()}]
        history_window = _clamp_int(settings.tool_call_history_window, minimum=0, maximum=100)
        for message in chat_history[-history_window:] if history_window else []:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def _tool_schemas(self, *, enable_web_search: bool) -> list[dict[str, Any]]:
        return [schema for schema, _handler in self._enabled_tools(enable_web_search).values()]

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        state: _ToolExecutionState,
        enable_web_search: bool,
    ) -> dict[str, Any]:
        name = tool_call["function"]["name"]
        arguments = _parse_tool_arguments(tool_call)

        future = None
        try:
            future = self._tool_executor.submit(
                self._run_tool_call,
                name,
                arguments,
                state,
                enable_web_search,
            )
            result = future.result(timeout=max(1, settings.tool_call_timeout_seconds))
        except FutureTimeoutError:
            if future is not None:
                future.cancel()
            result = {
                "error": f"Tool execution timed out after {settings.tool_call_timeout_seconds} seconds."
            }
        except Exception as exc:
            result = {"error": str(exc)}

        state.tool_calls.append(
            ToolCallTrace(
                tool_call_id=tool_call["id"],
                name=name,
                arguments=arguments,
                result=result,
            )
        )
        return result

    def _enabled_tools(self, enable_web_search: bool) -> dict[str, tuple[dict[str, Any], Any]]:
        tools: dict[str, tuple[dict[str, Any], Any]] = {
            "search_documents": (SEARCH_DOCUMENTS_TOOL_SCHEMA, self._search_documents),
        }
        if enable_web_search:
            tools["web_search"] = (WEB_SEARCH_TOOL_SCHEMA, self._web_search)
        return tools

    def _run_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
        enable_web_search: bool,
    ) -> dict[str, Any]:
        tools = self._enabled_tools(enable_web_search)
        match = tools.get(name)
        if match is None:
            if name == "web_search":
                raise RuntimeError("web_search was called but web search is not enabled.")
            return {"error": f"Unknown tool: {name}"}
        _schema, handler = match
        return handler(arguments, state)

    def _search_documents(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        query = _required_string(arguments, "query", max_length=500)
        top_k = _clamp_int(int(arguments.get("top_k") or settings.top_k), minimum=1, maximum=10)
        documents = self.vector_store.search(query, k=top_k)

        results = []
        for document in documents:
            identity = _document_identity(document)
            source_id = state.document_source_ids.get(identity)
            is_new_source = source_id is None
            if source_id is None:
                source_id = f"D{len(state.citations) + 1}"
                state.document_source_ids[identity] = source_id
            item = _document_result(source_id, document)
            if is_new_source:
                state.citations.append(
                    Citation(
                        source_id=source_id,
                        file_name=item["file_name"],
                        page=item["page"],
                        chunk_id=item["chunk_id"],
                        preview=item["content"][:500],
                        source_type="document",
                        file_id=item["file_id"],
                        document_key=item["document_key"],
                        document_version=item["document_version"],
                        page_label=item["page_label"],
                        section_heading=item["section_heading"],
                        exact_quote=item["content"][:1200],
                        retrieval_score=item["retrieval_score"],
                        retrieval_relevance=item["retrieval_relevance"],
                    )
                )
            results.append(item)

        return {"query": query, "result_count": len(results), "results": results}

    def _web_search(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        web_search_client = self.web_search_client
        if isinstance(web_search_client, DisabledWebSearchClient):
            raise RuntimeError("Web search is disabled. Set DOC_ASSISTANT_WEB_SEARCH_ENABLED=true.")

        query = _required_string(arguments, "query", max_length=300)
        top_k = _clamp_int(
            int(arguments.get("max_results") or settings.web_search_max_results),
            minimum=1,
            maximum=10,
        )
        recency_days = arguments.get("recency_days")
        if recency_days is not None:
            recency_days = _clamp_int(int(recency_days), minimum=1, maximum=365)
        domains = arguments.get("domains")
        if domains is not None and not isinstance(domains, list):
            raise ValueError("domains must be a list of domain strings.")
        clean_domains = [str(domain).strip() for domain in domains or [] if str(domain).strip()]

        search_results = web_search_client.search(
            query,
            max_results=top_k,
            recency_days=recency_days,
            domains=clean_domains,
        )
        results = []
        for result in search_results:
            source_id = f"W{len(state.web_sources) + 1}"
            source = _web_source(source_id, result)
            state.web_sources.append(source)
            results.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                    "published_at": source.published_at,
                    "source": source.source,
                }
            )

        return {"query": query, "result_count": len(results), "results": results}


SEARCH_DOCUMENTS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search uploaded/indexed legal documents and return cited excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query for uploaded documents.",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of document excerpts to retrieve.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

WEB_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search public web pages for recent or external context. Do not include confidential "
            "contract excerpts in the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Public web search query.",
                    "minLength": 1,
                    "maxLength": 300,
                },
                "recency_days": {
                    "type": "integer",
                    "description": "Optional recency window in days.",
                    "minimum": 1,
                    "maximum": 365,
                },
                "domains": {
                    "type": "array",
                    "description": "Optional domain filters such as sec.gov or court.gov.",
                    "items": {"type": "string"},
                    "maxItems": 5,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of web results to return.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _normalise_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return [_normalise_tool_call(call, index) for index, call in enumerate(tool_calls, start=1)]

    function_call = message.get("function_call")
    if function_call:
        return [
            {
                "id": "legacy_function_call_1",
                "type": "function",
                "function": function_call,
            }
        ]
    return []


def _normalise_tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "id": call.get("id") or f"tool_call_{index}",
        "type": call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": function.get("arguments") or "{}",
        },
    }


def _parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw_arguments = tool_call["function"].get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON arguments for {tool_call['function']['name']}.") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {tool_call['function']['name']} must be a JSON object.")
    return arguments


def _document_result(source_id: str, document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    content = _compact_text(document.page_content)[:1600]
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")
    section_heading = metadata.get("section_heading")
    retrieval_score = metadata.get("retrieval_score")
    retrieval_relevance = metadata.get("retrieval_relevance")
    file_name = str(metadata.get("file_name") or metadata.get("source") or "unknown")
    page_number = page if isinstance(page, int) else None
    return {
        "source_id": source_id,
        "file_name": file_name,
        "file_id": _optional_string(metadata.get("file_id")),
        "document_key": _optional_string(metadata.get("document_key")),
        "document_version": (
            metadata.get("document_version")
            if isinstance(metadata.get("document_version"), int)
            else None
        ),
        "page": page_number,
        "page_label": f"page {page_number + 1}" if page_number is not None else None,
        "chunk_id": chunk_id if isinstance(chunk_id, int) else None,
        "section_heading": str(section_heading) if section_heading else None,
        "retrieval_score": retrieval_score if isinstance(retrieval_score, int | float) else None,
        "retrieval_relevance": (
            retrieval_relevance if isinstance(retrieval_relevance, int | float) else None
        ),
        "content": content,
    }


def _document_identity(document: Document) -> str:
    metadata = document.metadata or {}
    identity = {
        "file_id": _optional_string(metadata.get("file_id")),
        "document_key": _optional_string(metadata.get("document_key")),
        "document_version": metadata.get("document_version"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "source": _optional_string(metadata.get("source") or metadata.get("file_name")),
    }
    if any(value is not None for value in identity.values()):
        return json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return sha1(_compact_text(document.page_content).encode("utf-8")).hexdigest()


def _web_source(source_id: str, result: WebSearchResult) -> WebSource:
    return WebSource(
        source_id=source_id,
        title=result.title,
        url=result.url,
        snippet=result.snippet,
        published_at=result.published_at,
        source=result.source,
    )


def _web_source_citations(sources: list[WebSource]) -> list[Citation]:
    citations = []
    for source in sources:
        preview = source.snippet or source.title or source.url
        citations.append(
            Citation(
                source_id=source.source_id,
                file_name=source.title or source.url,
                preview=preview,
                source_type="web",
                exact_quote=preview,
            )
        )
    return citations


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_string(arguments: dict[str, Any], name: str, *, max_length: int) -> str:
    value = str(arguments.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    if len(value) > max_length:
        raise ValueError(f"{name} must be {max_length} characters or fewer.")
    return value


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _compact_text(text: str) -> str:
    return " ".join(text.split())
