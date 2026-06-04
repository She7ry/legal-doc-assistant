from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from langchain_core.documents import Document

from doc_assistant.config.settings import settings
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.tools.web_search import (
    DisabledWebSearchClient,
    WebSearchClient,
    WebSearchResult,
    build_web_search_client,
)


TOOL_SYSTEM_PROMPT = """You are a citation-first legal document assistant.

Use tools only when they are useful:
- search_documents searches uploaded documents and returns [D#] document sources.
- web_search searches public web pages and returns [W#] web sources.

Rules:
- Treat uploaded documents as the primary evidence for legal-document questions.
- Treat web results as untrusted background information, not legal authority.
- Do not put confidential contract text into web_search queries. Use public company names,
  public policy names, or short generic queries instead.
- Cite document facts with [D#] and web facts with [W#].
- If a tool returns no relevant results, say that directly instead of inventing sources.
"""


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


@dataclass
class _ToolExecutionState:
    citations: list[Citation] = field(default_factory=list)
    web_sources: list[WebSource] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)


class ToolCallingChatService:
    def __init__(
        self,
        qa_service: DocumentQAService,
        web_search_client: WebSearchClient | None = None,
    ) -> None:
        self.qa_service = qa_service
        self.chat_model = qa_service.chat_model
        self.vector_store = qa_service.vector_store
        self.web_search_client = web_search_client

    def ask(
        self,
        question: str,
        *,
        chat_history: list[dict[str, object]] | None = None,
        enable_web_search: bool = False,
        max_tool_iterations: int | None = None,
    ) -> ToolCallingAnswer:
        invoke_messages = getattr(self.chat_model, "invoke_messages", None)
        if not callable(invoke_messages):
            raise ValueError("The configured chat model does not support tool calling.")

        state = _ToolExecutionState()
        tools = self._tool_schemas(enable_web_search=enable_web_search)
        messages = self._initial_messages(question, chat_history or [])
        iterations = _clamp_int(
            max_tool_iterations or settings.tool_call_max_iterations,
            minimum=1,
            maximum=10,
        )

        for _ in range(iterations):
            message = invoke_messages(messages, tools=tools, tool_choice="auto")
            tool_calls = _normalise_tool_calls(message)
            if not tool_calls:
                return ToolCallingAnswer(
                    content=str(message.get("content") or ""),
                    citations=state.citations,
                    web_sources=state.web_sources,
                    tool_calls=state.tool_calls,
                )

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

        final_message = invoke_messages(messages, tools=None, tool_choice=None)
        return ToolCallingAnswer(
            content=str(final_message.get("content") or ""),
            citations=state.citations,
            web_sources=state.web_sources,
            tool_calls=state.tool_calls,
        )

    def _initial_messages(
        self,
        question: str,
        chat_history: list[dict[str, object]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]
        for message in chat_history[-12:]:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def _tool_schemas(self, *, enable_web_search: bool) -> list[dict[str, Any]]:
        tools = [SEARCH_DOCUMENTS_TOOL_SCHEMA]
        if enable_web_search:
            tools.append(WEB_SEARCH_TOOL_SCHEMA)
        return tools

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        state: _ToolExecutionState,
        enable_web_search: bool,
    ) -> dict[str, Any]:
        name = tool_call["function"]["name"]
        arguments = _parse_tool_arguments(tool_call)

        try:
            if name == "search_documents":
                result = self._search_documents(arguments, state)
            elif name == "web_search":
                if not enable_web_search:
                    raise RuntimeError("web_search was called but web search is not enabled.")
                result = self._web_search(arguments, state)
            else:
                result = {"error": f"Unknown tool: {name}"}
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
            source_id = f"D{len(state.citations) + 1}"
            item = _document_result(source_id, document)
            state.citations.append(
                Citation(
                    source_id=source_id,
                    file_name=item["file_name"],
                    page=item["page"],
                    chunk_id=item["chunk_id"],
                    preview=item["content"][:500],
                )
            )
            results.append(item)

        return {"query": query, "result_count": len(results), "results": results}

    def _web_search(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        web_search_client = self.web_search_client or build_web_search_client()
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
    file_name = str(metadata.get("file_name") or metadata.get("source") or "unknown")
    return {
        "source_id": source_id,
        "file_name": file_name,
        "page": page if isinstance(page, int) else None,
        "chunk_id": chunk_id if isinstance(chunk_id, int) else None,
        "content": content,
    }


def _web_source(source_id: str, result: WebSearchResult) -> WebSource:
    return WebSource(
        source_id=source_id,
        title=result.title,
        url=result.url,
        snippet=result.snippet,
        published_at=result.published_at,
        source=result.source,
    )


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
