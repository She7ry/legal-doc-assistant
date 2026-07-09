"""Agent-callable external tools and tool metadata registry."""

from doc_assistant.tools.document_search import (
    DocumentSearchBackend,
    DocumentSearchExecution,
    DocumentSearchHit,
    DocumentSearchTool,
    SearchDocumentsInput,
)
from doc_assistant.tools.legal_review import check_conflict, review_clause
from doc_assistant.tools.web_search import (
    DisabledWebSearchClient,
    WebSearchClient,
    WebSearchExecution,
    WebSearchInput,
    WebSearchResult,
    WebSearchTool,
    WebSource,
    build_web_search_client,
    web_source,
    web_source_citations,
)

TOOL_REGISTRY = {
    "search_documents": {
        "label": "Search documents",
        "description": "Search uploaded or indexed legal documents and return cited excerpts.",
    },
    "web_search": {
        "label": "Web search",
        "description": "Search public web pages with optional recency and domain filters.",
    },
    "review_clause": {
        "label": "Clause review",
        "description": "Assess a clause type and produce structured risk reasons.",
    },
    "check_conflict": {
        "label": "Conflict check",
        "description": "Compare contract and policy excerpts for inconsistent obligations.",
    },
}

__all__ = [
    "TOOL_REGISTRY",
    "DisabledWebSearchClient",
    "DocumentSearchBackend",
    "DocumentSearchExecution",
    "DocumentSearchHit",
    "DocumentSearchTool",
    "SearchDocumentsInput",
    "WebSearchClient",
    "WebSearchExecution",
    "WebSearchInput",
    "WebSearchResult",
    "WebSearchTool",
    "WebSource",
    "build_web_search_client",
    "check_conflict",
    "review_clause",
    "web_source",
    "web_source_citations",
]
