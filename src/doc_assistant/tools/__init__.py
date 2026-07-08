"""Agent-callable external tools and tool metadata registry."""

from doc_assistant.tools.document_search import (
    SEARCH_DOCUMENTS_TOOL_SCHEMA,
    DocumentSearchBackend,
    DocumentSearchExecution,
    DocumentSearchHit,
    DocumentSearchTool,
)
from doc_assistant.tools.web_search import (
    DisabledWebSearchClient,
    WebSearchClient,
    WebSearchResult,
    build_web_search_client,
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
}

__all__ = [
    "SEARCH_DOCUMENTS_TOOL_SCHEMA",
    "TOOL_REGISTRY",
    "DisabledWebSearchClient",
    "DocumentSearchBackend",
    "DocumentSearchExecution",
    "DocumentSearchHit",
    "DocumentSearchTool",
    "WebSearchClient",
    "WebSearchResult",
    "build_web_search_client",
]
