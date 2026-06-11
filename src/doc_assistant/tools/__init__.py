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
    "DisabledWebSearchClient",
    "TOOL_REGISTRY",
    "WebSearchClient",
    "WebSearchResult",
    "build_web_search_client",
]
