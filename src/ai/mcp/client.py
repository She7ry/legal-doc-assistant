"""Small helpers shared by the remote MCP host integration."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from mcp import ClientSession
from mcp.types import CallToolResult, Tool

MAX_TOOL_RESULT_CHARACTERS = 50_000

def langchain_tool_schema(tool: Tool) -> dict[str, Any]:
    """Convert one discovered MCP Tool into the schema accepted by LangChain models."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def _collect_pages(
    fetch_page: Callable[[str | None], Awaitable[Any]],
    item_field: str,
) -> tuple[Any, ...]:
    items: list[Any] = []
    cursor: str | None = None
    while True:
        page = await fetch_page(cursor)
        items.extend(getattr(page, item_field))
        cursor = page.nextCursor
        if not cursor:
            return tuple(items)


async def discover_tools(session: ClientSession) -> tuple[Tool, ...]:
    """Discover every tool page advertised by an initialized MCP Server."""
    advertised = session.get_server_capabilities()
    if advertised is None:
        raise RuntimeError("MCP ClientSession must be initialized before discovery.")
    tools = await _collect_pages(session.list_tools, "tools") if advertised.tools else ()
    return cast(tuple[Tool, ...], tools)


def tool_result_text(result: CallToolResult) -> str:
    """Serialize one MCP tool result for a Host model's ToolMessage."""
    payload: Any = result.structuredContent
    if payload is None:
        payload = [item.model_dump(mode="json", by_alias=True) for item in result.content]
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    if len(serialized) <= MAX_TOOL_RESULT_CHARACTERS:
        return serialized

    preview = serialized[:MAX_TOOL_RESULT_CHARACTERS]
    while True:
        truncated = json.dumps(
            {
                "truncated": True,
                "original_characters": len(serialized),
                "preview": preview,
            },
            ensure_ascii=False,
        )
        excess = len(truncated) - MAX_TOOL_RESULT_CHARACTERS
        if excess <= 0:
            return truncated
        preview = preview[:-excess]
