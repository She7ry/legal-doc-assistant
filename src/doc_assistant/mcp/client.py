"""Small MCP client adapter shared by Host-side demos."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Prompt, Resource, ResourceTemplate, Tool

MAX_TOOL_RESULT_CHARACTERS = 50_000

@dataclass(frozen=True)
class DiscoveredCapabilities:
    """Capabilities advertised by one initialized MCP Server."""

    tools: tuple[Tool, ...] = ()
    resources: tuple[Resource, ...] = ()
    resource_templates: tuple[ResourceTemplate, ...] = ()
    prompts: tuple[Prompt, ...] = ()


@asynccontextmanager
async def open_stdio_session(
    parameters: StdioServerParameters,
) -> AsyncIterator[ClientSession]:
    """Start an independent stdio Server and yield an initialized MCP Client session."""
    async with (
        stdio_client(parameters) as streams,
        ClientSession(*streams) as session,
    ):
        await session.initialize()
        yield session


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


async def discover_capabilities(session: ClientSession) -> DiscoveredCapabilities:
    """Discover every advertised Tools, Resources, and Prompts page."""
    advertised = session.get_server_capabilities()
    if advertised is None:
        raise RuntimeError("MCP ClientSession must be initialized before discovery.")

    tools = await _collect_pages(session.list_tools, "tools") if advertised.tools else ()
    resources = (
        await _collect_pages(session.list_resources, "resources")
        if advertised.resources
        else ()
    )
    resource_templates = (
        await _collect_pages(session.list_resource_templates, "resourceTemplates")
        if advertised.resources
        else ()
    )
    prompts = (
        await _collect_pages(session.list_prompts, "prompts") if advertised.prompts else ()
    )
    return DiscoveredCapabilities(
        tools=cast(tuple[Tool, ...], tools),
        resources=cast(tuple[Resource, ...], resources),
        resource_templates=cast(tuple[ResourceTemplate, ...], resource_templates),
        prompts=cast(tuple[Prompt, ...], prompts),
    )


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
