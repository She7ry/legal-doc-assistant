from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult

from doc_assistant.mcp.client import (
    MAX_TOOL_RESULT_CHARACTERS,
    discover_capabilities,
    open_stdio_session,
    tool_result_text,
)
from doc_assistant.mcp.github_demo import github_server_parameters, run_github_agent

fake_github = FastMCP("Fake GitHub MCP Server")
search_calls: list[str] = []


@fake_github.tool()
def get_me() -> dict[str, str]:
    return {"login": "octocat"}


@fake_github.tool()
def search_repositories(query: str) -> dict[str, Any]:
    search_calls.append(query)
    return {"repositories": [{"full_name": "octocat/Hello-World"}]}


@fake_github.tool()
def get_file_contents(owner: str, repo: str, path: str) -> dict[str, str]:
    return {"owner": owner, "repo": repo, "path": path, "content": "hello"}


@fake_github.tool()
def delete_repository(owner: str, repo: str) -> dict[str, str]:
    return {"owner": owner, "repo": repo, "status": "deleted"}


@fake_github.resource("github://fixture/readme")
def fixture_readme() -> str:
    return "fixture readme"


@fake_github.resource("github://repos/{owner}/{repo}")
def fixture_repository(owner: str, repo: str) -> str:
    return f"{owner}/{repo}"


@fake_github.prompt()
def summarize_repository(repo: str) -> str:
    return f"Summarize {repo}"


class FakeHostModel:
    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = []
        self.requests: list[list[Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> FakeHostModel:
        self.tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_repositories",
                        "args": {"query": "user:octocat"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        assert isinstance(messages[-1], ToolMessage)
        assert "octocat/Hello-World" in str(messages[-1].content)
        return AIMessage(content="找到 octocat/Hello-World。")


def test_github_server_parameters_are_read_only_and_secret_scoped() -> None:
    parameters = github_server_parameters(
        {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "github-token",
            "UNRELATED_SECRET": "must-not-leak",
        }
    )

    assert parameters.command == "docker"
    assert parameters.args[-1] == "ghcr.io/github/github-mcp-server"
    assert "GITHUB_TOOLS" in parameters.args
    assert "GITHUB_TOOLSETS" not in parameters.args
    assert parameters.env == {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "github-token",
        "GITHUB_TOOLS": "get_me,get_file_contents,search_repositories",
        "GITHUB_READ_ONLY": "1",
    }
    assert "github-token" not in parameters.args
    assert "UNRELATED_SECRET" not in parameters.env


def test_github_server_parameters_support_explicit_local_binary() -> None:
    parameters = github_server_parameters(
        {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "github-token",
            "GITHUB_MCP_COMMAND": "github-mcp-server",
            "GITHUB_MCP_ARGS": "stdio",
        }
    )

    assert parameters.command == "github-mcp-server"
    assert parameters.args == ["stdio"]
    assert parameters.env == {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "github-token",
        "GITHUB_TOOLS": "get_me,get_file_contents,search_repositories",
        "GITHUB_READ_ONLY": "1",
    }


def test_github_server_parameters_require_token() -> None:
    with pytest.raises(ValueError, match="GITHUB_PERSONAL_ACCESS_TOKEN"):
        github_server_parameters({})


@pytest.mark.asyncio
async def test_host_agent_discovers_allowlisted_tools_and_calls_mcp() -> None:
    search_calls.clear()
    model = FakeHostModel()

    async with create_connected_server_and_client_session(
        fake_github,
        raise_exceptions=True,
    ) as session:
        answer = await run_github_agent(session, "搜索 octocat 的仓库", model=model)

    assert answer == "找到 octocat/Hello-World。"
    assert [tool["function"]["name"] for tool in model.tools] == [
        "get_me",
        "get_file_contents",
        "search_repositories",
    ]
    assert search_calls == ["user:octocat"]


@pytest.mark.asyncio
async def test_stdio_client_crosses_a_real_json_rpc_process_boundary() -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--fake-server"],
        cwd=Path(__file__).resolve().parents[1],
    )

    async with open_stdio_session(parameters) as session:
        capabilities = await discover_capabilities(session)
        result = await session.call_tool("search_repositories", {"query": "user:octocat"})

    assert {tool.name for tool in capabilities.tools} >= {
        "search_repositories",
        "delete_repository",
    }
    assert {str(resource.uri) for resource in capabilities.resources} == {
        "github://fixture/readme"
    }
    assert {template.uriTemplate for template in capabilities.resource_templates} == {
        "github://repos/{owner}/{repo}"
    }
    assert {prompt.name for prompt in capabilities.prompts} == {"summarize_repository"}
    assert not result.isError
    assert result.structuredContent == {
        "repositories": [{"full_name": "octocat/Hello-World"}]
    }


def test_tool_result_truncation_stays_bounded_valid_json() -> None:
    result = CallToolResult(
        content=[],
        structuredContent={"content": "x" * (MAX_TOOL_RESULT_CHARACTERS * 2)},
    )

    serialized = tool_result_text(result)
    payload = json.loads(serialized)

    assert len(serialized) <= MAX_TOOL_RESULT_CHARACTERS
    assert payload["truncated"] is True
    assert payload["original_characters"] > MAX_TOOL_RESULT_CHARACTERS


if __name__ == "__main__" and "--fake-server" in sys.argv:
    fake_github.run(transport="stdio")
