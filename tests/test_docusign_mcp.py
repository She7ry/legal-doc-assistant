from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations
from pydantic import Field

import ai.agent.tool_calling as tool_service_module
from ai.agent.tool_calling import ToolCallingChatService
from ai.mcp.docusign import (
    DOCUSIGN_MCP_URL,
    DOCUSIGN_SCOPES,
    _access_token,
    _authorization_url,
    _load_token,
    _save_token,
    _user_token_path,
)
from ai.rag.qa_service import DocumentQAService

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
fake_docusign = FastMCP("Fake Docusign")


@fake_docusign.tool(annotations=_READ_ONLY)
def getUserInfo() -> dict[str, object]:
    return {"accounts": [{"accountId": "account-1"}]}


@fake_docusign.tool(annotations=_READ_ONLY)
def getAllAgreements(accountId: str) -> dict[str, object]:
    return {
        "accountId": accountId,
        "agreements": [{"id": "agreement-1", "title": "供应商合同"}],
    }


@fake_docusign.tool(annotations=_READ_ONLY)
def getAgreementDetails(accountId: str, agreementId: str) -> dict[str, str]:
    return {"accountId": accountId, "agreementId": agreementId}


@fake_docusign.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def triggerWorkflow(accountId: str) -> dict[str, str]:
    return {"accountId": accountId, "status": "triggered"}


class EmptyVectorStore:
    user_id = "default"

    def search(self, query: str, k: int | None = None):
        del query, k
        return []


class DocusignToolModel(BaseChatModel):
    calls: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "docusign-mcp-test"

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self.bound_tool_names = [
            item["function"]["name"] if isinstance(item, dict) else item.name for item in tools
        ]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        raise AssertionError("The enabled MCP path must use async model invocation.")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self.calls += 1
        if self.calls == 1:
            assert "triggerWorkflow" not in self.bound_tool_names
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "getAllAgreements",
                        "args": {"accountId": "account-1"},
                        "id": "call-docusign",
                    }
                ],
            )
        else:
            assert isinstance(messages[-1], ToolMessage)
            assert '"source_id": "D1"' in str(messages[-1].content)
            message = AIMessage(content="找到供应商合同 [D1]。")
        return ChatResult(generations=[ChatGeneration(message=message)])


class NativeFallbackModel(DocusignToolModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Docusign 暂不可用。"))]
        )


def _enable_docusign(monkeypatch, tmp_path: Path) -> None:
    configured = tool_service_module.settings.with_overrides(
        docusign_mcp_enabled=True,
        docusign_client_id="client-id",
        docusign_client_secret="client-secret",
        docusign_token_path=tmp_path / "token.json",
    )
    monkeypatch.setattr(tool_service_module, "settings", configured)


def test_docusign_skill_body_enters_system_prompt_when_enabled(monkeypatch, tmp_path: Path) -> None:
    _enable_docusign(monkeypatch, tmp_path)
    service = ToolCallingChatService(
        DocumentQAService(vector_store=EmptyVectorStore(), chat_model=NativeFallbackModel())
    )
    body = tool_service_module.DOCUSIGN_SKILL_PATH.read_text(encoding="utf-8").split("---", 2)[2]

    system_prompt = service._initial_messages("查询协议。", [])[0]["content"]

    assert body.strip() in system_prompt
    assert "name: review-docusign-agreements" not in system_prompt


def test_docusign_skill_is_not_loaded_when_mcp_is_disabled(monkeypatch) -> None:
    configured = tool_service_module.settings.with_overrides(docusign_mcp_enabled=False)
    monkeypatch.setattr(tool_service_module, "settings", configured)
    service = ToolCallingChatService(
        DocumentQAService(vector_store=EmptyVectorStore(), chat_model=NativeFallbackModel())
    )

    system_prompt = service._initial_messages("查询协议。", [])[0]["content"]

    assert "review-docusign-agreements" not in system_prompt


def test_docusign_skill_load_failure_warns_and_continues(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    _enable_docusign(monkeypatch, tmp_path)
    monkeypatch.setattr(tool_service_module, "DOCUSIGN_SKILL_PATH", tmp_path / "missing.md")
    service = ToolCallingChatService(
        DocumentQAService(vector_store=EmptyVectorStore(), chat_model=NativeFallbackModel())
    )

    system_prompt = service._initial_messages("查询协议。", [])[0]["content"]

    assert "trusted_project_skill" not in system_prompt
    assert "Docusign skill could not be loaded" in caplog.text


def test_docusign_skill_metadata_matches_runtime_dependency() -> None:
    skill_dir = tool_service_module.DOCUSIGN_SKILL_PATH.parent
    skill_text = tool_service_module.DOCUSIGN_SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = skill_text.split("---", 2)[1].strip().splitlines()
    metadata = (skill_dir / "agents" / "openai.yaml").read_text(encoding="utf-8")
    short_description = next(
        line.split(":", 1)[1].strip().strip('"')
        for line in metadata.splitlines()
        if line.strip().startswith("short_description:")
    )

    assert [line.split(":", 1)[0] for line in frontmatter] == ["name", "description"]
    assert 25 <= len(short_description) <= 64
    assert "$review-docusign-agreements" in metadata
    assert 'type: "mcp"' in metadata
    assert 'value: "docusign"' in metadata
    assert 'transport: "streamable_http"' in metadata
    assert 'url: "https://mcp-d.docusign.com/mcp"' in metadata


def test_tool_calling_service_uses_only_read_only_docusign_tools(monkeypatch, tmp_path: Path) -> None:
    _enable_docusign(monkeypatch, tmp_path)

    @asynccontextmanager
    async def open_fake_session(*args, **kwargs):
        del args, kwargs
        async with create_connected_server_and_client_session(
            fake_docusign,
            raise_exceptions=True,
        ) as session:
            yield session

    monkeypatch.setattr(tool_service_module, "open_docusign_session", open_fake_session)
    model = DocusignToolModel()
    service = ToolCallingChatService(
        DocumentQAService(vector_store=EmptyVectorStore(), chat_model=model)
    )

    answer = service.ask("查询供应商合同。")

    assert answer.content == "找到供应商合同 [D1]。"
    assert answer.citations[0].source_type == "docusign"
    assert answer.tool_calls[0].name == "getAllAgreements"
    assert answer.tool_calls[0].result["source_id"] == "D1"


def test_tool_calling_service_falls_back_when_docusign_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _enable_docusign(monkeypatch, tmp_path)

    @asynccontextmanager
    async def unavailable(*args, **kwargs):
        del args, kwargs
        raise OSError("unavailable")
        yield

    monkeypatch.setattr(tool_service_module, "open_docusign_session", unavailable)
    service = ToolCallingChatService(
        DocumentQAService(vector_store=EmptyVectorStore(), chat_model=NativeFallbackModel())
    )

    answer = service.ask("继续使用本地工具。")

    assert answer.content == "Docusign 暂不可用。"
    assert answer.tool_calls == []


@pytest.mark.asyncio
async def test_expired_docusign_token_is_refreshed(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    _save_token(
        token_path,
        {
            "access_token": "expired",
            "refresh_token": "refresh-1",
            "expires_at": 0,
            "client_id": "client-id",
        },
    )

    def refresh(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        assert request.headers["Authorization"].startswith("Basic ")
        return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})

    async with httpx.AsyncClient(transport=httpx.MockTransport(refresh)) as client:
        token = await _access_token(
            client,
            client_id="client-id",
            client_secret="client-secret",
            token_path=token_path,
        )

    assert token == "fresh"
    assert _load_token(token_path)["refresh_token"] == "refresh-1"


def test_docusign_authorization_url_uses_pkce_and_read_only_scopes() -> None:
    query = parse_qs(urlparse(_authorization_url("client-id", "state", "verifier")).query)

    assert query["scope"] == [DOCUSIGN_SCOPES]
    assert query["resource"] == [DOCUSIGN_MCP_URL]
    assert query["code_challenge_method"] == ["S256"]


def test_docusign_tokens_are_isolated_by_app_user(tmp_path: Path) -> None:
    base = tmp_path / "token.json"

    assert _user_token_path(base, "user-a") != _user_token_path(base, "user-b")

    with pytest.raises(ValueError, match="user_id"):
        _user_token_path(base, " ")
