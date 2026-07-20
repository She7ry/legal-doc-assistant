"""Brave 网页搜索工具，供 ToolCallingChatService 可选启用。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlparse

import requests
from langchain_core.tools import InjectedToolCallId
from pydantic import BaseModel, Field, field_validator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai.config.settings import settings
from ai.rag.schemas import Citation
from ai.utils.text import optional_text

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")


@dataclass(frozen=True)
class WebSearchResult:
    """尚未分配引用编号的 Brave 搜索结果。"""

    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class WebSource:
    """已分配会话级引用编号的网页来源。"""

    source_id: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


class WebSearchInput(BaseModel):
    query: str = Field(
        description="不得包含保密信息的公开网页查询。",
        min_length=1,
        max_length=300,
    )
    recency_days: int | None = Field(
        default=None,
        description="仅检索最近若干天的结果。",
        ge=1,
        le=365,
    )
    domains: list[str] = Field(
        default_factory=list,
        description="最多五个限定域名。",
        max_length=5,
    )
    max_results: int | None = Field(
        default=None,
        description="最多返回的网页结果数。",
        ge=1,
        le=10,
    )
    tool_call_id: Annotated[str, InjectedToolCallId] = ""

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("query is required")
        return value


class BraveSearchClient:
    """返回结构化网页结果的 Brave Search API 客户端。"""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 10,
        max_retries: int = 3,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"X-Subscription-Token": api_key})
        retry = Retry(
            total=max(0, max_retries - 1),
            backoff_factor=1,
            status_forcelist=(429, *range(500, 600)),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=10)
        self.session.mount("https://", adapter)

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        params: dict[str, object] = {
            "q": _with_domain_filters(query, domains),
            "count": max_results,
        }
        if freshness := _recency_filter(recency_days):
            params["freshness"] = freshness

        response = self.session.get(
            BRAVE_SEARCH_URL,
            params=params,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Brave search failed: {response.status_code} {response.text}"
            )

        results = response.json().get("web", {}).get("results", [])
        return [
            WebSearchResult(
                title=str(item.get("title") or ""),
                url=str(item["url"]),
                snippet=str(item.get("description") or ""),
                published_at=optional_text(item.get("age") or item.get("page_age")),
                source=urlparse(str(item["url"])).netloc or None,
            )
            for item in results[:max_results]
            if item.get("url")
        ]


def build_web_search_client() -> BraveSearchClient:
    if not settings.web_search_api_key:
        raise ValueError("DOC_ASSISTANT_WEB_SEARCH_API_KEY is required for Brave search.")
    return BraveSearchClient(
        settings.web_search_api_key,
        settings.web_search_timeout_seconds,
        settings.web_search_max_retries,
    )


def web_source(source_id: str, result: WebSearchResult) -> WebSource:
    return WebSource(
        source_id=source_id,
        title=result.title,
        url=result.url,
        snippet=result.snippet,
        published_at=result.published_at,
        source=result.source,
    )


def web_source_citations(sources: list[WebSource]) -> list[Citation]:
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


def _with_domain_filters(query: str, domains: list[str] | None) -> str:
    clean_domains = [_clean_domain(domain) for domain in domains or [] if domain.strip()]
    if not clean_domains:
        return query
    if len(clean_domains) == 1:
        return f"{query} site:{clean_domains[0]}"
    filters = " OR ".join(f"site:{domain}" for domain in clean_domains)
    return f"{query} ({filters})"


def _clean_domain(domain: str) -> str:
    """规范化并校验域名过滤器，防止注入其他查询操作符。"""
    candidate = domain.strip().lower()
    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    if parsed.netloc:
        candidate = parsed.netloc
    candidate = candidate.strip(".")
    if not candidate or not _DOMAIN_PATTERN.fullmatch(candidate) or ".." in candidate:
        raise ValueError(f"Invalid domain filter: {domain}")
    return candidate


def _recency_filter(recency_days: int | None) -> str | None:
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "pd"
    if recency_days <= 7:
        return "pw"
    if recency_days <= 31:
        return "pm"
    return "py"
