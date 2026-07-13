"""网页搜索工具：DuckDuckGo / Brave / Bing，供 ToolCallingChatService 可选启用。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse

import requests
from langchain_core.tools import InjectedToolCallId
from pydantic import BaseModel, Field, field_validator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from doc_assistant.config.settings import settings
from doc_assistant.schemas.citation import Citation
from doc_assistant.utils.text import optional_text

DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
_DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")


@dataclass(frozen=True)
class WebSearchResult:
    """搜索引擎返回的单条结果（尚未分配 [Wx] 编号，由 ToolCalling 层转换）。"""

    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class WebSource:
    """A web result after conversation-scoped source IDs are assigned."""

    source_id: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    recency_days: int | None = Field(default=None, ge=1, le=365)
    domains: list[str] = Field(default_factory=list, max_length=5)
    max_results: int | None = Field(default=None, ge=1, le=10)
    tool_call_id: Annotated[str, InjectedToolCallId] = ""

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("query is required")
        return value


class WebSearchClient:
    """网页搜索抽象基类；子类实现 DuckDuckGo / Brave / Bing。

    子类只需实现同步 ``search`` 入口。
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 10,
        max_retries: int = 3,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        retry = Retry(
            total=max(0, max_retries - 1),
            backoff_factor=1,
            status_forcelist=(429, *range(500, 600)),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        if default_headers:
            self.session.headers.update(default_headers)

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        raise NotImplementedError

    def _get(self, params: dict[str, object]) -> requests.Response:
        return self.session.get(
            self.base_url, params=params, headers={}, timeout=self.timeout_seconds
        )


class DisabledWebSearchClient(WebSearchClient):
    """占位客户端：未开启 DOC_ASSISTANT_WEB_SEARCH_ENABLED 时使用，调用即报错。"""

    def __init__(self) -> None:
        pass

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        raise RuntimeError("Web search is disabled. Set DOC_ASSISTANT_WEB_SEARCH_ENABLED=true.")


class DuckDuckGoSearchClient(WebSearchClient):
    """通过 DuckDuckGo HTML 接口抓取搜索结果，无需 API Key，适合本地开发。"""

    def __init__(
        self,
        base_url: str = DUCKDUCKGO_HTML_URL,
        timeout_seconds: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            default_headers={"User-Agent": "legal-doc-assistant/0.1"},
        )

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        search_query = _with_domain_filters(query, domains)
        params: dict[str, object] = {"q": search_query}
        freshness = _duckduckgo_recency_filter(recency_days)
        if freshness:
            params["df"] = freshness
        response = self._get(params)
        if response.status_code >= 400:
            raise RuntimeError(f"DuckDuckGo search failed: {response.status_code} {response.text}")
        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        return parser.results[:max_results]


class BraveSearchClient(WebSearchClient):
    """Brave Search API 客户端，需配置 BRAVE_SEARCH_API_KEY，支持时效与域名过滤。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = BRAVE_SEARCH_URL,
        timeout_seconds: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            default_headers={"X-Subscription-Token": api_key},
        )

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        search_query = _with_domain_filters(query, domains)
        params: dict[str, object] = {"q": search_query, "count": max_results}
        freshness = _brave_recency_filter(recency_days)
        if freshness:
            params["freshness"] = freshness
        response = self._get(params)
        if response.status_code >= 400:
            raise RuntimeError(f"Brave search failed: {response.status_code} {response.text}")
        data = response.json()
        results = data.get("web", {}).get("results", [])
        return [
            WebSearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("description") or ""),
                published_at=optional_text(item.get("age") or item.get("page_age")),
                source=_domain_from_url(str(item.get("url") or "")),
            )
            for item in results[:max_results]
            if item.get("url")
        ]


class BingSearchClient(WebSearchClient):
    """Microsoft Bing Web Search API v7 客户端，需配置 BING_SEARCH_API_KEY。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = BING_SEARCH_URL,
        timeout_seconds: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
        )

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        search_query = _with_domain_filters(query, domains)
        params: dict[str, object] = {"q": search_query, "count": max_results}
        freshness = _bing_recency_filter(recency_days)
        if freshness:
            params["freshness"] = freshness
        response = self._get(params)
        if response.status_code >= 400:
            raise RuntimeError(f"Bing search failed: {response.status_code} {response.text}")
        data = response.json()
        results = data.get("webPages", {}).get("value", [])
        return [
            WebSearchResult(
                title=str(item.get("name") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("snippet") or ""),
                published_at=optional_text(item.get("dateLastCrawled")),
                source=_domain_from_url(str(item.get("url") or "")),
            )
            for item in results[:max_results]
            if item.get("url")
        ]


# ---------------------------------------------------------------------------
# 工厂与辅助函数
# ---------------------------------------------------------------------------


def build_web_search_client() -> WebSearchClient:
    """根据 settings.web_search_provider 返回对应客户端；未启用时返回 DisabledWebSearchClient。"""
    if not settings.web_search_enabled:
        return DisabledWebSearchClient()

    provider = settings.web_search_provider.strip().lower()
    base_url = settings.web_search_base_url.strip()
    timeout = settings.web_search_timeout_seconds
    max_retries = settings.web_search_max_retries

    if provider == "duckduckgo":
        return DuckDuckGoSearchClient(base_url or DUCKDUCKGO_HTML_URL, timeout, max_retries)
    if provider == "brave":
        if not settings.web_search_api_key:
            raise ValueError("DOC_ASSISTANT_WEB_SEARCH_API_KEY is required for Brave search.")
        return BraveSearchClient(settings.web_search_api_key, base_url or BRAVE_SEARCH_URL, timeout, max_retries)
    if provider == "bing":
        if not settings.web_search_api_key:
            raise ValueError("DOC_ASSISTANT_WEB_SEARCH_API_KEY is required for Bing search.")
        return BingSearchClient(settings.web_search_api_key, base_url or BING_SEARCH_URL, timeout, max_retries)

    raise ValueError(f"Unsupported web search provider: {settings.web_search_provider}")


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


class _DuckDuckGoHTMLParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜索结果页，提取 title / url / snippet。"""

    def __init__(self) -> None:
        super().__init__()
        self._items: list[dict[str, str]] = []
        self._capture: str | None = None
        self._snippet_index: int | None = None

    @property
    def results(self) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source=_domain_from_url(item.get("url", "")),
            )
            for item in self._items
            if item.get("url")
        ]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = {key: value or "" for key, value in attrs}
        classes = set(attrs_by_name.get("class", "").split())

        if tag == "a" and "result__a" in classes:
            self._items.append(
                {"title": "", "url": _decode_duckduckgo_url(attrs_by_name.get("href", "")), "snippet": ""}
            )
            self._capture = "title"
            return

        if self._items and "result__snippet" in classes:
            self._snippet_index = len(self._items) - 1
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div"}:
            self._capture = None
            self._snippet_index = None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or not self._items:
            return

        if self._capture == "title":
            current = self._items[-1]
            current["title"] = (current.get("title", "") + " " + text).strip()
            return

        if self._capture == "snippet" and self._snippet_index is not None:
            current = self._items[self._snippet_index]
            current["snippet"] = (current.get("snippet", "") + " " + text).strip()


def _with_domain_filters(query: str, domains: list[str] | None) -> str:
    """把 site:domain 过滤附加到搜索 query（Bing/Brave 原生支持，DuckDuckGo 靠 query 语法）。"""
    clean_domains = [_clean_domain(domain) for domain in domains or [] if domain.strip()]
    if not clean_domains:
        return query
    if len(clean_domains) == 1:
        return f"{query} site:{clean_domains[0]}"
    filters = " OR ".join(f"site:{domain}" for domain in clean_domains)
    return f"{query} ({filters})"


def _clean_domain(domain: str) -> str:
    """规范化并校验域名过滤器，防止注入非法 host。"""
    candidate = domain.strip().lower()
    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    if parsed.netloc:
        candidate = parsed.netloc
    candidate = candidate.strip(".")
    if not candidate or not _DOMAIN_PATTERN.fullmatch(candidate) or ".." in candidate:
        raise ValueError(f"Invalid domain filter: {domain}")
    return candidate


def _duckduckgo_recency_filter(recency_days: int | None) -> str | None:
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "d"
    if recency_days <= 7:
        return "w"
    if recency_days <= 31:
        return "m"
    return "y"


def _brave_recency_filter(recency_days: int | None) -> str | None:
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "pd"
    if recency_days <= 7:
        return "pw"
    if recency_days <= 31:
        return "pm"
    return "py"


def _bing_recency_filter(recency_days: int | None) -> str | None:
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "Day"
    if recency_days <= 7:
        return "Week"
    return "Month"


def _decode_duckduckgo_url(value: str) -> str:
    """DuckDuckGo 结果链接常是 /l/?uddg= 跳转，解析出真实目标 URL。"""
    url = value.strip()
    if url.startswith("//"):
        url = "https:" + url

    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _domain_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    return parsed.netloc or None
