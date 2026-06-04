from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests

from doc_assistant.config.settings import settings

DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


class WebSearchClient:
    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        raise NotImplementedError


class DisabledWebSearchClient(WebSearchClient):
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
    def __init__(self, base_url: str = DUCKDUCKGO_HTML_URL, timeout_seconds: int = 10) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        search_query = _with_domain_filters(query, domains)
        response = requests.get(
            self.base_url,
            params={"q": search_query},
            headers={"User-Agent": "legal-doc-assistant/0.1"},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"DuckDuckGo search failed: {response.status_code} {response.text}")

        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        return parser.results[:max_results]


class BraveSearchClient(WebSearchClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = BRAVE_SEARCH_URL,
        timeout_seconds: int = 10,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

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
        response = requests.get(
            self.base_url,
            params=params,
            headers={"X-Subscription-Token": self.api_key},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Brave search failed: {response.status_code} {response.text}")

        data = response.json()
        results = data.get("web", {}).get("results", [])
        return [
            WebSearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("description") or ""),
                published_at=_string_or_none(item.get("age") or item.get("page_age")),
                source=_domain_from_url(str(item.get("url") or "")),
            )
            for item in results[:max_results]
            if item.get("url")
        ]


class BingSearchClient(WebSearchClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = BING_SEARCH_URL,
        timeout_seconds: int = 10,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        *,
        max_results: int,
        recency_days: int | None = None,
        domains: list[str] | None = None,
    ) -> list[WebSearchResult]:
        search_query = _with_domain_filters(query, domains)
        response = requests.get(
            self.base_url,
            params={"q": search_query, "count": max_results},
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Bing search failed: {response.status_code} {response.text}")

        data = response.json()
        results = data.get("webPages", {}).get("value", [])
        return [
            WebSearchResult(
                title=str(item.get("name") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("snippet") or ""),
                published_at=_string_or_none(item.get("dateLastCrawled")),
                source=_domain_from_url(str(item.get("url") or "")),
            )
            for item in results[:max_results]
            if item.get("url")
        ]


def build_web_search_client() -> WebSearchClient:
    if not settings.web_search_enabled:
        return DisabledWebSearchClient()

    provider = settings.web_search_provider.strip().lower()
    base_url = settings.web_search_base_url.strip()
    timeout = settings.web_search_timeout_seconds

    if provider == "duckduckgo":
        return DuckDuckGoSearchClient(base_url or DUCKDUCKGO_HTML_URL, timeout)
    if provider == "brave":
        if not settings.web_search_api_key:
            raise ValueError("DOC_ASSISTANT_WEB_SEARCH_API_KEY is required for Brave search.")
        return BraveSearchClient(settings.web_search_api_key, base_url or BRAVE_SEARCH_URL, timeout)
    if provider == "bing":
        if not settings.web_search_api_key:
            raise ValueError("DOC_ASSISTANT_WEB_SEARCH_API_KEY is required for Bing search.")
        return BingSearchClient(settings.web_search_api_key, base_url or BING_SEARCH_URL, timeout)

    raise ValueError(f"Unsupported web search provider: {settings.web_search_provider}")


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._capture: str | None = None
        self._snippet_index: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = {key: value or "" for key, value in attrs}
        classes = set(attrs_by_name.get("class", "").split())

        if tag == "a" and "result__a" in classes:
            self.results.append(
                WebSearchResult(
                    title="",
                    url=_decode_duckduckgo_url(attrs_by_name.get("href", "")),
                )
            )
            self._capture = "title"
            return

        if self.results and "result__snippet" in classes:
            self._snippet_index = len(self.results) - 1
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div"}:
            self._capture = None
            self._snippet_index = None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or not self.results:
            return

        if self._capture == "title":
            current = self.results[-1]
            self.results[-1] = WebSearchResult(
                title=(current.title + " " + text).strip(),
                url=current.url,
                snippet=current.snippet,
                published_at=current.published_at,
                source=_domain_from_url(current.url),
            )
            return

        if self._capture == "snippet" and self._snippet_index is not None:
            current = self.results[self._snippet_index]
            self.results[self._snippet_index] = WebSearchResult(
                title=current.title,
                url=current.url,
                snippet=(current.snippet + " " + text).strip(),
                published_at=current.published_at,
                source=current.source or _domain_from_url(current.url),
            )


def _with_domain_filters(query: str, domains: list[str] | None) -> str:
    clean_domains = [domain.strip() for domain in domains or [] if domain.strip()]
    if not clean_domains:
        return query
    if len(clean_domains) == 1:
        return f"{query} site:{clean_domains[0]}"
    filters = " OR ".join(f"site:{domain}" for domain in clean_domains)
    return f"{query} ({filters})"


def _decode_duckduckgo_url(value: str) -> str:
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


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
