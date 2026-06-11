from __future__ import annotations

import pytest

from doc_assistant.tools.web_search import DuckDuckGoSearchClient


class FakeResponse:
    status_code = 200
    text = """
    <html>
      <a class="result__a" href="https://example.com/news">Example title</a>
      <div class="result__snippet">Example snippet</div>
    </html>
    """


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse()


def test_duckduckgo_search_uses_recency_and_sanitized_domain_filters() -> None:
    session = FakeSession()
    client = DuckDuckGoSearchClient(timeout_seconds=5, max_retries=1)
    client.session = session

    results = client.search(
        "supplier news",
        max_results=1,
        recency_days=7,
        domains=["https://Example.com/path"],
    )

    assert session.calls[0]["params"]["df"] == "w"
    assert session.calls[0]["params"]["q"] == "supplier news site:example.com"
    assert results[0].title == "Example title"
    assert results[0].snippet == "Example snippet"


def test_web_search_rejects_domain_operator_injection() -> None:
    client = DuckDuckGoSearchClient(max_retries=1)

    with pytest.raises(ValueError):
        client.search(
            "supplier news",
            max_results=1,
            domains=["example.com -site:competitor.com"],
        )
