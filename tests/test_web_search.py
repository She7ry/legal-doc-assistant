from __future__ import annotations

import pytest

from ai.agent.tools.web_search import BRAVE_SEARCH_URL, BraveSearchClient


class FakeResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict:
        return {
            "web": {
                "results": [
                    {
                        "title": "Example title",
                        "url": "https://example.com/news",
                        "description": "Example snippet",
                        "age": "2 days ago",
                    }
                ]
            }
        }


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url, *, params, timeout):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
            }
        )
        return FakeResponse()


def test_brave_search_maps_results_and_filters() -> None:
    session = FakeSession()
    client = BraveSearchClient("test-key", timeout_seconds=5, max_retries=1)
    client.session = session

    results = client.search(
        "supplier news",
        max_results=1,
        recency_days=7,
        domains=["https://Example.com/path"],
    )

    assert session.calls[0]["url"] == BRAVE_SEARCH_URL
    assert session.calls[0]["params"]["freshness"] == "pw"
    assert session.calls[0]["params"]["q"] == "supplier news site:example.com"
    assert results[0].title == "Example title"
    assert results[0].snippet == "Example snippet"
    assert results[0].source == "example.com"


def test_web_search_rejects_domain_operator_injection() -> None:
    client = BraveSearchClient("test-key", max_retries=1)

    with pytest.raises(ValueError):
        client.search(
            "supplier news",
            max_results=1,
            domains=["example.com -site:competitor.com"],
        )


def test_web_search_uses_requests_retry_adapter() -> None:
    client = BraveSearchClient("test-key", max_retries=3)
    retry = client.session.get_adapter("https://").max_retries

    assert retry.total == 2
    assert 429 in retry.status_forcelist
    assert 500 in retry.status_forcelist
