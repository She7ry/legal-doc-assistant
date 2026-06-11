from __future__ import annotations

import pytest

from doc_assistant.config.settings import Settings


def test_settings_reads_environment_per_instance(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_CHUNK_SIZE", "1200")

    assert Settings().chunk_size == 1200


def test_settings_with_overrides_returns_validated_copy() -> None:
    updated = Settings().with_overrides(top_k=3)

    assert updated.top_k == 3


def test_settings_rejects_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_agent_backoff_is_parsed_as_numbers(monkeypatch) -> None:
    monkeypatch.delenv("DOC_ASSISTANT_AGENT_STEP_RETRY_BACKOFF_SECONDS", raising=False)

    assert Settings().agent_step_retry_backoff_seconds == (2.0, 5.0)
