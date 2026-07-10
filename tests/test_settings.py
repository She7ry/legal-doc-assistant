from __future__ import annotations

import pytest

from doc_assistant.config.settings import RetrievalSettings, Settings


def test_settings_reads_environment_per_instance(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_CHUNK_SIZE", "1200")

    assert Settings().chunk_size == 1200


def test_settings_with_overrides_returns_validated_copy() -> None:
    updated = Settings().with_overrides(top_k=3)

    assert updated.top_k == 3


def test_settings_rejects_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(retrieval=RetrievalSettings(chunk_size=100, chunk_overlap=100))


def test_tool_call_iterations_are_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS", "4")

    assert Settings().tool_call_max_iterations == 4


def test_memory_top_k_is_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_MEMORY_TOP_K", "2")

    assert Settings().memory_top_k == 2
