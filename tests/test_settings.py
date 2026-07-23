from __future__ import annotations

import pytest

from ai.config.settings import Settings


def test_settings_reads_environment_per_instance(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_CHUNK_SIZE", "1200")

    assert Settings().chunk_size == 1200


def test_ocr_defaults_cover_simplified_chinese_and_english() -> None:
    configured = Settings()

    assert configured.pdf_ocr_enabled is False
    assert configured.pdf_ocr_lang == "chi_sim+eng"


def test_settings_with_overrides_returns_validated_copy() -> None:
    updated = Settings().with_overrides(top_k=3)

    assert updated.top_k == 3


def test_settings_rejects_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_tool_call_iterations_are_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS", "4")

    assert Settings().tool_call_max_iterations == 4


def test_chat_context_reserves_output_tokens() -> None:
    configured = Settings(chat_context_max_tokens=10_000, chat_max_output_tokens=2_000)

    assert configured.chat_input_max_tokens == 8_000

    with pytest.raises(ValueError, match="smaller"):
        Settings(chat_context_max_tokens=2_000, chat_max_output_tokens=2_000)


def test_docusign_mcp_settings_are_read_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_MCP_ENABLED", "true")
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_CLIENT_ID", "client-id")
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_TOKEN_PATH", str(tmp_path / "token.json"))

    configured = Settings()

    assert configured.docusign_mcp_enabled is True
    assert configured.docusign_client_id == "client-id"
    assert configured.docusign_token_path == tmp_path / "token.json"


def test_docusign_mcp_requires_oauth_client(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_MCP_ENABLED", "true")
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_CLIENT_ID", "")
    monkeypatch.setenv("DOC_ASSISTANT_DOCUSIGN_CLIENT_SECRET", "")

    with pytest.raises(ValueError, match="Docusign MCP"):
        Settings()


def test_memory_top_k_is_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOC_ASSISTANT_MEMORY_TOP_K", "2")

    assert Settings().memory_top_k == 2


def test_langsmith_settings_use_official_environment_names(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "agent-test")
    monkeypatch.setenv("LANGSMITH_TRACING_SAMPLING_RATE", "0.25")

    configured = Settings()

    assert configured.langsmith_tracing is True
    assert configured.langsmith_api_key == "lsv2_test"
    assert configured.langsmith_project == "agent-test"
    assert configured.langsmith_tracing_sampling_rate == 0.25
