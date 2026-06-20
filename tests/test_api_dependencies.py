from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import dependencies
from api.main import app
from doc_assistant.memory.service import MemoryService
from doc_assistant.memory.store import MemoryStore
from doc_assistant.retrieval import vector_store


def test_normalize_tenant_id_defaults_to_configured_tenant(monkeypatch) -> None:
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(default_tenant_id="default"))

    assert dependencies.normalize_tenant_id(None) == "default"


def test_normalize_tenant_id_rejects_unsafe_values(monkeypatch) -> None:
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(default_tenant_id="default"))

    with pytest.raises(ValueError):
        dependencies.normalize_tenant_id("../other")


def test_normalize_user_id_defaults_and_rejects_unsafe_values() -> None:
    assert dependencies.normalize_user_id(None) == "local-user"

    with pytest.raises(ValueError):
        dependencies.normalize_user_id("../other")


def test_require_api_key_accepts_configured_key(monkeypatch) -> None:
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(api_keys=("secret",)))

    dependencies.require_api_key(x_api_key="secret", credentials=None)


def test_require_api_key_rejects_bad_key(monkeypatch) -> None:
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(api_keys=("secret",)))

    with pytest.raises(HTTPException) as exc_info:
        dependencies.require_api_key(x_api_key="wrong", credentials=None)

    assert exc_info.value.status_code == 401


def test_protected_routes_require_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(dependencies, "settings", SimpleNamespace(api_keys=("secret",)))
    client = TestClient(app)

    response = client.get("/api/v1/documents")

    assert response.status_code == 401
    assert response.json()["code"] == "http_401"


def test_document_text_endpoint_returns_indexed_chunks() -> None:
    class FakeVectorStore:
        def get_document_text(self, *, document_key=None, file_id=None, document_version=None):
            assert document_key == "doc-key"
            assert file_id is None
            assert document_version is None
            return {
                "document": {
                    "file_name": "contract.pdf",
                    "file_id": "file-a",
                    "document_key": "doc-key",
                    "document_version": 2,
                    "file_extension": ".pdf",
                    "document_count": 1,
                    "chunk_count": 1,
                    "page_count": 1,
                    "indexed_at": "2026-06-17T00:00:00+00:00",
                    "warning_count": 0,
                },
                "chunks": [
                    {
                        "chunk_id": 0,
                        "text": "Payment is due within 30 days.",
                        "page": 0,
                        "page_label": "page 1",
                        "section_heading": "2. Payment",
                        "location_label": "page 1, chunk 0, 2. Payment",
                    }
                ],
                "total_chunks": 1,
            }

    app.dependency_overrides[dependencies.get_vector_store] = lambda: FakeVectorStore()
    try:
        client = TestClient(app)
        response = client.get("/api/v1/documents/text", params={"document_key": "doc-key"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_key"] == "doc-key"
    assert data["chunks"][0]["text"] == "Payment is due within 30 days."


def test_health_returns_runtime_diagnostics_and_request_id() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={"X-Request-Id": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "test-request-id"
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert data["version"] == app.version
    assert isinstance(data["auth_required"], bool)
    assert data["providers"]["chat"]["api_key_configured"] in {True, False}
    assert data["providers"]["embedding"]["api_key_configured"] in {True, False}
    assert ".pdf" in data["limits"]["supported_extensions"]
    assert {check["name"] for check in data["checks"]} >= {
        "uploads",
        "vector_store",
        "ingest_jobs",
        "memory_store",
        "chat_api_key",
        "embedding_api_key",
    }


def test_chat_conversation_messages_endpoint_restores_history(tmp_path) -> None:
    memory_service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)
    conversation_id = memory_service.ensure_context("tenant-a", "user-a", "conversation-a")
    memory_service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier question.",
    )
    memory_service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier answer.",
    )
    app.dependency_overrides[dependencies.get_memory_service] = lambda: memory_service
    app.dependency_overrides[dependencies.get_tenant_id] = lambda: "tenant-a"
    app.dependency_overrides[dependencies.get_user_id] = lambda: "user-a"
    try:
        client = TestClient(app)
        response = client.get("/api/v1/chat/conversations/conversation-a/messages")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "conversation-a",
        "messages": [
            {"role": "user", "content": "Earlier question."},
            {"role": "assistant", "content": "Earlier answer."},
        ],
    }


def test_chat_conversation_management_endpoints(tmp_path) -> None:
    memory_service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)
    app.dependency_overrides[dependencies.get_memory_service] = lambda: memory_service
    app.dependency_overrides[dependencies.get_tenant_id] = lambda: "tenant-a"
    app.dependency_overrides[dependencies.get_user_id] = lambda: "user-a"
    try:
        client = TestClient(app)
        created = client.post(
            "/api/v1/chat/conversations",
            json={"conversation_id": "conversation-a", "title": "Lease review"},
        )
        memory_service.record_user_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id="conversation-a",
            content="Earlier question.",
        )
        memory_service.record_assistant_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id="conversation-a",
            content="Earlier answer.",
        )
        listed = client.get("/api/v1/chat/conversations")
        archived = client.patch(
            "/api/v1/chat/conversations/conversation-a",
            json={"status": "archived", "title": "Archived lease review"},
        )
        active_after_archive = client.get("/api/v1/chat/conversations")
        archived_list = client.get("/api/v1/chat/conversations", params={"status": "archived"})
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 201
    assert created.json()["conversation_id"] == "conversation-a"
    assert created.json()["title"] == "Lease review"

    assert listed.status_code == 200
    listed_data = listed.json()
    assert listed_data["total"] == 1
    assert listed_data["conversations"][0]["conversation_id"] == "conversation-a"
    assert listed_data["conversations"][0]["message_count"] == 2

    assert archived.status_code == 200
    assert archived.json()["status"] == "archived"
    assert archived.json()["title"] == "Archived lease review"

    assert active_after_archive.status_code == 200
    assert active_after_archive.json()["total"] == 0
    assert archived_list.status_code == 200
    assert archived_list.json()["total"] == 1
    assert archived_list.json()["conversations"][0]["status"] == "archived"


def test_memory_stats_endpoint_returns_user_stats(tmp_path) -> None:
    memory_service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)
    memory_service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="client_context",
        content="Acme is the client.",
    )
    app.dependency_overrides[dependencies.get_memory_service] = lambda: memory_service
    app.dependency_overrides[dependencies.get_tenant_id] = lambda: "tenant-a"
    app.dependency_overrides[dependencies.get_user_id] = lambda: "user-a"
    try:
        client = TestClient(app)
        response = client.get("/api/v1/memories/stats")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "tenant-a"
    assert data["user_id"] == "user-a"
    assert data["total_memories"] == 1
    assert data["active_memories"] == 1
    assert data["retrievals"]["total"] == 0


def test_feedback_endpoint_records_feedback_and_adjusts_memory(tmp_path) -> None:
    memory_service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)
    memory = memory_service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="client_context",
        content="Acme is the client.",
        confidence=0.8,
    )
    app.dependency_overrides[dependencies.get_memory_service] = lambda: memory_service
    app.dependency_overrides[dependencies.get_tenant_id] = lambda: "tenant-a"
    app.dependency_overrides[dependencies.get_user_id] = lambda: "user-a"
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/feedback",
            json={
                "rating": "positive",
                "memory_ids": [memory.memory_id],
                "comment": "This helped.",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert data["rating"] == 1
    assert data["memory_ids"] == [memory.memory_id]
    assert data["adjusted_memories"][0]["previous_confidence"] == 0.8
    assert data["adjusted_memories"][0]["new_confidence"] == 0.83
    refreshed = memory_service.store.get_memory("tenant-a", "user-a", memory.memory_id)
    assert refreshed is not None
    assert refreshed.confidence == 0.83


def test_collection_name_for_tenant_preserves_default_collection(monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "settings", SimpleNamespace(default_tenant_id="default"))

    assert vector_store.collection_name_for_tenant("legal_documents", "default") == "legal_documents"
    assert vector_store.collection_name_for_tenant("legal_documents", "acme") == "legal_documents_acme"
