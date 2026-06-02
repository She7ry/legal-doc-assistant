from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api import dependencies
from api.main import app
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


def test_collection_name_for_tenant_preserves_default_collection(monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "settings", SimpleNamespace(default_tenant_id="default"))

    assert vector_store.collection_name_for_tenant("legal_documents", "default") == "legal_documents"
    assert vector_store.collection_name_for_tenant("legal_documents", "acme") == "legal_documents_acme"
