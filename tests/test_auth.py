from __future__ import annotations

import sqlite3
from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from backend import dependencies
from backend.auth_store import AuthStore, UsernameAlreadyExistsError, UserRecord
from backend.main import app

pytestmark = pytest.mark.real_auth


def test_auth_store_hashes_passwords_and_revokes_sessions(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    store = AuthStore(db_path)
    user = store.register("alice", "correct-horse")

    with sqlite3.connect(db_path) as connection:
        stored_hash = connection.execute("SELECT password_hash FROM users").fetchone()[0]
    assert "correct-horse" not in stored_hash
    assert store.authenticate("ALICE", "correct-horse") == user
    assert store.authenticate("alice", "wrong-password") is None
    with pytest.raises(UsernameAlreadyExistsError):
        store.register("Alice", "another-password")

    token = store.create_session(user.user_id)
    assert store.resolve_session(token) == user
    store.revoke_session(token)
    assert store.resolve_session(token) is None


def test_registered_users_get_separate_document_stores(tmp_path) -> None:
    store = AuthStore(tmp_path / "auth.sqlite3")

    class PersonalVectorStore:
        def __init__(self, user_id: str) -> None:
            self.user_id = user_id

        def list_documents(self):
            return [
                {
                    "file_name": f"{self.user_id}.pdf",
                    "file_id": self.user_id,
                    "document_key": self.user_id,
                }
            ]

    def vector_store_for_user(
        current_user: Annotated[UserRecord, Depends(dependencies.get_current_user)],
    ) -> PersonalVectorStore:
        return PersonalVectorStore(current_user.user_id)

    app.dependency_overrides[dependencies.get_auth_store] = lambda: store
    app.dependency_overrides[dependencies.get_vector_store] = vector_store_for_user
    alice = TestClient(app)
    bob = TestClient(app)
    try:
        alice_registration = alice.post(
            "/api/v1/auth/register",
            json={"username": "alice", "password": "correct-horse"},
        )
        bob_registration = bob.post(
            "/api/v1/auth/register",
            json={"username": "bob", "password": "correct-horse"},
        )
        assert alice_registration.status_code == 201
        assert bob_registration.status_code == 201
        assert "httponly" in alice_registration.headers["set-cookie"].lower()

        alice_user_id = alice_registration.json()["user_id"]
        bob_user_id = bob_registration.json()["user_id"]
        alice_documents = alice.get(
            "/api/v1/documents",
            headers={"X-User-Id": bob_user_id},
        ).json()["documents"]
        bob_documents = bob.get("/api/v1/documents").json()["documents"]

        assert alice_documents[0]["file_id"] == alice_user_id
        assert bob_documents[0]["file_id"] == bob_user_id
        assert alice_documents != bob_documents

        assert alice.post("/api/v1/auth/logout").status_code == 204
        assert alice.get("/api/v1/documents").status_code == 401
        assert bob.get("/api/v1/documents").status_code == 200
    finally:
        app.dependency_overrides.clear()
