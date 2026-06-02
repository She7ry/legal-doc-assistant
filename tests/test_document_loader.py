from __future__ import annotations

from types import SimpleNamespace

from doc_assistant.ingestion import document_loader


def test_save_uploaded_file_uses_tenant_directory_and_unique_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(document_loader, "settings", SimpleNamespace(upload_dir=tmp_path))

    first_path = document_loader.save_uploaded_file("Contract Copy.txt", b"one", tenant_id="acme")
    second_path = document_loader.save_uploaded_file("Contract Copy.txt", b"two", tenant_id="acme")

    assert first_path.parent == tmp_path / "acme"
    assert second_path.parent == tmp_path / "acme"
    assert first_path.name != second_path.name
    assert first_path.name.endswith("Contract_Copy.txt")
    assert first_path.read_bytes() == b"one"
    assert second_path.read_bytes() == b"two"
