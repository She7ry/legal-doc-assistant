from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from doc_assistant.config.settings import settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file_name(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum() or char in {".", "-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "uploaded_document"


def save_uploaded_file(file_name: str, content: bytes, tenant_id: str | None = None) -> Path:
    """Save raw bytes to the uploads directory and return the saved path."""
    safe_name = _safe_file_name(file_name)
    upload_dir = settings.upload_dir
    if tenant_id:
        upload_dir = upload_dir / _safe_file_name(tenant_id)

    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / f"{uuid4().hex}_{safe_name}"
    target_path.write_bytes(content)
    return target_path


def load_documents(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    else:
        documents = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()

    for document in documents:
        document.metadata["file_name"] = path.name
        document.metadata["source"] = str(path)

    return documents
