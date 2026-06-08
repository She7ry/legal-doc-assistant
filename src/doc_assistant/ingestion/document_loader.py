from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import zipfile
from xml.etree import ElementTree
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from doc_assistant.config.settings import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}
INGEST_WARNINGS_METADATA_KEY = "ingest_warnings"
_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


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
        documents = _load_pdf_documents(path)
    elif suffix == ".docx":
        documents = _load_docx_documents(path)
    else:
        documents = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()

    for document in documents:
        document.metadata["file_name"] = path.name
        document.metadata["source"] = str(path)
        document.metadata["file_extension"] = suffix

    return documents


def _load_pdf_documents(path: Path) -> list[Document]:
    documents = PyPDFLoader(str(path)).load()
    empty_pages = []

    for index, document in enumerate(documents):
        page = document.metadata.get("page", index)
        text = document.page_content or ""
        document.metadata["char_count"] = len(text)
        if not text.strip():
            empty_pages.append(int(page) if isinstance(page, int) else index)

    warnings = []
    if empty_pages:
        page_labels = ", ".join(str(page + 1) for page in empty_pages[:10])
        suffix = "..." if len(empty_pages) > 10 else ""
        warnings.append(
            f"PDF pages with no extractable text: {page_labels}{suffix}. "
            "They may be scanned images or contain unsupported layout."
        )

    if empty_pages and settings.pdf_ocr_enabled:
        ocr_text_by_page, ocr_warnings = _ocr_pdf_pages(path, empty_pages)
        warnings.extend(ocr_warnings)
        for document in documents:
            page = document.metadata.get("page")
            if isinstance(page, int) and page in ocr_text_by_page:
                document.page_content = ocr_text_by_page[page]
                document.metadata["ocr_applied"] = True
                document.metadata["char_count"] = len(document.page_content)
    elif empty_pages:
        warnings.append(
            "OCR fallback is disabled. Set DOC_ASSISTANT_PDF_OCR_ENABLED=true "
            "and install OCR dependencies if scanned PDFs must be indexed."
        )

    if warnings and documents:
        _append_warnings(documents[0], warnings)

    return documents


def _ocr_pdf_pages(path: Path, pages: list[int]) -> tuple[dict[int, str], list[str]]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        return {}, [
            "OCR fallback was requested, but pdf2image and pytesseract are not installed."
        ]

    extracted: dict[int, str] = {}
    warnings = []
    for page in pages:
        try:
            images = convert_from_path(
                str(path),
                first_page=page + 1,
                last_page=page + 1,
            )
            if not images:
                warnings.append(f"OCR produced no image for PDF page {page + 1}.")
                continue
            text = pytesseract.image_to_string(images[0], lang=settings.pdf_ocr_lang).strip()
        except Exception as exc:
            logger.warning("PDF OCR failed", extra={"path": str(path), "page": page}, exc_info=True)
            warnings.append(f"OCR failed for PDF page {page + 1}: {exc}")
            continue

        if text:
            extracted[page] = text
        else:
            warnings.append(f"OCR produced no text for PDF page {page + 1}.")

    return extracted, warnings


def _load_docx_documents(path: Path) -> list[Document]:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("word/document.xml") as document_xml:
                root = ElementTree.parse(document_xml).getroot()
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("Invalid DOCX file. Expected a Word .docx package.") from exc

    parts = []
    body = root.find(f"{_WORD_NAMESPACE}body")
    if body is not None:
        for child in body:
            if child.tag == f"{_WORD_NAMESPACE}p":
                text = _paragraph_text(child)
                if text:
                    parts.append(text)
            elif child.tag == f"{_WORD_NAMESPACE}tbl":
                table_text = _table_text(child)
                if table_text:
                    parts.append(table_text)

    content = "\n\n".join(parts).strip()
    metadata = {"source": str(path), "file_name": path.name}
    document = Document(page_content=content, metadata=metadata)
    if not content:
        _append_warnings(document, ["DOCX file contained no extractable text."])

    return [document]


def _paragraph_text(element: ElementTree.Element) -> str:
    parts = []
    for node in element.iter():
        if node.tag == f"{_WORD_NAMESPACE}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_WORD_NAMESPACE}tab":
            parts.append("\t")
        elif node.tag in {f"{_WORD_NAMESPACE}br", f"{_WORD_NAMESPACE}cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def _table_text(element: ElementTree.Element) -> str:
    rows = []
    for row in element.iter(f"{_WORD_NAMESPACE}tr"):
        cells = []
        for cell in row.iter(f"{_WORD_NAMESPACE}tc"):
            text = " ".join(
                paragraph
                for paragraph in (_paragraph_text(item) for item in cell.iter(f"{_WORD_NAMESPACE}p"))
                if paragraph
            )
            cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def _append_warnings(document: Document, warnings: list[str]) -> None:
    existing = document.metadata.get(INGEST_WARNINGS_METADATA_KEY)
    if isinstance(existing, list):
        document.metadata[INGEST_WARNINGS_METADATA_KEY] = existing + warnings
    else:
        document.metadata[INGEST_WARNINGS_METADATA_KEY] = warnings
