from __future__ import annotations

from types import SimpleNamespace
import zipfile

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


def test_markdown_extension_is_supported(tmp_path) -> None:
    path = tmp_path / "policy.markdown"
    path.write_text("# Policy\n\nUse written approval.", encoding="utf-8")

    documents = document_loader.load_documents(path)

    assert documents[0].page_content.startswith("# Policy")
    assert documents[0].metadata["file_extension"] == ".markdown"


def test_load_docx_extracts_paragraphs_and_tables(tmp_path) -> None:
    path = tmp_path / "contract.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Section 1 Term</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Party</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Acme</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    documents = document_loader.load_documents(path)

    assert "Section 1 Term" in documents[0].page_content
    assert "Party | Acme" in documents[0].page_content
    assert documents[0].metadata["file_extension"] == ".docx"


def test_load_docx_extracts_headers_footers_footnotes_and_alt_text(tmp_path) -> None:
    path = tmp_path / "contract.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
  <w:body>
    <w:p><w:r><w:t>Main agreement text.</w:t></w:r></w:p>
    <w:p><w:r><w:drawing><wp:docPr id="1" name="Logo" descr="Acme logo alt text"/></w:drawing></w:r></w:p>
  </w:body>
</w:document>
"""
    header_xml = """<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p><w:r><w:t>Confidential header</w:t></w:r></w:p>
</w:hdr>"""
    footer_xml = """<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p><w:r><w:t>Page footer</w:t></w:r></w:p>
</w:ftr>"""
    footnotes_xml = """<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:footnote><w:p><w:r><w:t>Footnote disclosure.</w:t></w:r></w:p></w:footnote>
</w:footnotes>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/header1.xml", header_xml)
        archive.writestr("word/footer1.xml", footer_xml)
        archive.writestr("word/footnotes.xml", footnotes_xml)

    documents = document_loader.load_documents(path)
    content = documents[0].page_content

    assert "Main agreement text." in content
    assert "Acme logo alt text" in content
    assert "Confidential header" in content
    assert "Page footer" in content
    assert "Footnote disclosure." in content
