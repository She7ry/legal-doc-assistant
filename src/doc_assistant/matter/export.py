from __future__ import annotations

from io import BytesIO
import json
import re
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from doc_assistant.matter.store import MatterArtifactRecord, MatterRecord


def artifact_markdown_filename(matter_id: str, artifact_id: str, version: int) -> str:
    matter_slug = _slug(matter_id) or "matter"
    artifact_slug = _slug(artifact_id) or "artifact"
    return f"{matter_slug}-{artifact_slug}-v{version}.md"


def artifact_docx_filename(matter_id: str, artifact_id: str, version: int) -> str:
    matter_slug = _slug(matter_id) or "matter"
    artifact_slug = _slug(artifact_id) or "artifact"
    return f"{matter_slug}-{artifact_slug}-v{version}.docx"


def artifact_bundle_filename(matter_id: str, export_format: str) -> str:
    matter_slug = _slug(matter_id) or "matter"
    format_slug = _slug(export_format) or "artifacts"
    return f"{matter_slug}-artifacts-{format_slug}.zip"


def render_artifact_markdown(
    *,
    matter: MatterRecord,
    artifact: MatterArtifactRecord,
) -> str:
    profile = matter.matter_profile
    lines = [
        f"# {artifact.title}",
        "",
        f"- Matter ID: {matter.matter_id}",
        f"- Artifact ID: {artifact.artifact_id}",
        f"- Artifact type: {artifact.artifact_type}",
        f"- Version: {artifact.version}",
        f"- Status: {artifact.status}",
        f"- Source task: {artifact.source_task_id}",
        f"- Updated: {artifact.updated_at.isoformat()}",
        "",
        "## Matter Profile",
        f"- Title: {matter.title}",
        f"- Document type: {_text(profile.get('document_type')) or 'Unknown'}",
        f"- Parties: {_format_value(profile.get('parties')) or 'Unspecified'}",
        f"- User side: {_text(profile.get('user_side')) or 'Unspecified'}",
        f"- Governing law: {_text(profile.get('governing_law')) or 'Unspecified'}",
        f"- Review scope: {_format_value(profile.get('review_scope')) or 'Unspecified'}",
        "",
        "## Summary",
        artifact.summary or "No summary.",
        "",
        "## Items",
    ]

    if artifact.items:
        for index, item in enumerate(artifact.items, start=1):
            lines.extend(_render_item(index, item))
    else:
        lines.append("- No structured items.")

    lines.extend(["", "## Citations"])
    if artifact.citations:
        for source_id in artifact.citations:
            lines.append(f"- {source_id}")
    else:
        lines.append("- No citations.")

    if artifact.source_finding_ids:
        lines.extend(["", "## Source Findings"])
        for finding_id in artifact.source_finding_ids:
            lines.append(f"- {finding_id}")

    if artifact.metadata:
        lines.extend(["", "## Metadata", "```json"])
        lines.append(json.dumps(artifact.metadata, ensure_ascii=False, indent=2))
        lines.append("```")

    return "\n".join(lines).strip() + "\n"


def render_artifacts_zip(
    *,
    matter: MatterRecord,
    artifacts: list[MatterArtifactRecord],
    export_format: str,
) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", _bundle_manifest(matter, artifacts, export_format))
        for artifact in artifacts:
            if export_format in {"markdown", "both"}:
                archive.writestr(
                    _bundle_path(
                        export_format=export_format,
                        filename=artifact_markdown_filename(
                            matter.matter_id,
                            artifact.artifact_id,
                            artifact.version,
                        ),
                        extension="md",
                    ),
                    render_artifact_markdown(matter=matter, artifact=artifact),
                )
            if export_format in {"docx", "both"}:
                archive.writestr(
                    _bundle_path(
                        export_format=export_format,
                        filename=artifact_docx_filename(
                            matter.matter_id,
                            artifact.artifact_id,
                            artifact.version,
                        ),
                        extension="docx",
                    ),
                    render_artifact_docx(matter=matter, artifact=artifact),
                )
    return buffer.getvalue()


def render_artifact_docx(
    *,
    matter: MatterRecord,
    artifact: MatterArtifactRecord,
) -> bytes:
    profile = matter.matter_profile
    body: list[str] = [
        _docx_paragraph(artifact.title, "Title"),
        _docx_paragraph(f"Matter ID: {matter.matter_id}"),
        _docx_paragraph(f"Artifact ID: {artifact.artifact_id}"),
        _docx_paragraph(f"Artifact type: {artifact.artifact_type}"),
        _docx_paragraph(f"Version: {artifact.version}"),
        _docx_paragraph(f"Status: {artifact.status}"),
        _docx_paragraph(f"Source task: {artifact.source_task_id}"),
        _docx_paragraph(f"Updated: {artifact.updated_at.isoformat()}"),
        _docx_paragraph("Matter Profile", "Heading1"),
        _docx_paragraph(f"Title: {matter.title}", "ListParagraph"),
        _docx_paragraph(
            f"Document type: {_text(profile.get('document_type')) or 'Unknown'}",
            "ListParagraph",
        ),
        _docx_paragraph(
            f"Parties: {_format_value(profile.get('parties')) or 'Unspecified'}",
            "ListParagraph",
        ),
        _docx_paragraph(
            f"User side: {_text(profile.get('user_side')) or 'Unspecified'}",
            "ListParagraph",
        ),
        _docx_paragraph(
            f"Governing law: {_text(profile.get('governing_law')) or 'Unspecified'}",
            "ListParagraph",
        ),
        _docx_paragraph(
            f"Review scope: {_format_value(profile.get('review_scope')) or 'Unspecified'}",
            "ListParagraph",
        ),
        _docx_paragraph("Summary", "Heading1"),
        _docx_paragraph(artifact.summary or "No summary."),
        _docx_paragraph("Items", "Heading1"),
    ]

    if artifact.items:
        for index, item in enumerate(artifact.items, start=1):
            body.extend(_render_docx_item(index, item))
    else:
        body.append(_docx_paragraph("No structured items.", "ListParagraph"))

    body.append(_docx_paragraph("Citations", "Heading1"))
    if artifact.citations:
        for source_id in artifact.citations:
            body.append(_docx_paragraph(source_id, "ListParagraph"))
    else:
        body.append(_docx_paragraph("No citations.", "ListParagraph"))

    if artifact.source_finding_ids:
        body.append(_docx_paragraph("Source Findings", "Heading1"))
        for finding_id in artifact.source_finding_ids:
            body.append(_docx_paragraph(finding_id, "ListParagraph"))

    if artifact.metadata:
        body.append(_docx_paragraph("Metadata", "Heading1"))
        body.append(
            _docx_paragraph(json.dumps(artifact.metadata, ensure_ascii=False, indent=2))
        )

    document_xml = _document_xml("\n".join(body))
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", _styles_xml())
        archive.writestr("word/_rels/document.xml.rels", _document_rels_xml())
    return buffer.getvalue()


def _bundle_manifest(
    matter: MatterRecord,
    artifacts: list[MatterArtifactRecord],
    export_format: str,
) -> str:
    return json.dumps(
        {
            "matter_id": matter.matter_id,
            "title": matter.title,
            "export_format": export_format,
            "artifact_count": len(artifacts),
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_type": artifact.artifact_type,
                    "title": artifact.title,
                    "version": artifact.version,
                    "status": artifact.status,
                    "updated_at": artifact.updated_at.isoformat(),
                }
                for artifact in artifacts
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _bundle_path(*, export_format: str, filename: str, extension: str) -> str:
    if export_format == "both":
        if extension == "md":
            return f"markdown/{filename}"
        return f"{extension}/{filename}"
    return filename


def _render_item(index: int, item: dict[str, Any]) -> list[str]:
    title = (
        _text(item.get("category"))
        or _text(item.get("question"))
        or _text(item.get("issue"))
        or _text(item.get("trigger"))
        or _text(item.get("deadline"))
        or _text(item.get("item_id"))
        or f"Item {index}"
    )
    lines = ["", f"### {index}. {title}"]
    for key, value in item.items():
        formatted = _format_value(value)
        if not formatted:
            continue
        lines.append(f"- {_label(key)}: {formatted}")
    return lines


def _render_docx_item(index: int, item: dict[str, Any]) -> list[str]:
    title = (
        _text(item.get("category"))
        or _text(item.get("question"))
        or _text(item.get("issue"))
        or _text(item.get("trigger"))
        or _text(item.get("deadline"))
        or _text(item.get("item_id"))
        or f"Item {index}"
    )
    paragraphs = [_docx_paragraph(f"{index}. {title}", "Heading2")]
    for key, value in item.items():
        formatted = _format_value(value)
        if formatted:
            paragraphs.append(_docx_paragraph(f"{_label(key)}: {formatted}", "ListParagraph"))
    return paragraphs


def _docx_paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{escape(style)}"/></w:pPr>' if style else ""
    runs = []
    parts = str(text).split("\n") or [""]
    for index, part in enumerate(parts):
        if index:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(f"<w:r><w:t xml:space=\"preserve\">{escape(part)}</w:t></w:r>")
    return f"<w:p>{style_xml}{''.join(runs)}</w:p>"


def _document_xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}"
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
        "</w:body></w:document>"
    )


def _content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )


def _document_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '<w:name w:val="Normal"/><w:qFormat/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title">'
        '<w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:after="240"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1">'
        '<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2">'
        '<w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="ListParagraph">'
        '<w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:ind w:left="360"/></w:pPr></w:style>'
        "</w:styles>"
    )


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            return ", ".join(_format_value(item) for item in value if _format_value(item))
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _label(key: str) -> str:
    text = key.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Field"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:80]
