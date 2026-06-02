from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import wrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import file_sha256, load_documents

FIXTURE_DIR = PROJECT_ROOT / "data" / "eval" / "fixtures"
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"
PDF_PATH = FIXTURE_DIR / "sample_supply_contract.pdf"


PAGES = [
    [
        "ACME SUPPLY AGREEMENT",
        "Document ID: EVAL-CONTRACT-001",
        "",
        "Section 1. Parties",
        "Buyer: Northwind Retail LLC. Supplier: Blue Harbor Components Ltd.",
        "",
        "Section 2. Delivery",
        "Supplier must deliver conforming goods to Buyer's Shanghai warehouse on or before the delivery date stated in each purchase order.",
        "A delivery is late if the goods arrive more than five business days after the applicable delivery date.",
        "",
        "Section 5. Payment",
        "Buyer must pay undisputed invoices within 30 calendar days after receiving a valid invoice.",
        "Disputed invoice amounts may be withheld until the dispute is resolved in writing by both parties.",
    ],
    [
        "ACME SUPPLY AGREEMENT",
        "Document ID: EVAL-CONTRACT-001",
        "",
        "Section 12.2 Liquidated Damages",
        "Marker: EVAL-C-12.2.",
        "If Supplier delivers goods more than five business days late, Supplier must pay liquidated damages equal to 0.5% of the delayed shipment value for each day of delay.",
        "Liquidated damages are capped at 10% of the delayed shipment value.",
        "Payment of liquidated damages does not limit Buyer's right to reject nonconforming goods.",
        "",
        "Section 14. Confidentiality",
        "Each party must protect the other party's confidential information using reasonable care and may use it only to perform this agreement.",
        "The confidentiality obligation survives for three years after termination.",
    ],
]


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    _write_simple_pdf(PDF_PATH, PAGES)
    gold_source = _find_gold_source(PDF_PATH, "EVAL-C-12.2")

    dataset = {
        "version": "0.1",
        "description": "Starter RAG evaluation dataset with one synthetic legal PDF.",
        "documents": [
            {
                "file_name": PDF_PATH.name,
                "path": str(PDF_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_id": file_sha256(PDF_PATH),
            }
        ],
        "cases": [
            {
                "id": "eval_001_liquidated_damages_cap",
                "question": "What is the cap on liquidated damages for late delivery?",
                "answer_type": "answerable",
                "gold_answer": "Liquidated damages are capped at 10% of the delayed shipment value.",
                "gold_sources": [gold_source],
                "required_answer_terms": ["10%", "delayed shipment value"],
                "forbidden_answer_terms": ["20%", "contract total value"],
            },
            {
                "id": "eval_002_cyber_insurance_refusal",
                "question": "Does the agreement require Supplier to buy cybersecurity insurance?",
                "answer_type": "unanswerable",
                "gold_answer": "The indexed document does not state a cybersecurity insurance requirement.",
                "gold_sources": [],
                "required_refusal_terms": [
                    "not found",
                    "not provided",
                    "cannot determine",
                    "relevant text was not found",
                    "do not contain",
                    "does not contain",
                    "do not specify",
                    "does not specify",
                    "do not mention",
                    "does not mention",
                ],
            },
        ],
    }

    DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {DATASET_PATH}")


def _find_gold_source(pdf_path: Path, marker: str) -> dict[str, object]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(load_documents(pdf_path))
    for index, chunk in enumerate(chunks):
        if marker in chunk.page_content:
            metadata = chunk.metadata or {}
            return {
                "file_name": pdf_path.name,
                "page": metadata.get("page"),
                "chunk_id": index,
                "marker": marker,
            }

    raise RuntimeError(f"Could not find marker {marker!r} in generated PDF chunks.")


def _write_simple_pdf(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = []
    page_object_numbers: list[int] = []
    font_object_number = 3 + len(pages) * 2

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")

    for page_index, lines in enumerate(pages):
        page_object_number = 3 + page_index * 2
        content_object_number = page_object_number + 1
        page_object_numbers.append(page_object_number)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("ascii")
        )
        stream = _page_stream(lines)
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode("ascii")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    _write_pdf_objects(path, objects)


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "50 750 Td"]
    first_line = True
    for raw_line in lines:
        wrapped_lines = wrap(raw_line, width=92) if raw_line else [""]
        for line in wrapped_lines:
            if first_line:
                first_line = False
            else:
                commands.append("0 -15 Td")
            commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    content.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


if __name__ == "__main__":
    main()
