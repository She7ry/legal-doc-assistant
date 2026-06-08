from __future__ import annotations

from doc_assistant.schemas.citation import Citation
from doc_assistant.services.answer_guard import validate_answer


def _citation(source_id: str = "S1", preview: str = "Payment is due within 30 days.") -> Citation:
    return Citation(
        source_id=source_id,
        file_name="contract.pdf",
        preview=preview,
    )


def test_validate_answer_passes_when_citations_are_valid() -> None:
    answer = (
        "## 简短结论\n"
        "Payment is due within 30 days [S1]. Confidence: High.\n\n"
        "## 文档依据\n"
        "Section 3 requires payment within 30 days [S1]."
    )

    result = validate_answer(answer, [_citation()], has_retrieved_documents=True)

    assert result.passed is True
    assert result.confidence == "High"
    assert result.needs_repair is False


def test_validate_answer_flags_unknown_citation_ids() -> None:
    answer = "The notice period is 10 days [S9]."

    result = validate_answer(answer, [_citation()], has_retrieved_documents=True)

    assert result.passed is False
    assert any("S9" in issue for issue in result.issues)
    assert result.needs_repair is True


def test_validate_answer_flags_strong_legal_conclusions() -> None:
    answer = "This clause is invalid and you will definitely win [S1]."

    result = validate_answer(answer, [_citation()], has_retrieved_documents=True)

    assert result.passed is False
    assert any("strong legal conclusion" in issue for issue in result.issues)


def test_validate_answer_requires_refusal_without_retrieved_documents() -> None:
    answer = "The contract requires arbitration in New York."

    result = validate_answer(answer, [], has_retrieved_documents=False)

    assert result.passed is False
    assert any("without retrieved documents" in issue for issue in result.issues)
